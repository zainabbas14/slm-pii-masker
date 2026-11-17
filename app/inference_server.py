"""FastAPI microservice for the fine-tuned PII masker.

Request flow::

    text ─► model (Ollama / llama.cpp) ─► parse + Pydantic validation
                │                           ├─ valid & faithful ─► regex guard ─► "model" / "model+regex_guard"
                │ error / timeout           ├─ valid, rephrased ─► regex masking + model routing ─► "hybrid"
                ▼                           └─ invalid ─────────► "regex_fallback"
          regex_fallback

The API never returns text that has not passed through at least one masking layer. Raw ticket text is
never logged.

Run:  uvicorn app.inference_server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Literal, Protocol

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.regex_fallback import analyze_with_rules, apply_guard, mask_pii
from app.schemas import (
    TAG_PATTERN,
    AnalysisMeta,
    AnalyzeRequest,
    AnalyzeResponse,
    TicketAnalysis,
    is_faithful_masking,
    load_prompt_config,
    ordered_types,
    parse_model_output,
)

logger = logging.getLogger("pii_masker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    backend: Literal["ollama", "llamacpp", "rules_only"] = "ollama"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "pii-masker"
    ollama_format: Literal["schema", "json", "none"] = "schema"
    llamacpp_url: str = "http://localhost:8080"
    request_timeout_s: float = 60.0
    max_input_chars: int = 6000
    num_ctx: int = 4096
    enable_regex_guard: bool = True


# ----------------------------------------------------------------------------- model backends


class ModelBackend(Protocol):
    name: str

    async def generate(self, system: str, user: str, max_tokens: int) -> str: ...

    async def healthy(self) -> bool: ...

    async def aclose(self) -> None: ...


class OllamaBackend:
    def __init__(self, s: Settings):
        self.name = f"ollama:{s.ollama_model}"
        self._s = s
        self._client = httpx.AsyncClient(base_url=s.ollama_host, timeout=s.request_timeout_s)

    async def generate(self, system: str, user: str, max_tokens: int) -> str:
        payload = {
            "model": self._s.ollama_model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": 0, "num_ctx": self._s.num_ctx, "num_predict": max_tokens},
        }
        if self._s.ollama_format == "schema":
            payload["format"] = TicketAnalysis.model_json_schema()
        elif self._s.ollama_format == "json":
            payload["format"] = "json"
        r = await self._client.post("/api/chat", json=payload)
        r.raise_for_status()
        return r.json()["message"]["content"]

    async def healthy(self) -> bool:
        try:
            r = await self._client.get("/api/tags", timeout=3)
            wanted = self._s.ollama_model.split(":")[0]
            names = [m.get("name", "").split(":")[0] for m in r.json().get("models", [])]
            return r.status_code == 200 and wanted in names
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()


class LlamaCppBackend:
    """llama.cpp ``llama-server`` via its OpenAI-compatible endpoint."""

    def __init__(self, s: Settings):
        self.name = "llama.cpp"
        self._client = httpx.AsyncClient(base_url=s.llamacpp_url, timeout=s.request_timeout_s)

    async def generate(self, system: str, user: str, max_tokens: int) -> str:
        payload = {
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        r = await self._client.post("/v1/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    async def healthy(self) -> bool:
        try:
            r = await self._client.get("/health", timeout=3)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()


class NoModelBackend:
    """Rules-only mode: every request uses the deterministic fallback."""

    name = "rules_only"

    async def generate(self, system: str, user: str, max_tokens: int) -> str:
        raise RuntimeError("model backend disabled (BACKEND=rules_only)")

    async def healthy(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None


def build_backend(s: Settings) -> ModelBackend:
    if s.backend == "ollama":
        return OllamaBackend(s)
    if s.backend == "llamacpp":
        return LlamaCppBackend(s)
    return NoModelBackend()


# ----------------------------------------------------------------------------- core pipeline


async def analyze(text: str, backend: ModelBackend, settings: Settings) -> AnalyzeResponse:
    cfg = load_prompt_config()
    t0 = time.perf_counter()
    user_msg = cfg["user_template"].format(ticket=text)
    max_tokens = min(2048, len(text) // 2 + 256)  # output ≈ input length; tags are shorter than the values

    raw: str | None = None
    reason: str | None = None
    try:
        if isinstance(backend, NoModelBackend):
            reason = "rules_only_mode"
        else:
            raw = await backend.generate(cfg["system_prompt"], user_msg, max_tokens)
    except httpx.TimeoutException:
        reason = "model_timeout"
    except httpx.HTTPError as e:
        reason = f"model_unavailable: {type(e).__name__}"
    except Exception as e:  # noqa: BLE001 - any backend failure must degrade to the fallback
        reason = f"model_error: {type(e).__name__}"

    analysis: TicketAnalysis | None = None
    if raw is not None:
        analysis, reason = parse_model_output(raw)

    added: list[str] = []
    if analysis is None:
        result = analyze_with_rules(text)
        source = "regex_fallback"
    elif not is_faithful_masking(text, analysis.masked_text):
        # The model rewrote the text. Keep its routing, but rebuild masked_text deterministically from the source.
        masked, _ = mask_pii(text)
        result = TicketAnalysis(
            masked_text=masked,
            detected_pii_types=ordered_types(TAG_PATTERN.findall(masked)),
            ticket_category=analysis.ticket_category,
            urgency_level=analysis.urgency_level,
        )
        source, reason = "hybrid", "unfaithful_masked_text"
    else:
        result, source = analysis, "model"
        if settings.enable_regex_guard:
            result, added = apply_guard(analysis)
            if added:
                source = "model+regex_guard"

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    # Log metadata only. Never the ticket text.
    logger.info(
        "analyze source=%s reason=%s chars=%d types=%s latency_ms=%.1f",
        source,
        reason,
        len(text),
        ",".join(result.detected_pii_types) or "-",
        latency_ms,
    )
    return AnalyzeResponse(
        result=result,
        meta=AnalysisMeta(
            source=source,
            model=backend.name,
            latency_ms=latency_ms,
            fallback_reason=reason if source in ("regex_fallback", "hybrid") else None,
            guard_added_types=added,
        ),
    )


# ----------------------------------------------------------------------------- FastAPI app


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    load_prompt_config()  # fail fast if the contract file is missing or inconsistent
    app.state.settings = settings
    app.state.backend = build_backend(settings)
    logger.info("backend=%s guard=%s", app.state.backend.name, settings.enable_regex_guard)
    try:
        yield
    finally:
        await app.state.backend.aclose()


app = FastAPI(
    title="Enterprise PII Masker & Ticket Categorizer",
    version="1.0.0",
    description=(
        "Fine-tuned Qwen3-4B (QLoRA, GGUF Q4_K_M) behind Pydantic validation and a deterministic regex safety net."
    ),
    lifespan=lifespan,
)


@app.get("/health")
async def health(request: Request):
    model_ok = await request.app.state.backend.healthy()
    return {
        "status": "ok" if model_ok else "degraded",
        "backend": request.app.state.backend.name,
        "model_reachable": model_ok,
        "note": None if model_ok else "Requests are served by the regex fallback until the model is reachable.",
    }


@app.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(body: AnalyzeRequest, request: Request) -> AnalyzeResponse:
    settings: Settings = request.app.state.settings
    if len(body.text) > settings.max_input_chars:
        raise HTTPException(status_code=413, detail=f"text exceeds {settings.max_input_chars} characters")
    return await analyze(body.text, request.app.state.backend, settings)


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception):
    logger.exception("unhandled error: %s", type(exc).__name__)
    return JSONResponse(status_code=500, content={"detail": "internal error"})
