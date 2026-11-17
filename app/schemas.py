"""Output contract shared by the model, the regex fallback and the API.

The system prompt and label vocabularies are loaded from ``configs/prompt_config.json``, the same file
that notebook 01 published with the dataset, so training, evaluation and serving cannot drift apart.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

PROMPT_CONFIG_PATH = Path(
    os.environ.get("PROMPT_CONFIG_PATH", Path(__file__).resolve().parent.parent / "configs" / "prompt_config.json")
)

PiiTag = Literal["PERSON", "EMAIL", "PHONE", "SSN", "CREDIT_CARD", "IP_ADDRESS", "API_KEY"]
Category = Literal[
    "Billing_and_Payments",
    "Technical_Support",
    "Security_and_Access",
    "Legal_and_Compliance",
    "General_Inquiry",
]
Urgency = Literal["Low", "Medium", "High"]

PII_TAGS: list[str] = list(PiiTag.__args__)
CATEGORIES: list[str] = list(Category.__args__)
URGENCY_LEVELS: list[str] = list(Urgency.__args__)

TAG_PATTERN = re.compile(r"\[(" + "|".join(PII_TAGS) + r")\]")
TAG_SPLIT = re.compile(r"(\[(?:" + "|".join(PII_TAGS) + r")\])")


@lru_cache(maxsize=1)
def load_prompt_config() -> dict:
    cfg = json.loads(PROMPT_CONFIG_PATH.read_text(encoding="utf-8"))
    if cfg["pii_tags"] != PII_TAGS or cfg["categories"] != CATEGORIES or cfg["urgency_levels"] != URGENCY_LEVELS:
        raise RuntimeError(f"{PROMPT_CONFIG_PATH} label vocabulary does not match app/schemas.py")
    return cfg


def ordered_types(tags) -> list[str]:
    """Canonical order used in training targets: PERSON, EMAIL, PHONE, SSN, CREDIT_CARD, IP_ADDRESS, API_KEY."""
    found = set(tags)
    return [t for t in PII_TAGS if t in found]


class TicketAnalysis(BaseModel):
    """The strict JSON object the model must return. Unknown keys are rejected."""

    model_config = ConfigDict(extra="forbid", strict=True)

    masked_text: str
    detected_pii_types: list[PiiTag]
    ticket_category: Category
    urgency_level: Urgency

    @field_validator("detected_pii_types")
    @classmethod
    def _unique_types(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("detected_pii_types must not contain duplicates")
        return v

    @model_validator(mode="after")
    def _tags_match_types(self) -> TicketAnalysis:
        tags_in_text = set(TAG_PATTERN.findall(self.masked_text))
        if tags_in_text != set(self.detected_pii_types):
            raise ValueError(
                f"tags in masked_text {sorted(tags_in_text)} != detected_pii_types {sorted(self.detected_pii_types)}"
            )
        return self


def extract_json(text: str) -> dict | None:
    """Strip code fences / surrounding chatter and parse the first balanced top-level JSON object."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def parse_model_output(raw: str) -> tuple[TicketAnalysis | None, str | None]:
    """Returns (analysis, error). Exactly one of them is None."""
    obj = extract_json(raw)
    if obj is None:
        return None, "unparseable_json"
    try:
        return TicketAnalysis.model_validate(obj), None
    except ValidationError as e:
        first = e.errors()[0]
        return None, f"schema_invalid: {first.get('loc')} {first.get('msg')}"[:200]


# What a single tag is allowed to stand for. Type-aware spans stop a tag from silently "absorbing" text the
# model dropped (e.g. "[EMAIL]" matching "jane@corp.com today").
TAG_SPAN_PATTERNS = {
    "PERSON": r"[^\n@\d\[\]]{1,60}?",
    "EMAIL": r"[^\s\[\]]{3,120}?",
    "PHONE": r"[0-9A-Za-z()+.\-/ \t]{5,40}?",
    "SSN": r"[0-9A-Za-z \-]{7,15}?",
    "CREDIT_CARD": r"[0-9 \-]{12,25}?",
    "IP_ADDRESS": r"[0-9A-Fa-f:.%/\[\]]{2,60}?",
    "API_KEY": r"[^\s]{8,256}?",
}


def is_faithful_masking(source: str, masked: str, max_tags: int = 60) -> bool:
    """True if ``masked`` equals ``source`` except that PII-shaped spans were replaced by their tags."""
    source, masked = source.strip(), masked.strip()
    parts = [p for p in TAG_SPLIT.split(masked) if p]
    tags = [p[1:-1] for p in parts if TAG_PATTERN.fullmatch(p)]
    if not tags:
        return masked == source
    if len(tags) > max_tags:
        return False
    pattern = "".join(f"(?:{TAG_SPAN_PATTERNS[p[1:-1]]})" if TAG_PATTERN.fullmatch(p) else re.escape(p) for p in parts)
    return re.fullmatch(pattern, source) is not None


# ----------------------------------------------------------------------------- API models

AnalysisSource = Literal["model", "model+regex_guard", "hybrid", "regex_fallback"]


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw ticket, e-mail or legal/HR snippet")

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class AnalysisMeta(BaseModel):
    source: AnalysisSource = Field(description="Which path produced the result")
    model: str
    latency_ms: float
    fallback_reason: str | None = None
    guard_added_types: list[PiiTag] = Field(
        default_factory=list, description="PII types the regex guard caught that the model missed"
    )


class AnalyzeResponse(BaseModel):
    result: TicketAnalysis
    meta: AnalysisMeta
