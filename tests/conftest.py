import json
import sys
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import inference_server  # noqa: E402
from app.inference_server import Settings  # noqa: E402


class FakeBackend:
    """Stands in for Ollama / llama.cpp. ``responder`` maps the user prompt to raw model text or raises."""

    name = "fake-model"

    def __init__(self, responder: Callable[[str], str]):
        self.responder = responder
        self.calls: list[str] = []

    async def generate(self, system: str, user: str, max_tokens: int) -> str:
        self.calls.append(user)
        return self.responder(user)

    async def healthy(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None


def ticket_from_prompt(user_prompt: str) -> str:
    return user_prompt.split("<ticket>\n", 1)[1].rsplit("\n</ticket>", 1)[0]


def as_json(masked: str, types: list[str], category: str = "General_Inquiry", urgency: str = "Low") -> str:
    return json.dumps(
        {"masked_text": masked, "detected_pii_types": types, "ticket_category": category, "urgency_level": urgency}
    )


@pytest.fixture
def make_client():
    clients = []

    def _make(responder: Callable[[str], str] | None = None, **settings_overrides) -> TestClient:
        client = TestClient(inference_server.app)
        client.__enter__()  # runs the lifespan
        settings = Settings(backend="rules_only", **settings_overrides)
        client.app.state.settings = settings
        client.app.state.backend = FakeBackend(responder) if responder else inference_server.build_backend(settings)
        clients.append(client)
        return client

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


def raise_timeout(_: str) -> str:
    raise httpx.ReadTimeout("simulated timeout")


def raise_connect(_: str) -> str:
    raise httpx.ConnectError("simulated connection refused")
