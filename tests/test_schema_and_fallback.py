"""Unit tests for the output contract, the regex fallback/guard, and the API's degradation paths.

No model or GPU is needed: the model backend is replaced by a fake that returns canned outputs.
"""

import json

import pytest

from app.regex_fallback import analyze_with_rules, apply_guard, find_pii, luhn_valid, mask_pii, route_ticket
from app.schemas import (
    TAG_PATTERN,
    TicketAnalysis,
    extract_json,
    is_faithful_masking,
    load_prompt_config,
    parse_model_output,
)
from tests.conftest import as_json, raise_connect, raise_timeout, ticket_from_prompt

# ============================================================================ JSON parsing


class TestExtractJson:
    def test_plain_object(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_code_fence_and_chatter(self):
        raw = 'Sure! Here you go:\n```json\n{"masked_text": "hi [PERSON]", "x": [1, 2]}\n```\nAnything else?'
        assert extract_json(raw) == {"masked_text": "hi [PERSON]", "x": [1, 2]}

    def test_braces_and_escaped_quotes_inside_strings(self):
        raw = '{"masked_text": "he wrote \\"{oops}\\" and a \\\\ slash", "n": 1} trailing'
        assert extract_json(raw) == {"masked_text": 'he wrote "{oops}" and a \\ slash', "n": 1}

    @pytest.mark.parametrize("raw", ["", "no json here", '{"unterminated": ', "[1, 2, 3]", "{'single': 'quotes'}"])
    def test_rejects_non_objects(self, raw):
        assert extract_json(raw) is None


# ============================================================================ schema & tag integrity


VALID = {
    "masked_text": "Refund [CREDIT_CARD] for [PERSON], email [EMAIL].",
    "detected_pii_types": ["PERSON", "EMAIL", "CREDIT_CARD"],
    "ticket_category": "Billing_and_Payments",
    "urgency_level": "Medium",
}


class TestSchema:
    def test_valid_object(self):
        assert TicketAnalysis.model_validate(VALID).ticket_category == "Billing_and_Payments"

    def test_no_pii_is_valid(self):
        TicketAnalysis.model_validate({**VALID, "masked_text": "What are your hours?", "detected_pii_types": []})

    @pytest.mark.parametrize(
        "patch, fragment",
        [
            ({"notes": "extra key"}, "notes"),
            ({"ticket_category": "Sales"}, "ticket_category"),
            ({"urgency_level": "high"}, "urgency_level"),
            ({"detected_pii_types": ["PERSON", "PERSON", "EMAIL", "CREDIT_CARD"]}, "duplicates"),
            ({"detected_pii_types": ["PERSON", "EMAIL"]}, "detected_pii_types"),  # tag in text, not listed
            ({"detected_pii_types": ["PERSON", "EMAIL", "CREDIT_CARD", "SSN"]}, "SSN"),  # listed, not in text
            ({"detected_pii_types": ["NAME"]}, "detected_pii_types"),
            ({"detected_pii_types": "PERSON"}, "detected_pii_types"),
        ],
    )
    def test_rejections(self, patch, fragment):
        parsed, err = parse_model_output(json.dumps({**VALID, **patch}))
        assert parsed is None and err.startswith("schema_invalid")

    def test_missing_key(self):
        bad = {k: v for k, v in VALID.items() if k != "urgency_level"}
        assert parse_model_output(json.dumps(bad))[0] is None

    def test_unparseable(self):
        assert parse_model_output("I think this ticket is about billing.") == (None, "unparseable_json")

    def test_contract_file_matches_schema(self):
        cfg = load_prompt_config()
        assert "{ticket}" in cfg["user_template"]
        for tag in cfg["pii_tags"]:
            assert f"[{tag}]" in cfg["system_prompt"]


class TestFaithfulness:
    SRC = "Hi, I am Jane Doe, mail jane@corp.com today."

    def test_faithful(self):
        assert is_faithful_masking(self.SRC, "Hi, I am [PERSON], mail [EMAIL] today.")

    def test_unchanged_text_without_pii(self):
        assert is_faithful_masking("No PII here.", "No PII here.")

    def test_rephrased(self):
        assert not is_faithful_masking(self.SRC, "Hello, I am [PERSON], mail [EMAIL] today.")

    def test_dropped_sentence(self):
        assert not is_faithful_masking(self.SRC, "Hi, I am [PERSON].")

    def test_tag_cannot_absorb_dropped_words(self):
        assert not is_faithful_masking(self.SRC, "Hi, I am [PERSON], mail [EMAIL].")

    def test_multiline_ticket(self):
        src = "Hello,\nplease call 415-555-0132.\n\nThanks, Omar Ali"
        assert is_faithful_masking(src, "Hello,\nplease call [PHONE].\n\nThanks, [PERSON]")


# ============================================================================ regex detectors


class TestDetectors:
    @pytest.mark.parametrize(
        "number, ok",
        [("4111 1111 1111 1111", True), ("5555-5555-5555-4444", True), ("4111 1111 1111 1112", False), ("1234", False)],
    )
    def test_luhn(self, number, ok):
        assert luhn_valid(number) is ok

    @pytest.mark.parametrize(
        "text, tag",
        [
            ("contact jane.doe+billing@acme-corp.co.uk now", "EMAIL"),
            ("SSN on file 123-45-6789.", "SSN"),
            ("card 4111-1111-1111-1111 was charged", "CREDIT_CARD"),
            ("traffic from 203.0.113.42 at noon", "IP_ADDRESS"),
            ("ipv6 2001:db8:85a3::8a2e:370:7334 flagged", "IP_ADDRESS"),
            ("call me at (415) 555-0132 today", "PHONE"),
            ("call +44 20 7946 0958 please", "PHONE"),
            ("leaked key api_8CRydRSlkObzTNx6ou067rS8fRla5f in repo", "API_KEY"),
            ("Authorization: Bearer abcdEFGH1234ijklMNOP5678qrst", "API_KEY"),
            ("token ghp_" + "a1B2" * 9 + " exposed", "API_KEY"),
            ("Dear Mr. Okafor, thanks", "PERSON"),
            ("Hello, my name is Priya Sharma and I need help", "PERSON"),
        ],
    )
    def test_detects(self, text, tag):
        masked, types = mask_pii(text)
        assert types == [tag], (masked, types)
        assert f"[{tag}]" in masked

    @pytest.mark.parametrize(
        "text",
        [
            "Invoice INV-20931 is overdue by 30 days",
            "Version 4.2.1 crashes with error 0x80070057",
            "card 4111 1111 1111 1112 failed",  # Luhn-invalid
            "Our office is at 221 Baker Street",
            "Meeting on 2026-11-17 at 10:30",
            "IP 999.10.10.10 is not real",
            "See ticket #00012345",
        ],
    )
    def test_no_false_positives(self, text):
        assert mask_pii(text) == (text, [])

    def test_does_not_touch_existing_tags(self):
        text = "Ping [PERSON] at [EMAIL] or bob@corp.com"
        masked, types = mask_pii(text)
        assert masked == "Ping [PERSON] at [EMAIL] or [EMAIL]"
        assert types == ["EMAIL"]

    def test_card_beats_phone_on_overlap(self):
        spans = find_pii("pay with 4111 1111 1111 1111")
        assert [s.tag for s in spans] == ["CREDIT_CARD"]

    def test_email_inside_url_not_split(self):
        masked, _ = mask_pii("write to a.b@example.com.")
        assert masked == "write to [EMAIL]."


# ============================================================================ routing & full fallback


class TestRouting:
    @pytest.mark.parametrize(
        "text, category, urgency",
        [
            ("I was charged twice on my invoice, please refund", "Billing_and_Payments", "Low"),
            ("The VPN keeps crashing with a timeout error", "Technical_Support", "Medium"),
            ("URGENT: unauthorized login attempts, account compromised", "Security_and_Access", "High"),
            ("We received a subpoena, legal needs records", "Legal_and_Compliance", "High"),
            ("What are your office hours?", "General_Inquiry", "Low"),
        ],
    )
    def test_route(self, text, category, urgency):
        assert route_ticket(text) == (category, urgency)

    def test_security_with_secret_is_never_low(self):
        assert route_ticket("please revoke the token", ["API_KEY"]) == ("Security_and_Access", "Medium")

    def test_fallback_always_schema_valid(self):
        text = "URGENT refund to 4111 1111 1111 1111 for Mr. Smith, mail smith@corp.com, SSN 123-45-6789"
        result = analyze_with_rules(text)
        TicketAnalysis.model_validate(result.model_dump())
        assert result.detected_pii_types == ["PERSON", "EMAIL", "SSN", "CREDIT_CARD"]
        assert "4111" not in result.masked_text and "smith@corp.com" not in result.masked_text
        assert result.urgency_level == "High"


class TestGuard:
    def test_guard_catches_missed_email(self):
        model_out = TicketAnalysis(
            masked_text="[PERSON] wrote from jane@corp.com",
            detected_pii_types=["PERSON"],
            ticket_category="General_Inquiry",
            urgency_level="Low",
        )
        guarded, added = apply_guard(model_out)
        assert guarded.masked_text == "[PERSON] wrote from [EMAIL]"
        assert guarded.detected_pii_types == ["PERSON", "EMAIL"]
        assert added == ["EMAIL"]

    def test_guard_noop_on_clean_output(self):
        clean = TicketAnalysis.model_validate(VALID)
        assert apply_guard(clean) == (clean, [])


# ============================================================================ API: happy path and degradation


TICKET = "Hi, I'm Jane Doe (jane@corp.com). I was charged twice, please refund."
GOOD_MASKED = "Hi, I'm [PERSON] ([EMAIL]). I was charged twice, please refund."


class TestApi:
    def test_model_path(self, make_client):
        client = make_client(lambda p: as_json(GOOD_MASKED, ["PERSON", "EMAIL"], "Billing_and_Payments", "Medium"))
        r = client.post("/v1/analyze", json={"text": TICKET})
        assert r.status_code == 200
        body = r.json()
        assert body["meta"]["source"] == "model"
        assert body["meta"]["fallback_reason"] is None
        assert body["result"]["masked_text"] == GOOD_MASKED
        assert client.app.state.backend.calls and ticket_from_prompt(client.app.state.backend.calls[0]) == TICKET

    def test_code_fenced_output_is_accepted(self, make_client):
        fenced = "```json\n" + as_json(GOOD_MASKED, ["PERSON", "EMAIL"], "Billing_and_Payments") + "\n```"
        r = make_client(lambda p: fenced).post("/v1/analyze", json={"text": TICKET})
        assert r.json()["meta"]["source"] == "model"

    def test_guard_masks_leak(self, make_client):
        leaky = "Hi, I'm [PERSON] (jane@corp.com). I was charged twice, please refund."
        r = make_client(lambda p: as_json(leaky, ["PERSON"], "Billing_and_Payments")).post(
            "/v1/analyze", json={"text": TICKET}
        )
        body = r.json()
        assert body["meta"]["source"] == "model+regex_guard"
        assert body["meta"]["guard_added_types"] == ["EMAIL"]
        assert "jane@corp.com" not in body["result"]["masked_text"]
        assert body["result"]["detected_pii_types"] == ["PERSON", "EMAIL"]

    def test_guard_can_be_disabled(self, make_client):
        leaky = "Hi, I'm [PERSON] (jane@corp.com). I was charged twice, please refund."
        client = make_client(lambda p: as_json(leaky, ["PERSON"]), enable_regex_guard=False)
        assert client.post("/v1/analyze", json={"text": TICKET}).json()["meta"]["source"] == "model"

    def test_rephrased_output_becomes_hybrid(self, make_client):
        rewritten = as_json("Customer [PERSON] wants a refund.", ["PERSON"], "Billing_and_Payments", "Medium")
        body = make_client(lambda p: rewritten).post("/v1/analyze", json={"text": TICKET}).json()
        assert body["meta"]["source"] == "hybrid"
        assert body["meta"]["fallback_reason"] == "unfaithful_masked_text"
        assert body["result"]["ticket_category"] == "Billing_and_Payments"  # routing kept from the model
        assert body["result"]["masked_text"] == GOOD_MASKED  # masking rebuilt from source

    @pytest.mark.parametrize(
        "responder, reason_prefix",
        [
            (lambda p: "Sorry, I can't help with that.", "unparseable_json"),
            (lambda p: '{"masked_text": "x", "detected_pii_types": []}', "schema_invalid"),
            (lambda p: as_json(GOOD_MASKED, ["PERSON"]), "schema_invalid"),  # tag/type mismatch
            (raise_timeout, "model_timeout"),
            (raise_connect, "model_unavailable"),
        ],
    )
    def test_fallback_paths(self, make_client, responder, reason_prefix):
        body = make_client(responder).post("/v1/analyze", json={"text": TICKET}).json()
        assert body["meta"]["source"] == "regex_fallback"
        assert body["meta"]["fallback_reason"].startswith(reason_prefix)
        assert "jane@corp.com" not in body["result"]["masked_text"]
        assert body["result"]["ticket_category"] == "Billing_and_Payments"
        TicketAnalysis.model_validate(body["result"])

    def test_rules_only_backend(self, make_client):
        body = make_client().post("/v1/analyze", json={"text": TICKET}).json()
        assert body["meta"]["source"] == "regex_fallback"
        assert body["meta"]["model"] == "rules_only"
        assert body["meta"]["fallback_reason"] == "rules_only_mode"

    @pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": "   "}, {"text": 123}])
    def test_rejects_bad_input(self, make_client, payload):
        assert make_client().post("/v1/analyze", json=payload).status_code == 422

    def test_rejects_oversized_input(self, make_client):
        client = make_client(max_input_chars=50)
        assert client.post("/v1/analyze", json={"text": "x" * 51}).status_code == 413

    def test_health(self, make_client):
        body = make_client(lambda p: "{}").get("/health").json()
        assert body == {"status": "ok", "backend": "fake-model", "model_reachable": True, "note": None}

    def test_every_response_tag_is_known(self, make_client):
        body = make_client().post("/v1/analyze", json={"text": TICKET + " Card 4111 1111 1111 1111."}).json()
        for tag in TAG_PATTERN.findall(body["result"]["masked_text"]):
            assert tag in body["result"]["detected_pii_types"]


# ============================================================================ deployment artefacts


def test_modelfile_system_prompt_matches_contract():
    import re
    from pathlib import Path

    modelfile = (Path(__file__).resolve().parent.parent / "Modelfile").read_text(encoding="utf-8")
    system = re.search(r'^SYSTEM """(.*?)"""', modelfile, flags=re.DOTALL | re.MULTILINE).group(1)
    assert system == load_prompt_config()["system_prompt"]
    assert "PARAMETER temperature 0" in modelfile and "PARAMETER repeat_penalty 1.0" in modelfile
