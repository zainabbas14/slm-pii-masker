"""Deterministic PII masking and keyword routing.

Used in two ways by the API:

* **Guard**: after the model answers, ``mask_pii`` runs over the model's ``masked_text`` and catches any
  high-confidence PII the model missed (emails, SSNs, Luhn-valid cards, IPs, API keys, phones).
* **Fallback**: if the model is unreachable or its output fails validation, ``analyze_with_rules`` produces a
  schema-valid answer on its own, so the API never returns unmasked text.

Regex cannot find names reliably. Only names introduced by an honorific or a phrase such as "my name is"
are masked here, so PERSON recall in fallback mode is deliberately conservative.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass

from app.schemas import TAG_PATTERN, TicketAnalysis, ordered_types

# ----------------------------------------------------------------------------- detectors


@dataclass(frozen=True)
class Match:
    start: int
    end: int
    tag: str
    priority: int  # lower wins when spans overlap


def luhn_valid(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


_API_KEY_RE = re.compile(
    r"""(?<![A-Za-z0-9_-])(?:
        (?:api|svc|tok|ent|key)[_-][A-Za-z0-9]{20,}      # generic prefixed tokens
      | sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}             # sk-... style secret keys
      | gh[pousr]_[A-Za-z0-9]{36,}                       # GitHub tokens
      | github_pat_[A-Za-z0-9_]{40,}
      | AKIA[0-9A-Z]{16}                                  # AWS access key id
      | AIza[0-9A-Za-z_-]{35}                             # Google API key
      | xox[abposr]-[A-Za-z0-9-]{10,}                     # Slack tokens
      | hf_[A-Za-z0-9]{30,}                               # Hugging Face tokens
      | eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}   # JWT
    )(?![A-Za-z0-9_-])""",
    re.VERBOSE,
)
_BEARER_RE = re.compile(r"(?i)\b(?:bearer|api[_ -]?key|token|secret)\s*[:=]?\s+([A-Za-z0-9_\-.]{24,})")
_EMAIL_RE = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?![\w-])")
_SSN_RE = re.compile(r"(?<![\d-])(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?![\d-])")
_CARD_RE = re.compile(r"(?<![\d-])\d(?:[ -]?\d){12,18}(?![\d-])")
_IPV4_RE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
_IPV6_RE = re.compile(r"(?<![0-9A-Fa-f:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![0-9A-Fa-f:])")
_PHONE_RE = re.compile(
    r"(?<![\w+])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?|\d{2,4}[\s.-])\d{2,4}[\s.-]?\d{3,4}(?:[\s.-]?\d{2,4})?(?![\w])"
)
_PERSON_RE = re.compile(
    r"""(?:
        \b(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s+
      | \b(?:[Mm]y\ name\ is|[Tt]his\ is|[Ii]\ am|[Ii]'m)\s+(?=[A-Z][a-z])
    )
    (?P<name>[A-Z][a-z]+(?:[ -][A-Z][a-z]+){0,2})""",
    re.VERBOSE,
)

_PRIORITY = {"API_KEY": 0, "EMAIL": 1, "IP_ADDRESS": 2, "CREDIT_CARD": 3, "SSN": 4, "PHONE": 5, "PERSON": 6}


def _valid_ip(candidate: str) -> bool:
    try:
        ipaddress.ip_address(candidate)
        return True
    except ValueError:
        return False


def find_pii(text: str) -> list[Match]:
    """All high-confidence PII spans, overlaps resolved by priority then length."""
    found: list[Match] = []

    def add(start: int, end: int, tag: str) -> None:
        found.append(Match(start, end, tag, _PRIORITY[tag]))

    for m in _API_KEY_RE.finditer(text):
        add(m.start(), m.end(), "API_KEY")
    for m in _BEARER_RE.finditer(text):
        add(m.start(1), m.end(1), "API_KEY")
    for m in _EMAIL_RE.finditer(text):
        add(m.start(), m.end(), "EMAIL")
    for m in _IPV4_RE.finditer(text):
        if _valid_ip(m.group()):
            add(m.start(), m.end(), "IP_ADDRESS")
    for m in _IPV6_RE.finditer(text):
        if m.group().count(":") >= 2 and _valid_ip(m.group()) and re.search(r"[0-9A-Fa-f]{2}", m.group()):
            add(m.start(), m.end(), "IP_ADDRESS")
    for m in _CARD_RE.finditer(text):
        if luhn_valid(m.group()):
            add(m.start(), m.end(), "CREDIT_CARD")
    for m in _SSN_RE.finditer(text):
        add(m.start(), m.end(), "SSN")
    for m in _PHONE_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 7 <= len(digits) <= 15:
            add(m.start(), m.end(), "PHONE")
    for m in _PERSON_RE.finditer(text):
        add(m.start("name"), m.end("name"), "PERSON")

    # Never touch characters inside tags that are already present.
    protected = [(m.start(), m.end()) for m in TAG_PATTERN.finditer(text)]
    found = [f for f in found if not any(f.start < e and s < f.end for s, e in protected)]

    found.sort(key=lambda f: (f.priority, -(f.end - f.start), f.start))
    chosen: list[Match] = []
    for f in found:
        if all(f.end <= c.start or f.start >= c.end for c in chosen):
            chosen.append(f)
    return sorted(chosen, key=lambda f: f.start)


def mask_pii(text: str) -> tuple[str, list[str]]:
    """Replace every detected span with its tag. Returns (masked_text, newly_masked_types)."""
    matches = find_pii(text)
    out, cursor = [], 0
    for m in matches:
        out.append(text[cursor : m.start])
        out.append(f"[{m.tag}]")
        cursor = m.end
    out.append(text[cursor:])
    return "".join(out), ordered_types(m.tag for m in matches)


# ----------------------------------------------------------------------------- keyword router

CATEGORY_LEXICON: dict[str, dict[str, int]] = {
    "Billing_and_Payments": {
        "invoice": 3,
        "billing": 3,
        "billed": 3,
        "payment": 3,
        "refund": 3,
        "charged": 3,
        "chargeback": 3,
        "overcharged": 3,
        "reimbursement": 3,
        "credit card": 2,
        "subscription": 2,
        "pricing": 2,
        "overdue": 2,
        "receipt": 2,
        "transaction": 2,
        "payroll": 2,
        "balance": 2,
        "wire transfer": 2,
        "debit": 2,
        "fee": 1,
        "purchase": 1,
        "price": 1,
        "paid": 1,
        "pay": 1,
    },
    "Technical_Support": {
        "error": 3,
        "bug": 3,
        "crash": 3,
        "crashes": 3,
        "crashed": 3,
        "not working": 3,
        "outage": 3,
        "troubleshoot": 3,
        "down": 2,
        "server": 2,
        "install": 2,
        "software": 2,
        "network": 2,
        "latency": 2,
        "timeout": 2,
        "timing out": 2,
        "configuration": 2,
        "configuring": 2,
        "database": 2,
        "deployment": 2,
        "upgrade": 2,
        "reboot": 2,
        "browser": 2,
        "laptop": 2,
        "printer": 2,
        "vpn": 2,
        "firmware": 2,
        "webhook": 2,
        "api": 1,
        "app": 1,
        "device": 1,
        "website": 1,
    },
    "Security_and_Access": {
        "password": 3,
        "unauthorized": 3,
        "breach": 3,
        "compromised": 3,
        "phishing": 3,
        "credentials": 3,
        "api key": 3,
        "2fa": 3,
        "mfa": 3,
        "locked out": 3,
        "suspicious": 3,
        "malware": 3,
        "hacked": 3,
        "revoke": 3,
        "leaked": 3,
        "security": 2,
        "access": 2,
        "permission": 2,
        "permissions": 2,
        "token": 2,
        "login": 2,
        "log in": 2,
        "authentication": 2,
        "firewall": 2,
        "sso": 1,
    },
    "Legal_and_Compliance": {
        "gdpr": 3,
        "compliance": 3,
        "legal": 3,
        "lawsuit": 3,
        "court": 3,
        "attorney": 3,
        "lawyer": 3,
        "subpoena": 3,
        "hipaa": 3,
        "nda": 3,
        "litigation": 3,
        "harassment": 3,
        "grievance": 3,
        "counsel": 3,
        "contract": 2,
        "agreement": 2,
        "regulation": 2,
        "regulatory": 2,
        "audit": 2,
        "consent": 2,
        "liability": 2,
        "clause": 2,
        "dispute": 2,
        "termination": 2,
        "disciplinary": 2,
        "settlement": 2,
        "policy": 1,
        "privacy": 1,
        "terms": 1,
    },
}
_CATEGORY_PRIORITY = ["Security_and_Access", "Legal_and_Compliance", "Billing_and_Payments", "Technical_Support"]
URGENCY_HIGH = [
    "urgent",
    "immediately",
    "asap",
    "emergency",
    "critical",
    "breach",
    "compromised",
    "fraud",
    "unauthorized",
    "lawsuit",
    "subpoena",
    "outage",
    "data leak",
    "as soon as possible",
    "within 24 hours",
    "right away",
    "escalate",
    "production down",
    "system down",
    "hacked",
    "legal action",
    "final notice",
]
URGENCY_MEDIUM = [
    "soon",
    "delay",
    "delayed",
    "issue",
    "problem",
    "error",
    "failed",
    "failure",
    "overdue",
    "follow up",
    "follow-up",
    "reminder",
    "pending",
    "deadline",
    "not working",
    "complaint",
    "incorrect",
    "disputed",
    "this week",
    "unable to",
]


def _phrase_re(phrase: str) -> re.Pattern:
    return re.compile(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])")


_COMPILED = {cat: [(_phrase_re(k), w) for k, w in lex.items()] for cat, lex in CATEGORY_LEXICON.items()}
_HIGH = [_phrase_re(p) for p in URGENCY_HIGH]
_MEDIUM = [_phrase_re(p) for p in URGENCY_MEDIUM]


def route_ticket(text: str, pii_types: list[str] | None = None) -> tuple[str, str]:
    """Keyword routing. Returns (ticket_category, urgency_level)."""
    t = TAG_PATTERN.sub(" ", text).lower()
    scores = {cat: sum(w * len(p.findall(t)) for p, w in pats) for cat, pats in _COMPILED.items()}
    best = min(scores, key=lambda c: (-scores[c], _CATEGORY_PRIORITY.index(c)))
    category = best if scores[best] >= 3 else "General_Inquiry"

    if any(p.search(t) for p in _HIGH):
        urgency = "High"
    elif any(p.search(t) for p in _MEDIUM):
        urgency = "Medium"
    else:
        urgency = "Low"
    exposes_secret = bool({"SSN", "CREDIT_CARD", "API_KEY"} & set(pii_types or []))
    if category == "Security_and_Access" and exposes_secret and urgency == "Low":
        urgency = "Medium"
    return category, urgency


def analyze_with_rules(text: str) -> TicketAnalysis:
    """Fully deterministic analysis. Always returns a schema-valid result."""
    masked, _ = mask_pii(text)
    types = ordered_types(TAG_PATTERN.findall(masked))
    category, urgency = route_ticket(text, types)
    return TicketAnalysis(masked_text=masked, detected_pii_types=types, ticket_category=category, urgency_level=urgency)


def apply_guard(analysis: TicketAnalysis) -> tuple[TicketAnalysis, list[str]]:
    """Re-scan the model's masked_text for PII it missed. Returns (analysis, types_added_by_guard)."""
    masked, newly = mask_pii(analysis.masked_text)
    if not newly:
        return analysis, []
    types = ordered_types(set(analysis.detected_pii_types) | set(TAG_PATTERN.findall(masked)))
    guarded = TicketAnalysis(
        masked_text=masked,
        detected_pii_types=types,
        ticket_category=analysis.ticket_category,
        urgency_level=analysis.urgency_level,
    )
    return guarded, newly
