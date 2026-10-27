"""Guardrails: the safety layer that sits AROUND the LLM.

Defence in depth - several independent layers, so one miss does not mean a breach:

  1. INPUT      regex screen for prompt-injection / prompt-leak / exfiltration / destructive intent
  2. RETRIEVAL  strip instructions hidden inside retrieved documents (indirect injection)
  3. PERMISSION each agent may only call its own tools
  4. LOOP/COST  step cap, tool-call cap, LLM-call cap, wall-clock timeout, repeat detection
  5. SQL        single read-only SELECT on whitelisted tables, row limit (+ read-only DB connection)
  6. OUTPUT     canary-leak check, secret redaction, citation verification

Regex screens are a first filter, not a complete defence - determined attackers can
paraphrase around them. That is why the later layers exist (see README -> Limitations).
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from dataclasses import dataclass, field

from . import config

ALLOWED_TABLES = {"papers"}
PERMISSIONS = {
    "researcher": {"retrieve_documents"},
    "analyst": {"sql_query", "calculator"},
    "writer": set(),  # the writer composes text only - no tools at all
}

_F = re.I | re.M
INPUT_PATTERNS = [
    ("injection", re.compile(r"\b(ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}\b(previous|prior|above|earlier|all|any|your|the|these)\b[^.\n]{0,30}\b(instructions?|prompts?|rules?|guidelines?|polic(?:y|ies)|guardrails?|restrictions?)\b", _F)),
    ("injection", re.compile(r"\b(you are now|from now on you|pretend (to be|you are)|act as if you (have|are)|developer[\s-]mode|dev[\s-]mode|jailbreak|do anything now)\b", _F)),
    ("injection", re.compile(r"^\s*#{0,3}\s*(system|assistant)\s*:", _F)),
    ("injection", re.compile(r"<\|?\s*(system|im_start|im_end|endoftext)", _F)),
    ("injection", re.compile(r"\b(disable|turn off|deactivate|remove)\b[^.\n]{0,20}\b(guardrails?|safety|filters?|restrictions?|safeguards?)\b", _F)),
    ("leak", re.compile(r"\b(reveal|show|print|repeat|display|output|tell me|leak|dump)\b[^.\n]{0,60}\b(system (prompt|instructions?|message)|initial prompt|hidden (prompt|instructions?)|your (instructions|prompt|rules)|internal (reference )?code|canary)\b", _F)),
    ("leak", re.compile(r"\brepeat\b[^.\n]{0,20}\b(everything|all|text|words)\b[^.\n]{0,20}\b(above|before)\b", _F)),
    ("exfil", re.compile(r"\b(print|show|reveal|display|send|dump|leak|output|cat|read)\b[^.\n]{0,50}(\.env\b|api[\s_-]?keys?|passwords?|secrets?|credentials?|tokens?)", _F)),
    ("exfil", re.compile(r"\b(send|post|upload|email|forward|exfiltrate|transmit)\b[^.\n]{0,60}(https?://|\b[\w.-]+@[\w.-]+\.\w+)", _F)),
    ("tool_abuse", re.compile(r"__\w+__|\bimport\s+os\b|\bsubprocess\b|\beval\s*\(|\bexec\s*\(|os\.system", _F)),
    ("destructive", re.compile(r"\b(drop|truncate|alter)\s+(table|database)\b|\bdelete\s+from\b|\bupdate\s+\w+\s+set\b|\binsert\s+into\b", _F)),
]
# Extra patterns that only make sense inside *documents* (text addressed to the AI):
INDIRECT_PATTERNS = INPUT_PATTERNS + [
    ("indirect", re.compile(r"\b(notice|note|message|instructions?)s?\s+(to|for)\s+(the\s+)?(ai|assistants?|llms?|language models?|chatbots?|agents?)\b", _F)),
]

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)\b(api[_-]?key|secret|password|token)\s*[:=]\s*\S+"),
]
CITE_RE = re.compile(r"\[([^\[\]\s]+\.(?:pdf|txt|md)) p\.(\d+)\]")
REFUSAL_RE = re.compile(r"could not find|cannot answer|can't answer|not (found|covered|available) in|no relevant", re.I)
FORBIDDEN_SQL = re.compile(r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach|pragma|vacuum|reindex|grant|revoke|load_extension)\b", re.I)
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
BLOCK_MESSAGE = "I can only answer questions about the indexed research papers, and I can't help with that request."


def _variants(text: str) -> list[str]:
    """Normalise unicode / zero-width tricks and also try a de-leetspeak variant."""
    t = unicodedata.normalize("NFKC", text)
    t = re.sub(r"[\u200b-\u200f\u2060\ufeff]", "", t)
    return [t, t.translate(LEET)]


def _first_match(patterns, text: str):
    for variant in _variants(text):
        for category, rx in patterns:
            if rx.search(variant):
                return category
    return None


@dataclass
class Decision:
    allowed: bool
    reason: str = ""
    category: str = ""
    payload: str | None = None


@dataclass
class OutputVerdict:
    answer: str
    blocked: bool = False
    flags: list = field(default_factory=list)


class Guardrails:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.events: list[dict] = []
        self.begin_request()

    # ----- limits (soft when enabled, generous hard caps when disabled) -----
    @property
    def step_limit(self):
        return config.MAX_STEPS if self.enabled else config.HARD_STEPS

    @property
    def max_llm_calls(self):
        return config.MAX_LLM_CALLS if self.enabled else config.HARD_LLM_CALLS

    @property
    def timeout_s(self):
        return config.TIMEOUT_S if self.enabled else config.HARD_TIMEOUT_S

    def begin_request(self):
        self.events = []
        self.llm_calls = 0
        self.tool_calls = 0
        self._seen: set = set()
        self._t0 = time.monotonic()

    def _log(self, layer: str, category: str, reason: str):
        self.events.append({"layer": layer, "category": category, "reason": reason})

    # ----- 1. input -----
    def check_input(self, text: str) -> Decision:
        if not self.enabled:
            return Decision(True)
        if len(text) > 2000:
            self._log("input", "length", "question longer than 2000 characters")
            return Decision(False, "question too long", "length")
        cat = _first_match(INPUT_PATTERNS, text)
        if cat:
            self._log("input", cat, f"matched {cat} pattern")
            return Decision(False, f"blocked ({cat})", cat)
        return Decision(True)

    # ----- 2. retrieved content (indirect injection) -----
    def sanitize_retrieved(self, text: str) -> tuple[str, bool]:
        if not self.enabled:
            return text, False
        kept, flagged = [], False
        for sent in re.split(r"(?<=[.!?\n])\s+", text):
            cat = _first_match(INDIRECT_PATTERNS, sent)
            if cat:
                flagged = True
                self._log("retrieval", "indirect_injection", "instruction-like sentence removed from document")
                kept.append("[removed: suspicious instruction found in document]")
            else:
                kept.append(sent)
        return " ".join(kept), flagged

    # ----- 3 & 4. permissions + loop/cost -----
    def check_tool_call(self, agent: str, tool: str, args) -> Decision:
        if not self.enabled:
            return Decision(True)
        if tool not in PERMISSIONS.get(agent, set()):
            self._log("permission", "permission", f"{agent} tried to call '{tool}'")
            return Decision(False, f"agent '{agent}' may not use tool '{tool}'", "permission")
        if not isinstance(args, dict):
            return Decision(False, "action_input must be a JSON object", "invalid_args")
        key = (agent, tool, json.dumps(args, sort_keys=True, default=str))
        if key in self._seen:
            self._log("loop", "loop", f"repeated identical call to {tool}")
            return Decision(False, "identical tool call repeated - stop looping", "loop")
        self._seen.add(key)
        self.tool_calls += 1
        if self.tool_calls > config.MAX_TOOL_CALLS:
            self._log("loop", "budget", "tool-call budget exceeded")
            return Decision(False, "tool-call budget exceeded", "budget")
        for v in args.values():
            if isinstance(v, str):
                cat = _first_match(INPUT_PATTERNS, v)
                if cat:
                    self._log("tool_args", cat, f"tool argument matched {cat} pattern")
                    return Decision(False, f"tool argument blocked ({cat})", cat)
        return Decision(True)

    def note_llm_call(self) -> Decision:
        self.llm_calls += 1
        if self.llm_calls > self.max_llm_calls:
            self._log("budget", "budget", "LLM-call budget exceeded")
            return Decision(False, "LLM-call budget exceeded", "budget")
        if time.monotonic() - self._t0 > self.timeout_s:
            self._log("budget", "timeout", "time budget exceeded")
            return Decision(False, "time budget exceeded", "timeout")
        return Decision(True)

    # ----- 5. SQL -----
    def validate_sql(self, sql: str) -> Decision:
        if not self.enabled:
            return Decision(True, payload=sql)
        s = sql.strip().rstrip(";").strip()

        def deny(reason):
            self._log("sql", "sql", reason)
            return Decision(False, reason, "sql")

        if ";" in s:
            return deny("multiple statements are not allowed")
        if "--" in s or "/*" in s:
            return deny("SQL comments are not allowed")
        if not re.match(r"^select\b", s, re.I):
            return deny("only SELECT statements are allowed")
        if FORBIDDEN_SQL.search(s):
            return deny("write/DDL keyword found")
        tables = re.findall(r"\b(?:from|join)\s+([A-Za-z_][\w.]*)", s, re.I)
        bad = [t for t in tables if t.lower() not in ALLOWED_TABLES]
        if bad:
            return deny(f"table not allowed: {bad[0]}")
        if not re.search(r"\blimit\b", s, re.I):
            s += " LIMIT 50"
        return Decision(True, payload=s)

    # ----- 6. output -----
    def check_output(self, answer: str, retrieved: set, used_docs: bool) -> OutputVerdict:
        if not self.enabled:
            return OutputVerdict(answer)
        if config.SYSTEM_CANARY.lower() in answer.lower():
            self._log("output", "leak", "system prompt canary found in answer")
            return OutputVerdict(BLOCK_MESSAGE, blocked=True, flags=["leak"])
        flags = []
        for rx in SECRET_PATTERNS:
            if rx.search(answer):
                answer = rx.sub("[REDACTED]", answer)
                flags.append("secret_redacted")
                self._log("output", "secret", "secret-like string redacted")
        cites = CITE_RE.findall(answer)
        valid = [(s, int(p)) for s, p in cites if (s, int(p)) in retrieved]
        for s, p in cites:
            if (s, int(p)) not in retrieved:
                answer = answer.replace(f"[{s} p.{p}]", "")
                flags.append("fabricated_citation")
                self._log("output", "citation", f"citation [{s} p.{p}] was not in retrieved context")
        if used_docs and retrieved and not valid and not REFUSAL_RE.search(answer):
            flags.append("ungrounded")
            self._log("output", "ungrounded", "answer has no valid citation")
        return OutputVerdict(answer.strip(), flags=flags)
