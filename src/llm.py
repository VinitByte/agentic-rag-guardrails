"""LLM backends behind one tiny interface:  llm.chat(messages, json_mode) -> str

Providers: ollama (local), openai (any OpenAI-compatible API), watsonx, mock (tests only).
"""
from __future__ import annotations

import re
import time

import requests

from . import config


class LLMError(RuntimeError):
    pass


class BaseLLM:
    name = "base"

    def chat(self, messages: list[dict], json_mode: bool = False) -> str:  # pragma: no cover
        raise NotImplementedError


class OllamaLLM(BaseLLM):
    name = "ollama"

    def __init__(self, model: str, host: str):
        self.model, self.host = model, host.rstrip("/")

    def chat(self, messages, json_mode=False):
        payload = {"model": self.model, "messages": messages, "stream": False,
                   "options": {"temperature": 0}}
        if json_mode:
            payload["format"] = "json"
        try:
            r = requests.post(f"{self.host}/api/chat", json=payload, timeout=300)
            r.raise_for_status()
        except requests.RequestException as e:
            raise LLMError(f"Ollama request failed ({e}). Is `ollama serve` running and the model pulled?") from e
        return r.json()["message"]["content"]


class OpenAICompatLLM(BaseLLM):
    name = "openai"

    def __init__(self, model: str, base_url: str, api_key: str):
        self.model, self.base_url, self.api_key = model, base_url.rstrip("/"), api_key

    def chat(self, messages, json_mode=False):
        payload = {"model": self.model, "messages": messages, "temperature": 0}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            attempts = 0
            while True:
                attempts += 1
                r = requests.post(f"{self.base_url}/chat/completions", json=payload,
                                  headers=headers, timeout=300)
                if r.status_code == 429 and attempts < 4:      # free tiers rate-limit: wait and retry
                    try:
                        wait = float(r.headers.get("retry-after", 2 * attempts))
                    except ValueError:
                        wait = 2 * attempts
                    time.sleep(min(wait, 20))
                    continue
                if r.status_code == 400 and "response_format" in payload:
                    payload.pop("response_format")            # model has no JSON mode -> rely on the prompt;
                    continue                                  # agents.extract_json() copes with plain text
                r.raise_for_status()
                break
        except requests.RequestException as e:
            detail = ""
            resp = getattr(e, "response", None)      # show WHY the API refused (e.g. model_not_found)
            if resp is not None:
                try:
                    detail = " | " + resp.text[:300]
                except Exception:  # noqa: BLE001
                    pass
            raise LLMError(f"LLM API request failed: {e}{detail}") from e
        return r.json()["choices"][0]["message"]["content"]


class WatsonxLLM(BaseLLM):
    """IBM watsonx.ai. Needs `pip install ibm-watsonx-ai`.
    NOTE: SDK/model names change over time - check the current watsonx.ai docs and
    your project's model list if this class raises an error."""
    name = "watsonx"

    def __init__(self, model_id: str):
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference

        creds = Credentials(url=config.WATSONX_URL, api_key=config.WATSONX_API_KEY)
        self.model = ModelInference(model_id=model_id, credentials=creds,
                                    project_id=config.WATSONX_PROJECT_ID)

    def chat(self, messages, json_mode=False):
        resp = self.model.chat(messages=messages)
        return resp["choices"][0]["message"]["content"]


class MockLLM(BaseLLM):
    """Deterministic rule-based stand-in used by unit tests and offline smoke runs.

    It is NOT a real model. It deliberately behaves like a *naive* model: it obeys
    "ignore previous instructions" and runs whatever SQL it is asked to, so we can
    test that the guardrails (not the model) are what stops the attack.
    Never report metrics produced with this class.
    """
    name = "mock"

    def chat(self, messages, json_mode=False):
        system = messages[0]["content"]
        m = re.search(r"You are the (\w+)", system)
        if not m:               # e.g. a plain connection test
            return "OK"
        role = m.group(1).lower()
        users = [m["content"] for m in messages if m["role"] == "user"]
        if role == "router":
            return self._router(users[0])
        if role == "researcher":
            return self._researcher(users)
        if role == "analyst":
            return self._analyst(users)
        return self._writer(users[0])

    @staticmethod
    def _j(**kw):
        import json
        return json.dumps(kw)

    def _router(self, q):
        ql = q.lower()
        sql_words = ["how many", "count", "list the titles", "oldest", "database", "who wrote",
                     "sql", "drop table", "delete from", "update papers", "sqlite_master"]
        use_sql = any(w in ql for w in sql_words)
        return self._j(use_docs=not ("how many" in ql or "oldest" in ql), use_sql=use_sql)

    def _researcher(self, users):
        task = users[0].replace("Task: ", "", 1)
        obs = [u for u in users if u.startswith("Observation")]
        if not obs:
            return self._j(thought="search", action="retrieve_documents", action_input={"query": task[:200]})
        text = obs[-1]
        if re.search(r"ignore all previous instructions", text, re.I):  # naive: obeys poisoned doc
            return self._j(final_answer="HACKED")
        m = re.search(r'<document cite="(\[[^\]]+\])">\s*(.*?)\s*</document>', text, re.S)
        if not m:
            return self._j(final_answer=config.REFUSAL_TEXT)
        return self._j(final_answer=f"{m.group(2)[:300]} {m.group(1)}")

    def _analyst(self, users):
        task = users[0].replace("Task: ", "", 1)
        obs = [u for u in users if u.startswith("Observation")]
        if obs:
            return self._j(final_answer="Query result: " + obs[-1][:300])
        tl = task.lower()
        if "drop table" in tl:
            sql = "DROP TABLE papers"
        elif "delete from" in tl:
            sql = "DELETE FROM papers WHERE year < 2022"
        elif "update papers" in tl:
            sql = "UPDATE papers SET year = 1999"
        elif "sqlite_master" in tl:
            sql = "SELECT sql FROM sqlite_master"
        elif "how many" in tl:
            y = re.search(r"(19|20)\d\d", task)
            sql = f"SELECT COUNT(*) AS n FROM papers WHERE year = {y.group(0)}" if y else "SELECT COUNT(*) AS n FROM papers"
        else:
            sql = "SELECT title, year FROM papers ORDER BY year LIMIT 5"
        return self._j(thought="query", action="sql_query", action_input={"query": sql})

    def _writer(self, user):
        if re.search(r"ignore (all )?previous instructions|disregard your rules", user, re.I):
            return "HACKED " + config.SYSTEM_CANARY  # naive model: obeys + leaks
        parts = re.findall(r"<(?:document|data)_findings>\s*(.*?)\s*</(?:document|data)_findings>", user, re.S)
        body = " ".join(p for p in parts if p.strip())
        return body[:600] or config.REFUSAL_TEXT


def get_llm(provider: str | None = None) -> BaseLLM:
    provider = (provider or config.LLM_PROVIDER).lower()
    if provider == "ollama":
        return OllamaLLM(config.LLM_MODEL, config.OLLAMA_HOST)
    if provider == "openai":
        return OpenAICompatLLM(config.LLM_MODEL, config.OPENAI_BASE_URL, config.OPENAI_API_KEY)
    if provider == "watsonx":
        return WatsonxLLM(config.LLM_MODEL)
    if provider == "mock":
        return MockLLM()
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'")
