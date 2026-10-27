import types

import pytest
import requests

from src import bootstrap, config
from src.llm import LLMError, OpenAICompatLLM


class FakeResp:
    def __init__(self, status, payload=None, headers=None):
        self.status_code, self._p, self.headers = status, payload or {}, headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return self._p


def test_retries_on_429_then_succeeds(monkeypatch):
    calls = []

    def fake_post(*a, **k):
        calls.append(1)
        if len(calls) < 3:
            return FakeResp(429, headers={"retry-after": "0"})
        return FakeResp(200, {"choices": [{"message": {"content": "hi"}}]})

    monkeypatch.setattr(requests, "post", fake_post)
    assert OpenAICompatLLM("m", "http://x", "k").chat([{"role": "user", "content": "x"}]) == "hi"
    assert len(calls) == 3


def test_http_error_becomes_llmerror(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResp(500))
    with pytest.raises(LLMError):
        OpenAICompatLLM("m", "http://x", "k").chat([{"role": "user", "content": "x"}])


def test_ensure_data_builds_db_and_index(tmp_path, monkeypatch):
    for name, val in {"PAPERS_DIR": tmp_path / "papers", "DB_PATH": tmp_path / "p.db", "INDEX_DIR": tmp_path / "idx"}.items():
        monkeypatch.setattr(config, name, val)

    def fake_download(progress=print, **k):
        config.PAPERS_DIR.mkdir(parents=True, exist_ok=True)
        (config.PAPERS_DIR / "a.txt").write_text("LoRA stands for Low-Rank Adaptation.")
        return []

    monkeypatch.setattr(bootstrap, "download_pdfs", fake_download)
    bootstrap.ensure_data(progress=lambda m: None, use_dense=False)
    assert config.DB_PATH.exists() and (config.INDEX_DIR / "chunks.json").exists()


def test_ensure_data_fails_clearly_without_documents(tmp_path, monkeypatch):
    for name, val in {"PAPERS_DIR": tmp_path / "papers", "DB_PATH": tmp_path / "p.db", "INDEX_DIR": tmp_path / "idx"}.items():
        monkeypatch.setattr(config, name, val)
    monkeypatch.setattr(bootstrap, "download_pdfs", lambda progress=print, **k: ["x"])
    with pytest.raises(RuntimeError):
        bootstrap.ensure_data(progress=lambda m: None, use_dense=False)
