import importlib.util

import numpy as np

from src.retriever import Retriever

CHUNKS = [
    {"id": 0, "text": "LoRA stands for Low-Rank Adaptation and freezes the pre-trained weights.", "source": "lora.txt", "page": 1},
    {"id": 1, "text": "Retrieval-augmented generation combines parametric and non-parametric memory.", "source": "rag.txt", "page": 1},
    {"id": 2, "text": "The Transformer uses eight attention heads in the base model.", "source": "attn.txt", "page": 1},
]


def test_build_load_roundtrip_has_no_pickle_files(tmp_path):
    built = Retriever.build(CHUNKS, tmp_path, use_dense=False)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["chunks.json", "meta.json"]
    loaded = Retriever.load(tmp_path)
    q = "What does LoRA stand for?"
    assert [h["id"] for h in loaded.search(q, k=3)] == [h["id"] for h in built.search(q, k=3)]
    assert loaded.search(q, k=1)[0]["source"] == "lora.txt"


def test_load_works_without_meta_json(tmp_path):
    Retriever.build(CHUNKS, tmp_path, use_dense=False)
    (tmp_path / "meta.json").unlink()
    assert Retriever.load(tmp_path).search("attention heads", k=1)[0]["source"] == "attn.txt"


def test_dense_file_ignored_when_sentence_transformers_missing(tmp_path, monkeypatch):
    Retriever.build(CHUNKS, tmp_path, use_dense=False)
    np.save(tmp_path / "dense.npy", np.zeros((3, 4), dtype="float32"))   # e.g. built on another machine
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a, **k: None if n == "sentence_transformers" else real(n, *a, **k))
    r = Retriever.load(tmp_path)
    assert r.dense is None and r.search("LoRA", k=1)[0]["source"] == "lora.txt"
