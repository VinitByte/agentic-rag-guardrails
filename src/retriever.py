"""Hybrid retriever: TF-IDF (keyword) + dense embeddings (semantic), fused with
Reciprocal Rank Fusion (RRF). If dense embeddings are disabled it is TF-IDF only."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config


class DenseEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # heavy import, load lazily
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        v = self.model.encode(texts, normalize_embeddings=True, batch_size=64,
                              show_progress_bar=len(texts) > 64)
        return np.asarray(v, dtype="float32")


def fit_tfidf(texts: list[str]):
    """Fit the keyword index. It is rebuilt from the chunk texts every time (takes ~1-3 s),
    so the saved index is just chunks.json (+ optional dense.npy): portable across machines
    and Python / scikit-learn versions, and nothing is unpickled."""
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=100_000, sublinear_tf=True)
    return tfidf, tfidf.fit_transform(texts)


class Retriever:
    def __init__(self, chunks, tfidf, tfidf_matrix, dense=None, meta=None):
        self.chunks, self.tfidf, self.tfidf_matrix = chunks, tfidf, tfidf_matrix
        self.dense, self.meta = dense, meta or {}
        self._embedder = None

    # ---------- build / load ----------
    @classmethod
    def build(cls, chunks: list[dict], index_dir, use_dense: bool | None = None) -> "Retriever":
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        texts = [c["text"] for c in chunks]
        tfidf, tm = fit_tfidf(texts)
        if use_dense is None:
            use_dense = config.EMBEDDER == "sentence-transformers"
        dense, meta = None, {"n_chunks": len(chunks), "dense": False}
        dense_path = index_dir / "dense.npy"
        if dense_path.exists():
            dense_path.unlink()
        if use_dense:
            try:
                dense = DenseEmbedder(config.EMBED_MODEL).encode(texts)
                np.save(dense_path, dense)
                meta.update(dense=True, embed_model=config.EMBED_MODEL)
            except ImportError:
                print("sentence-transformers is not installed -> using TF-IDF only "
                      "(pip install sentence-transformers for hybrid search)")
        (index_dir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
        (index_dir / "meta.json").write_text(json.dumps(meta))
        return cls(chunks, tfidf, tm, dense, meta)

    @classmethod
    def load(cls, index_dir) -> "Retriever":
        d = Path(index_dir)
        if not (d / "chunks.json").exists():
            raise FileNotFoundError(f"No index found in {d}. Run:  python -m src.ingest")
        chunks = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        tfidf, tm = fit_tfidf([c["text"] for c in chunks])
        meta_path = d / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        dense = None
        if (d / "dense.npy").exists() and importlib.util.find_spec("sentence_transformers") is not None:
            dense = np.load(d / "dense.npy")      # ignored (TF-IDF only) where sentence-transformers is not installed
        return cls(chunks, tfidf, tm, dense, meta)

    # ---------- search ----------
    def _get_embedder(self):
        if self._embedder is None:
            self._embedder = DenseEmbedder(self.meta.get("embed_model", config.EMBED_MODEL))
        return self._embedder

    def search(self, query: str, k: int = config.TOP_K, pool: int = 50) -> list[dict]:
        sparse = (self.tfidf_matrix @ self.tfidf.transform([query]).T).toarray().ravel()
        rankings = [np.argsort(-sparse)[:pool]]
        if self.dense is not None:
            qv = self._get_embedder().encode([query])[0]
            rankings.append(np.argsort(-(self.dense @ qv))[:pool])
        scores: dict[int, float] = {}
        for ranking in rankings:                      # Reciprocal Rank Fusion
            for rank, idx in enumerate(ranking):
                scores[int(idx)] = scores.get(int(idx), 0.0) + 1.0 / (60 + rank)
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [{**self.chunks[i], "score": round(s, 4)} for i, s in top]
