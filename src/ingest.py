"""Parse documents -> clean -> chunk (keeping source + page) -> build the search index.

    python -m src.ingest                     # index data/papers  -> index/
    python -m src.ingest --backend docling   # use IBM Docling for parsing
    python -m src.ingest --docs data/poisoned --index index_poisoned
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .retriever import Retriever


def clean_text(t: str) -> str:
    t = re.sub(r"-\n(?=[a-z])", "", t)       # join words hyphenated across lines
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def parse_pdf_pymupdf(path: Path) -> list[dict]:
    try:
        import pymupdf as fitz  # newer PyMuPDF name
    except ImportError:
        import fitz  # older name
    pages = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            txt = clean_text(page.get_text())
            if txt:
                pages.append({"page": i, "text": txt})
    return pages


def parse_pdf_docling(path: Path) -> list[dict]:
    """IBM Docling gives better structure (tables, headings) but is a heavy install.
    Simplification: whole document = one 'page', so citations read p.1."""
    from docling.document_converter import DocumentConverter
    result = DocumentConverter().convert(str(path))
    return [{"page": 1, "text": clean_text(result.document.export_to_markdown())}]


def parse_document(path: Path, backend: str = "pymupdf") -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf_docling(path) if backend == "docling" else parse_pdf_pymupdf(path)
    if suffix in {".txt", ".md"}:
        return [{"page": 1, "text": clean_text(path.read_text(encoding="utf-8", errors="ignore"))}]
    return []


def build_index(docs_dir, index_dir, backend="pymupdf", use_dense=None) -> Retriever:
    docs_dir = Path(docs_dir)
    splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE,
                                              chunk_overlap=config.CHUNK_OVERLAP)
    chunks: list[dict] = []
    for path in sorted(docs_dir.iterdir()):
        pages = parse_document(path, backend)
        if not pages:
            continue
        print(f"  parsed {path.name}: {len(pages)} page(s)")
        for pg in pages:
            for piece in splitter.split_text(pg["text"]):
                chunks.append({"id": len(chunks), "text": piece, "source": path.name, "page": pg["page"]})
    if not chunks:
        raise SystemExit(f"No .pdf/.txt/.md documents found in {docs_dir}. "
                         f"Run `python -m scripts.download_data` first.")
    print(f"Created {len(chunks)} chunks. Building index ...")
    return Retriever.build(chunks, index_dir, use_dense=use_dense)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=str(config.PAPERS_DIR))
    ap.add_argument("--index", default=str(config.INDEX_DIR))
    ap.add_argument("--backend", choices=["pymupdf", "docling"], default="pymupdf")
    ap.add_argument("--no-dense", action="store_true", help="TF-IDF only (no embedding model download)")
    a = ap.parse_args()
    build_index(a.docs, a.index, a.backend, use_dense=False if a.no_dense else None)
    print("Index saved to", a.index)


if __name__ == "__main__":
    main()
