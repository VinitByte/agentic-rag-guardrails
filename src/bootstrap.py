"""First-run setup used by the deployed app: make sure the DB, PDFs and search indexes exist.

On Streamlit Community Cloud the file system starts empty, so the app downloads the 10 public
arXiv PDFs and builds the indexes the first time it starts (about 1-2 minutes).
"""
import time

import requests

from scripts.build_db import PAPERS, build_db

from . import config
from .ingest import build_index


def download_pdfs(progress=print, pause: float = 1.0, timeout: int = 60) -> list:
    """Download the arXiv PDFs that are missing. Returns the ids that failed."""
    config.PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "agentic-rag-student-project (educational use)"}
    failed = []
    for i, (arxiv_id, title, *_rest) in enumerate(PAPERS, start=1):
        out = config.PAPERS_DIR / f"{arxiv_id}.pdf"
        if out.exists() and out.stat().st_size > 10_000:
            continue
        progress(f"Downloading paper {i}/{len(PAPERS)}: {title}")
        try:
            r = requests.get(f"https://arxiv.org/pdf/{arxiv_id}.pdf", headers=headers, timeout=timeout)
            r.raise_for_status()
            out.write_bytes(r.content)
            time.sleep(pause)  # be polite to arXiv
        except Exception as e:  # noqa: BLE001 - keep going with the papers we do have
            failed.append(arxiv_id)
            progress(f"  could not download {arxiv_id}: {e}")
    return failed


def _has_documents(folder) -> bool:
    return folder.exists() and any(p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md"} for p in folder.iterdir())


def ensure_data(progress=print, use_dense=None):
    """Create papers.db and the main search index if they do not exist yet."""
    if not config.DB_PATH.exists():
        build_db(config.DB_PATH)
    if not (config.INDEX_DIR / "chunks.json").exists():
        download_pdfs(progress)
        if not _has_documents(config.PAPERS_DIR):
            raise RuntimeError("No documents available: the arXiv download failed. "
                               "Check the server's internet access or commit the PDFs / a pre-built index.")
        progress("Building the search index ...")
        build_index(config.PAPERS_DIR, config.INDEX_DIR, use_dense=use_dense)
    return config.INDEX_DIR


def ensure_poisoned_index():
    """Tiny index of the fictional 'poisoned' report (for the indirect-injection demo)."""
    if not (config.POISONED_INDEX_DIR / "chunks.json").exists():
        build_index(config.POISONED_DIR, config.POISONED_INDEX_DIR, use_dense=False)
    return config.POISONED_INDEX_DIR
