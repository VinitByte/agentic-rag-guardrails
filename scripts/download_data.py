"""Downloads the 10 arXiv PDFs (the document corpus) and builds the SQLite DB.

Run from the project root:   python -m scripts.download_data
Use --db-only to skip PDFs.
"""
import argparse

from scripts.build_db import build_db
from src import config
from src.bootstrap import download_pdfs

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-only", action="store_true")
    args = ap.parse_args()
    if not args.db_only:
        failed = download_pdfs(pause=3.0)
        if failed:
            print("Failed:", failed, "- download these manually from https://arxiv.org/abs/<id>")
    print("Built DB:", build_db(config.DB_PATH))
