"""Builds the small SQLite database the Analyst agent queries.

The metadata below describes the 10 arXiv papers in the corpus (public facts:
title, first author, year of first arXiv submission) plus a coarse topic label.
"""
import argparse
import sqlite3
from pathlib import Path

from src import config

PAPERS = [
    # arxiv_id,     title,                                                                       first_author, year, topic
    ("1706.03762", "Attention Is All You Need",                                                   "Vaswani",  2017, "architecture"),
    ("1810.04805", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "Devlin", 2018, "pretraining"),
    ("2005.14165", "Language Models are Few-Shot Learners",                                       "Brown",    2020, "language models"),
    ("2005.11401", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",            "Lewis",    2020, "retrieval"),
    ("2106.09685", "LoRA: Low-Rank Adaptation of Large Language Models",                          "Hu",       2021, "fine-tuning"),
    ("2201.11903", "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",       "Wei",      2022, "prompting"),
    ("2210.03629", "ReAct: Synergizing Reasoning and Acting in Language Models",                  "Yao",      2022, "agents"),
    ("2302.04761", "Toolformer: Language Models Can Teach Themselves to Use Tools",               "Schick",   2023, "agents"),
    ("2307.09288", "Llama 2: Open Foundation and Fine-Tuned Chat Models",                         "Touvron",  2023, "language models"),
    ("2310.11511", "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection", "Asai",  2023, "retrieval"),
]

SCHEMA = """
CREATE TABLE papers (
    id            INTEGER PRIMARY KEY,
    arxiv_id      TEXT NOT NULL,
    title         TEXT NOT NULL,
    first_author  TEXT NOT NULL,
    year          INTEGER NOT NULL,
    topic         TEXT NOT NULL
);
"""


def build_db(path=config.DB_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.executemany(
        "INSERT INTO papers (arxiv_id, title, first_author, year, topic) VALUES (?,?,?,?,?)",
        PAPERS,
    )
    con.commit()
    con.close()
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(config.DB_PATH))
    args = ap.parse_args()
    print("Built", build_db(args.db))
