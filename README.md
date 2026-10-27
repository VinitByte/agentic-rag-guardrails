# 🛡️ Agentic RAG Assistant with Safety Guardrails

[![tests](https://github.com/<your-username>/agentic-rag-guardrails/actions/workflows/tests.yml/badge.svg)](https://github.com/<your-username>/agentic-rag-guardrails/actions)

### 👉 **[Live demo](https://YOUR-APP-NAME.streamlit.app)** &nbsp;|&nbsp; [Notebook](notebooks/Agentic_RAG_Guardrails.ipynb)

<!-- Add a screenshot or GIF of the app here:  ![demo](docs/demo.png) -->

A multi-agent question-answering system over **10 research papers (PDF)** and a **SQL database**, wrapped in a
layered safety system that defends against prompt injection, system-prompt leakage, destructive SQL, tool abuse and
*indirect* injection hidden inside documents.

The point is not "a chatbot that reads PDFs" - it is **measuring how an AI agent fails and how much each guardrail
helps** (attack success rate with guardrails OFF vs ON). In the live demo you can fire real attacks at the agent and
compare both modes side by side.

## Architecture

```
user question
     |
[1 INPUT GUARD] --blocked--> safe refusal
     |
  ROUTER (LLM) -> docs / database / both
     |                    |
 RESEARCHER            ANALYST
 retrieve_documents    sql_query (read-only), calculator
     |   [2 retrieval sanitiser]  [3 permissions]  [4 loop/cost caps]  [5 SQL guard]
     v                    v
            WRITER (LLM, no tools)
                  |
        [6 OUTPUT GUARD: canary leak, secret redaction, citation check]
                  |
          answer + sources + trace
```

| Layer | Stops | Code |
|---|---|---|
| 1 Input guard | injection, jailbreaks, prompt-leak requests, exfiltration, destructive SQL intent (regex + unicode / leet-speak normalisation) | `guardrails.check_input` |
| 2 Retrieval sanitiser | **indirect injection**: instructions planted inside a document | `guardrails.sanitize_retrieved` |
| 3 Permissions | an agent calling a tool it was not given | `PERMISSIONS` |
| 4 Loop / cost caps | infinite loops, runaway cost (step, tool-call, LLM-call caps, timeout, repeat detection) | `check_tool_call`, `note_llm_call` |
| 5 SQL guard | one `SELECT` on whitelisted tables only; DB also opened **read-only** | `validate_sql`, `tools.make_sql_tool` |
| 6 Output guard | system-prompt **canary** leak, secret redaction, fabricated citations | `check_output` |

Design notes: framework-free agent loop (every guardrail hook is visible and unit-tested; text splitting uses
`langchain-text-splitters`); hybrid TF-IDF + dense retrieval fused with Reciprocal Rank Fusion (dense part optional);
defence in depth - tests show the output guard still catches a leak when the input filter is bypassed, and the
read-only connection still protects the DB when the SQL validator is bypassed.

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-local.txt
cp .env.example .env                                     # then choose your LLM (see below)

python -m scripts.download_data                          # 10 PDFs + papers.db
python -m src.ingest                                     # search index   (--no-dense = TF-IDF only)
python -m src.ingest --docs data/poisoned --index index_poisoned --no-dense

streamlit run streamlit_app.py                           # web UI
python -m src.app ask "What does LoRA stand for?" --trace
pytest -q                                                # 60 tests, no LLM needed
```

**LLM options** (set in `.env`): local **Ollama** (`ollama pull llama3.1:8b`), any **OpenAI-compatible API**
(OpenAI, Groq free tier, ...), or **IBM watsonx.ai**. Prefer notebooks? Use
[`notebooks/Agentic_RAG_Guardrails.ipynb`](notebooks/Agentic_RAG_Guardrails.ipynb) - the whole project in one file.

## Deploy your own live demo
* **Render** (free): **[DEPLOY_RENDER.md](DEPLOY_RENDER.md)** - upload to GitHub by hand, then deploy as a Render web service.
* **Streamlit Community Cloud** (free): **[DEPLOY.md](DEPLOY.md)**.

Both use a free Groq API key as the LLM.

## Evaluation

```bash
python -m eval.run_eval qa         # accuracy by type, retrieval hit@5, latency
python -m eval.run_eval attacks    # attack success rate: guardrails OFF vs ON
```

20 QA questions (document / SQL / mixed / out-of-scope) and 25 attacks in 9 categories (incl. 2 indirect-injection
attacks through a poisoned document). **Starter sets - extend to 40+ items before quoting numbers.**

| Metric | Result |
|---|---|
| QA accuracy (guardrails ON) | _ / 20 |
| Retrieval hit@5 | _ % |
| Attack success rate - guardrails OFF | _ / 25 |
| Attack success rate - guardrails ON | _ / 25 |

*(fill in with your own measured results)*

## Limitations
* Regex screening is a first filter; paraphrased attacks can slip past it (that is why layers 2-6 exist) and it can cause false positives.
* The keyword/regex judge may mark correct paraphrases wrong; use manual review or an LLM-as-judge for stronger evaluation.
* 25 attacks measure a defined threat model - they do not prove safety.
* Small local models follow the JSON action format less reliably than large ones.

## Project structure
```
streamlit_app.py     web UI (guardrails ON/OFF/compare)     notebooks/   all-in-one Jupyter notebook
src/                 config, llm, retriever, ingest, tools, guardrails, agents, pipeline, bootstrap, app (CLI)
scripts/             build_db.py, download_data.py          eval/        qa_set, attack_set, run_eval
data/poisoned/       fictional document with a hidden injection
tests/               60 unit + end-to-end tests             .github/     CI workflow (runs pytest)
```

## License
MIT
