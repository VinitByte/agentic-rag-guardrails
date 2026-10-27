# Deploy on Render (code uploaded to GitHub by hand)

Render runs your Streamlit app as a **Web Service** straight from your GitHub repo and gives you a public link like
`https://agentic-rag-guardrails.onrender.com`. The language model is a free **Groq** API key.

**Free-plan facts you must plan for:** 512 MB RAM; the service **sleeps after 15 minutes without visitors** (the next
visit wakes it, ~1 minute); and the disk is **temporary** - it is wiped on every restart/redeploy. That is why you will
upload a small **pre-built search index** to GitHub: the app then starts in seconds and never has to download PDFs
or parse them in 512 MB of RAM.

---
## Part A - Build the index on your PC (once)
Inside the project folder (venv active, `pip install -r requirements.txt` done):
```bash
python -m scripts.download_data        # downloads 10 PDFs and creates data/papers.db
python -m src.ingest --no-dense        # creates index/chunks.json and index/meta.json
```
Check that these three files exist (each should be well under 25 MB):
`data/papers.db`, `index/chunks.json`, `index/meta.json`.

## Part B - Get a free Groq key
https://console.groq.com -> **API Keys** -> **Create API Key** -> copy the `gsk_...` key (shown once).
Pick a current model name at https://console.groq.com/docs/models (names change; default below is `llama-3.3-70b-versatile`).

## Part C - Upload to GitHub by hand
1. github.com -> **New repository** -> name `agentic-rag-guardrails` -> **Public** -> create (no README/licence).
2. Click **uploading an existing file** (or *Add file -> Upload files*).
3. Open the project folder in File Explorer, select **the contents** (not the folder itself, not the zip) and drag them in.
   The browser accepts **max 100 files and 25 MB per file per upload**, so do 2-3 batches and click **Commit changes** after each:
   * Batch 1: `streamlit_app.py`, `requirements.txt`, `render.yaml`, `README.md`, `DEPLOY_RENDER.md`, `LICENSE`, `.env.example`, folders `src/`, `scripts/`
   * Batch 2: folders `data/` (must contain `papers.db` and the `poisoned/` folder), `index/` (only `chunks.json` + `meta.json`), `eval/`, `tests/`, `notebooks/`
   * Optional (dot-folders - drag them if your PC shows them): `.streamlit/config.toml`, `.github/`

   **Do NOT upload:** `.env`, `.streamlit/secrets.toml`, `.venv/`, `__pycache__/`, the PDFs in `data/papers/`, `index/dense.npy`, `agentic_rag/`, `results/`.
4. **Check the repo page.** `streamlit_app.py` and `requirements.txt` must be visible **at the top level** of the repo.
   If you only see one folder (e.g. `agentic-rag-guardrails/`), everything went one level too deep - either re-upload the contents, or set
   **Root Directory** to that folder name in Render (Part D).
5. Open `index/chunks.json` and `data/papers.db` on GitHub to confirm they are there. Search the repo for `gsk_` - it must find nothing.

*(Editing a file later: open it on GitHub -> pencil icon -> edit -> Commit. Uploading a file with the same name overwrites it. Render redeploys automatically after each commit.)*

## Part D - Create the service on Render
1. https://render.com -> sign up with **GitHub** -> authorise access to your repo.
2. **New +** -> **Web Service** -> select `agentic-rag-guardrails` -> Connect.
3. Fill in:
   | Field | Value |
   |---|---|
   | Name | `agentic-rag-guardrails` (becomes part of your URL) |
   | Region | closest to you (e.g. Singapore) |
   | Branch | `main` |
   | Root Directory | *(leave empty)* |
   | Runtime / Language | **Python 3** |
   | Build Command | `pip install -r requirements.txt` |
   | Start Command | `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true` |
   | Instance Type | **Free** |
4. **Environment Variables** -> add each one:
   | Key | Value |
   |---|---|
   | `PYTHON_VERSION` | `3.11.9` (if Render rejects it, use the version it suggests) |
   | `LLM_PROVIDER` | `openai` |
   | `LLM_MODEL` | `llama-3.3-70b-versatile` (or a current Groq model) |
   | `OPENAI_BASE_URL` | `https://api.groq.com/openai/v1` |
   | `OPENAI_API_KEY` | your `gsk_...` key |
   | `EMBEDDER` | `tfidf` |
   | `DAILY_LIMIT` | `150` |
   | `SESSION_LIMIT` | `12` |
   | `REPO_URL` | `https://github.com/<your-username>/agentic-rag-guardrails` |
5. (Under Advanced) **Health Check Path:** `/_stcore/health`
6. **Create Web Service.** Watch the **Logs**: install (~3-5 min) -> "Your service is live". Open the `https://....onrender.com` link.

*Shortcut:* **New + -> Blueprint** reads `render.yaml` and pre-fills all of the above; you only type `OPENAI_API_KEY` and `REPO_URL`.

## Part E - Finish the portfolio
* Edit `README.md` on GitHub: replace `YOUR-APP-NAME` / `<your-username>` with your real live link and username.
* Repo **About** (gear icon): add the live link as *Website*, plus topics (`rag`, `llm-agents`, `ai-safety`, `prompt-injection`).
* Resume / LinkedIn: `Live demo: <link>` and `GitHub: <repo link>`.
* Before an interview, open the link yourself ~2 minutes earlier so the free instance is awake.

## Notes
* The "Guardrails OFF" attack lab only touches a throw-away copy of the database, so visitors cannot damage anything.
* Request limits (`DAILY_LIMIT`, `SESSION_LIMIT`) protect your Groq key; lower them if you share the link widely, and rotate the key if you see abuse.
* Groq's free tier has strict per-minute limits; the app retries automatically on "429" but may be slow under load.

## Troubleshooting
| Symptom | Fix |
|---|---|
| Build fails: Python version | set `PYTHON_VERSION` to a full version Render supports (e.g. `3.11.9`, `3.12.x`) |
| `ModuleNotFoundError: src` or `No such file streamlit_app.py` | files are in a sub-folder -> set **Root Directory**, or re-upload the contents |
| App says "No documents available" / slow first start | `index/chunks.json` and `data/papers.db` were not uploaded - do Part A + C |
| "No API key configured" | add `OPENAI_API_KEY` etc. under **Environment**, then redeploy |
| "... 401" | wrong or revoked key |
| "... 404 / model not found" | Groq renamed the model - check console.groq.com/docs/models and update `LLM_MODEL` |
| Out of memory (service restarts) | make sure the index is uploaded (no PDF parsing at start) and `EMBEDDER=tfidf` |
| Page stuck on "Please wait..." | try adding `--server.enableCORS false --server.enableXsrfProtection false` to the Start Command |
| Very slow / 429 | free-tier rate limits - wait a minute |
