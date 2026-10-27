# Deploy: GitHub + live link (Streamlit Community Cloud)

GitHub stores your **code**; it cannot run a Python app. The free host **Streamlit Community Cloud** runs the app
straight from your GitHub repo and gives you a public link like `https://your-name.streamlit.app`.
The language model is a free **Groq** API key (Ollama can't run on free hosting).

You need: a GitHub account, Git (or GitHub Desktop), a Groq account.

---
## Step 0 - Test locally first (10 min)
```bash
python -m venv .venv && .venv\Scripts\activate            # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
mkdir .streamlit && copy .streamlit\secrets.toml.example .streamlit\secrets.toml   # Mac/Linux: cp
# edit .streamlit/secrets.toml -> paste your Groq key (Step 1)
streamlit run streamlit_app.py
```
The first start downloads the 10 papers and builds the index (1-2 min). If this works locally, deployment will work.

## Step 1 - Get a free Groq API key
1. https://console.groq.com -> sign up -> **API Keys** -> **Create API Key** -> copy it (starts with `gsk_`). It is shown once.
2. Open https://console.groq.com/docs/models and pick a current chat model (e.g. `llama-3.3-70b-versatile`). Model names change - if the one in `secrets.toml.example` is gone, use a current one.
3. Free tiers have strict per-minute limits; the app retries automatically on "429" errors but may feel slow if many people use it at once.

## Step 2 - Put the project on GitHub
1. Create an empty **public** repo on github.com -> New repository -> name `agentic-rag-guardrails` -> do NOT add a README/licence (you already have them).
2. In the project folder:
```bash
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"

git init
git add .
git status          # CHECK: .env and .streamlit/secrets.toml must NOT appear in the list
git commit -m "Agentic RAG with safety guardrails"
git branch -M main
git remote add origin https://github.com/<your-username>/agentic-rag-guardrails.git
git push -u origin main
```
(Sign in via the browser window / token when asked. Prefer clicking? Use **GitHub Desktop**: File -> Add local repository -> Publish repository.)

**If you ever commit an API key by mistake: revoke it in the Groq console immediately** - deleting the file later is not enough, the key stays in git history.

## Step 3 - Deploy on Streamlit Community Cloud
1. Go to https://share.streamlit.io -> **Continue with GitHub** -> authorise.
2. Click **Create app** -> *Deploy a public app from GitHub*.
3. Fill in: **Repository** `<your-username>/agentic-rag-guardrails`, **Branch** `main`, **Main file path** `streamlit_app.py`.
4. Choose your own **App URL** (e.g. `agentic-rag-guardrails`).
5. Click **Advanced settings**:
   * **Python version:** 3.11 or 3.12 (Community Cloud does *not* read `runtime.txt`; the version is set here or later under app Settings).
   * **Secrets:** paste the contents of your `secrets.toml` (the same keys as `.streamlit/secrets.toml.example`, with your real key and your real GitHub URL for `REPO_URL`).
6. Click **Deploy**. It installs `requirements.txt` (a few minutes), then the app downloads the papers and builds the index on first start (1-2 min more). Watch progress under **Manage app -> logs**.
7. Your live link appears in the browser bar, e.g. `https://agentic-rag-guardrails.streamlit.app`.

## Step 4 - Finish the portfolio
* Edit `README.md`: replace `YOUR-APP-NAME` with your real link and `<your-username>` everywhere; add a screenshot (`docs/demo.png`).
* On the GitHub repo page: **About** (gear icon) -> add the live link as *Website* and topics like `rag`, `llm-agents`, `ai-safety`, `prompt-injection`.
* Resume / LinkedIn: add `Live demo: <link>` and `GitHub: <repo link>` to the project.
* Every `git push` to `main` redeploys the app automatically.

## Notes
* **Sleeping apps:** free apps go to sleep when nobody visits for a while; the next visit wakes them (about a minute). Open the link yourself before an interview.
* **Cost protection:** the app caps requests (`DAILY_LIMIT`, `SESSION_LIMIT` secrets). Lower them if you share the link widely. Rotate the key if you see abuse.
* **OFF mode is safe:** the "Guardrails OFF" attack lab only ever touches a throw-away copy of the database.
* **Faster cold starts (optional):** build the index locally (`python -m scripts.download_data` then `python -m src.ingest --no-dense`) and commit `index/chunks.json`, `index/meta.json` and `data/papers.db`. The app then skips the download.

## Troubleshooting
| Symptom | Fix |
|---|---|
| "No API key configured" | add `OPENAI_API_KEY`, `LLM_MODEL`, `OPENAI_BASE_URL` in app **Settings -> Secrets** and reboot |
| `ModuleNotFoundError` in logs | the package is missing from `requirements.txt` - add it, push |
| Build fails on `pymupdf` | set Python 3.11 or 3.12 in app Settings |
| "download failed" / "No documents available" | arXiv unreachable: run locally, commit the pre-built index (see Notes), push |
| Answers say "LLM API request failed ... 401" | wrong/expired key |
| "... 404 / model not found" | the model name changed - check console.groq.com/docs/models |
| Slow answers / "429" | free-tier rate limit - wait a minute, or lower traffic |
| Weird/invalid-JSON behaviour | use a bigger model (70B class); tiny models follow the JSON action format badly |

## Alternative host
**Hugging Face Spaces** also gives a free public link (create a *Streamlit* Space, push the same files, add the secrets in Space settings).
