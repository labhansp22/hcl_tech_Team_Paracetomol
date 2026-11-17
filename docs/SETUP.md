# Pre-Hackathon Setup (Windows)

Do all of this **before** the hackathon, on **every teammate's laptop**. Downloading on the day wastes time and venue Wi-Fi is often slow.
Total download is about **2 GB**. Allow about 30 minutes. The LLM is used **through API keys only** (Groq, then Gemini as backup); nothing runs locally.

## 1. Software to install

| # | Tool | Why | Install | Status on this PC |
|---|---|---|---|---|
| 1 | **Python 3.10 or 3.11 (64-bit)** | Everything runs on it | python.org/downloads, ticking "Add to PATH" | ✅ 3.10.8 |
| 2 | **Git** | Version control, branches per teammate | git-scm.com | ✅ 2.56 |
| 3 | **VS Code** + extensions: *Python*, *Pylance*, *Jupyter* | Editor; Jupyter for the Data Science track | code.visualstudio.com | ✅ (check extensions) |
| 4 | **GitHub CLI** (`gh`) | Create the repo, open PRs and log in from the terminal | `winget install GitHub.cli` | ❌ install (optional but handy) |
| 5 | **DB Browser for SQLite** | Inspect `mock.db` and the audit tables | sqlitebrowser.org | optional |
| 6 | **Microsoft Visual C++ Redistributable (x64)** | Needed by PyTorch on Windows; install only if `import torch` fails | learn.microsoft.com → "Latest supported VC++ downloads" | check |
| — | Node.js | Not needed (we use Streamlit, not React) | — | ✅ already installed |

## 2. Accounts and free API keys

| Account | Get it at | Used for |
|---|---|---|
| **GitHub** (every teammate) | github.com | Commits credited to each person |
| **Groq** API key | console.groq.com/keys | Primary LLM (free, fast) |
| **Google AI Studio** API key | aistudio.google.com/apikey | Backup LLM (free) |

Put keys only in `.env` (copied from `.env.example`). Never commit them.

**Both keys are needed.** Free tiers have rate limits, so when Groq returns a rate-limit error the app switches to Gemini automatically.

**Each teammate should create their own keys.** Sharing one key means sharing one rate limit.

**If both APIs fail** (for example, the venue network is down), the app still returns search results and tool output without the LLM-written explanation. Keep a phone hotspot ready.

## 3. Python environment (in the project folder)

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
python -m pip install --upgrade pip
```

```bash
pip install -r requirements.txt
```

This downloads about 1–1.5 GB (PyTorch CPU, ChromaDB, LightGBM…). If `chromadb` fails to build, upgrade pip and retry; it ships prebuilt wheels for Python 3.10 and 3.11.

## 4. Pre-download the search models (one-time download, then cached)

| Model | Size | Used for |
|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` | ~470 MB | Embeddings (multilingual) |
| `all-MiniLM-L6-v2` | ~90 MB | Embeddings (English, faster) |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~90 MB | Optional reranker |

These small search models run on the laptop (they are not LLMs). The first run of the check script (Step 5) downloads them into the cache.

**Disk:** C: has about 25 GB free, which is enough. To keep the model cache on D: anyway, run this before the first download, then open a new terminal:

```bash
setx HF_HOME D:\ai-cache\huggingface
```

## 5. Verify everything

```bash
python scripts/check_env.py
```

Every line should say **PASS**. "LLM groq" and "LLM gemini" fail until you add the keys to `.env`.

## 6. Day-of checklist

- [ ] `check_env.py` passes on all three laptops
- [ ] Each teammate's `git config user.name` / `user.email` matches their GitHub account
- [ ] `gh auth login` done, or a GitHub personal access token ready
- [ ] Phone hotspot available as a backup for venue Wi-Fi (the LLM needs internet)
- [ ] `docs/PLAYBOOK.md` open; the brief card is ready to fill in
