# routewise-demo

A standalone RAG Q&A assistant that consumes the deployed routewise router purely
over HTTP -- the same way any third-party app would. It does **not** import or
depend on the routewise source repo.

## What it proves

- **Difficulty-based tier routing**: every question is scored and routed to the
  cheapest tier that can handle it (cheap / mid / frontier).
- **Semantic cache**: repeat/near-duplicate questions return free, instantly.
- **Web-search branch**: time-sensitive questions skip the LLM entirely.
- **Forced-tier comparison**: the same question answered on cheap/mid/frontier
  shows what the router saved you.

## Setup

```powershell
python -m venv venv
venv\Scripts\python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set your routewise API key:

```
ROUTEWISE_API_KEY=<your key>
ROUTEWISE_BASE_URL=https://llm-router-d2b2.onrender.com
```

## Run

```powershell
venv\Scripts\python ingest.py          # downloads Wikipedia corpus, builds index (first run pulls MiniLM)
venv\Scripts\python run_demo.py        # retrieve -> route -> tables
```

`ingest.py --topics "Some Article" "Other Article"` overrides the default topic list.

## Layout

| file | job |
|---|---|
| `ingest.py` | fetch Wikipedia plaintext -> chunk -> MiniLM embeddings -> `data/index.npz` |
| `answer.py` | load index, top-k retrieval, build context prompt, call `/route` |
| `run_demo.py` | end-to-end session + comparison tables |
| `config.py` | API key / base URL from env or `.env` |

The bare question is sent as `query` so the difficulty classifier scores the
question itself; the retrieval context travels in the `messages` field so the
model still sees it when answering. `/route` caps `query` at 1000 chars, so the
context prompt is trimmed to ~950 chars.
