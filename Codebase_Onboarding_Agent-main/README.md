# Codebase Onboarding Agent

A production-oriented agent that helps developers understand an unfamiliar repository by combining
**codebase RAG**, **live GitHub data**, and **persistent user memory** — orchestrated with LangGraph
and grounded with verifiable citations.

```text
User Query
   ↓
Query/Intent Router          (heuristic + LLM refinement)
   ↓
┌──────────────┬──────────────┬──────────────┐
│ Codebase RAG │ GitHub Tools │ User Memory  │
│ FAISS+BM25   │ PRs/commits/ │ SQLite/      │
│ +RRF+rerank  │ issues/CI    │ Postgres     │
└──────────────┴──────────────┴──────────────┘
              ↓
       Context Fusion (agent tool loop)
              ↓
       Groq LLM (OpenAI-compatible API) / extractive fallback
              ↓
      Answer + Citations  →  citation verification
              ↓
        Update Memory (concepts, familiarity)
```

## Features

- **Codebase intelligence** — Tree-sitter AST parsing (regex fallback), semantic symbol-level
  chunks with `file:line` metadata, hybrid retrieval (FAISS/NumPy vectors + BM25) fused with
  **Reciprocal Rank Fusion** and an optional LLM reranker, a dependency/call graph
  (imports + name-resolved calls), and **incremental indexing** driven by content hashes and git.
- **GitHub live context** — commits, PRs, issues, branches, CI check-runs, file history, search.
  Answers distinguish *"what the code contains"* from *"what is happening in development"*.
- **Persistent memory** — every Q&A is stored (SQLAlchemy; SQLite or Postgres) with extracted
  concepts and a per-user familiarity score, so the agent skips re-explaining known concepts and
  adapts depth over time.
- **LangGraph agent** — `route → gather → synthesize → memorize`. The router picks which tool
  groups the agent sees (`search_code`, `get_file`, `get_symbol`, `trace_dependencies`,
  `search_github`, `get_pr`, `get_commit`, `get_ci_status`, `search_memory`, …) so not every
  source is hit on every query.
- **Grounded answers** — inline citations like `[auth/service.py:42-91]`, `[PR #421]`,
  `[commit abc1234]`, verified against the evidence the tools actually returned. If evidence is
  insufficient the agent says so instead of hallucinating.
- **"Onboard Me" mode** — project overview, architecture, entry points, major components,
  dependencies, auth/data flow, testing, deployment, and current development activity.
- **Evaluation pipeline** — repo-specific benchmark measuring Recall@K, MRR, answer correctness,
  citation accuracy, hallucination rate, and latency across
  `vector | bm25 | hybrid | hybrid+RRF | hybrid+RRF+rerank`, plus a memory A/B check.
  **CI fails when quality drops below thresholds.**
- **Groq-powered synthesis** — Groq's OpenAI-compatible chat API (`GROQ_API_KEY`, default
  model `openai/gpt-oss-120b`, any tool-calling Groq model via `GROQ_MODEL`), with rate-limit
  and malformed-tool-call retries tuned for the free tier.
- **Offline-first** — without an LLM API key the agent runs in extractive mode
  (deterministic tool calls + cited evidence snippets), so dev, tests, and CI need no secrets.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                              # core (pure-python, runs everywhere)
pip install -e ".[treesitter,faiss,embeddings,ui]"   # optional: native accuracy/speed extras

cp .env.example .env                                 # add GROQ_API_KEY / GITHUB_TOKEN

# 1. Index a repository (local path or GitHub URL)
onboarding-agent index path/to/repo
onboarding-agent index https://github.com/pallets/flask

# 2. Ask questions
onboarding-agent ask "How does request routing work?"
onboarding-agent ask "Which open PRs touch the auth module?"

# 3. Generate the onboarding guide
onboarding-agent onboard

# 4. Incremental re-index after changes (hash + git diff based)
onboarding-agent refresh

# 5. API + UI
onboarding-agent serve                               # FastAPI on :8000 (/docs)
streamlit run src/onboarding_agent/ui/streamlit_app.py
```

### Docker

```bash
docker compose up --build     # FastAPI on :8000 + Postgres-backed memory
# then: curl -X POST localhost:8000/index -d '{"repo": "/repo"}' -H 'content-type: application/json'
```

## API

| Endpoint         | Method | Description                                   |
|------------------|--------|-----------------------------------------------|
| `/health`        | GET    | Index/LLM/GitHub status                       |
| `/index`         | POST   | `{"repo": "<path or URL>"}` — full index      |
| `/index/refresh` | POST   | Incremental re-index (changed files only)     |
| `/query`         | POST   | `{"question", "user_id"}` → answer + citations + verification |
| `/onboard`       | POST   | Structured onboarding guide                   |

## Evaluation

```bash
python -m evals.run_eval            # writes eval_results/summary.json
python -m evals.run_eval --check    # exit 1 below thresholds (used by CI)
```

The benchmark (`evals/benchmark.jsonl`) is repo-specific: questions about *this* codebase with
expected files and answer keywords. Metrics per retrieval config, e.g.:

- `recall@10` / `mrr` for `vector`, `bm25`, `hybrid`, `hybrid_rrf`, `hybrid_rrf_rerank`
- `answer_correctness` (keyword coverage or expected-file citation)
- `citation_accuracy` (share of citations supported by gathered evidence)
- `hallucination_rate` (answers containing unsupported citations)
- `memory_improves_context` (prior interaction visible to a follow-up conversation)

Thresholds live in `evals/run_eval.py::DEFAULT_THRESHOLDS` and gate CI (`eval-gate` job).

## Configuration

All via environment / `.env` (see `.env.example`): `GROQ_API_KEY`, `GROQ_MODEL`
(default `openai/gpt-oss-120b`), `LLM_MAX_TOKENS`, `GITHUB_TOKEN`, `GITHUB_REPO`,
`DATABASE_URL`, `INDEX_DIR`, `EMBEDDING_PROVIDER` (`hash` | `sentence-transformers`),
`RETRIEVAL_TOP_K`, `RRF_K`, `RERANKER` (`none` | `llm`).

## Architecture

```text
src/onboarding_agent/
├── config.py / logging.py / models.py / service.py   # settings, structlog, domain, wiring
├── ingestion/repo.py        # clone/walk/hash repository files
├── parsing/                 # Tree-sitter AST parser (+regex fallback), semantic chunker
├── retrieval/               # hash/ST embeddings, FAISS/NumPy vectors, BM25, RRF, rerankers
├── graphs/code_graph.py     # imports/calls/defines graph (networkx), trace_dependencies
├── indexing/                # full build + persistence, incremental (hash/git-diff) updates
├── github/client.py         # commits, PRs, issues, branches, CI, history, search
├── memory/                  # SQLAlchemy models + familiarity-aware memory store
├── agent/                   # LangGraph graph, router, tools, Groq LLM, citations
├── onboarding/generator.py  # "Onboard Me" guide
├── api/                     # FastAPI app + schemas
└── ui/streamlit_app.py      # chat + onboarding UI
```

## Development

```bash
pytest                       # unit + integration (offline, no secrets needed)
ruff check src tests evals
python -m evals.run_eval --check
```
