# NaijaSense AI

Behavioral intelligence for Nigerian lifestyle contexts: **Task A** (persona-conditioned reviews) and **Task B** (persona-only recommendations), plus an optional unified demo hub.

**Team:** TAOTECH SOLUTIONS · **Paper:** [`docs/SOLUTION_PAPER.md`](docs/SOLUTION_PAPER.md)

---

## Live endpoints

| | URL |
|--|-----|
| App | [naija-sense-ai.vercel.app](https://naija-sense-ai.vercel.app/) |
| Task A API | `POST` [/task-a/user-modeling](https://naija-sense-ai.vercel.app/task-a/user-modeling) |
| Task B API | `POST` [/task-b/recommendation](https://naija-sense-ai.vercel.app/task-b/recommendation) |
| Task A / B UI | [/task-a](https://naija-sense-ai.vercel.app/task-a) · [/task-b](https://naija-sense-ai.vercel.app/task-b) |
| Unified demo | [/unified](https://naija-sense-ai.vercel.app/unified) |

| Task | Input | Output |
|------|--------|--------|
| **A (User modeling)** | `user_persona`, `product_details` (strings) | `rating`, `review_reasoning`, `review_text` |
| **B (Recommendation)** | `user_persona`: `{ user_id, persona }` | `recommendations` (paragraph), `agent_reasoning` |

Stack: Groq **Llama 3.1 8B** (router) + **Llama 3.3 70B** (generator). Set `GROQ_API_KEY` and `ORCHESTRATOR_PROVIDER=groq` in production.

---

## Architecture

### Submission endpoints (Task A & Task B)

```mermaid
flowchart TB
    C[Client] --> TA[POST /task-a/user-modeling]
    C --> TB[POST /task-b/recommendation]
    TA --> P1[parse_task_a_inputs<br/>domain from product text]
    P1 --> A1[TaskATwoPassAgent]
    A1 --> R1[rating + review_reasoning + review_text]
    TB --> P2[parse_task_b_persona<br/>intent: lifestyle / advisory / team]
    P2 --> B1[TaskBPipelineAgent]
    B1 --> S1[Stage 1: top-30 from 5k corpus index]
    S1 --> D1[diversify + dedupe variants]
    D1 --> S2[Stage 2: Groq router rank]
    S2 --> S3[Groq generator paragraph]
    S3 --> R2[recommendations + agent_reasoning]
```

**Task A flow (`TaskATwoPassAgent`):**

1. Parse `user_persona` + `product_details` strings; infer **product domain** from product text (`core/task_a_inputs.py`).
2. **Pass 1 (router):** JSON `{ rating, review_reasoning }` with domain few-shots + optional corpus search (top 3).
3. **Pass 2 (generator):** first-person `review_text` with **rating locked** from Pass 1.

**Task B flow (`TaskBPipelineAgent`):**

1. Parse `user_persona.persona` only (no separate query/context field); detect advisory-only or team-culture intent when relevant.
2. **Stage 1:** `retrieve_top_k` on the shared **5,011-row** corpus, diversify by domain, dedupe variant titles (`canonical_item_title`).
3. **Stage 2:** Groq router ranks `item_id`s; Groq generator writes one grounded paragraph; stage-1 template fallback if Groq fails.

### Unified hub (demo + benchmarks)

```mermaid
flowchart LR
    U[User] --> FE[Behavioral Intelligence Hub<br/>Next.js /unified]
    U --> SW[Swagger /docs]
    FE -->|POST| AGW[Agent Gateway v1<br/>multi-turn buffer]
    FE -->|POST NDJSON| STR[Streaming Gateway]
    FE -->|POST thumbs| FB[Feedback JSONL]
    SW --> AGW
    AGW --> SAFE[Safety Layer]
    STR --> SAFE
    SAFE --> IR[Intent Router]
    IR --> O[Orchestrator]
    O -.optional.-> SCR[Silent History by user_id]
    SCR --> HUS[(Historical User Store)]
    O --> UMA[User Modeling Agent]
    O --> RGA[Review Generation + Critic]
    O --> RA[Recommendation Agent<br/>deterministic hybrid scorer]
    RGA --> RAG[(Corpus / RAG)]
```

**Unified hub only** (not used on `/task-a` or `/task-b`):

0. **Silent context retrieval** by `user_id` before routing (`HistoricalUserStore` → `UserMemory` + `HistoricalPersona`).
1. **Task A (orchestrator):** persona merge → RAG → generate → optional critique→regenerate → persist to memory.
2. **Task B (orchestrator):** multi-turn buffer → deterministic hybrid scorer (`0.5×interest + 0.25×memory + …`) → `chain_of_thought` in `explainability` (legacy `/api/v1/recommend` and ablation harness).

See [`docs/SOLUTION_PAPER.md`](docs/SOLUTION_PAPER.md) for the full submission vs demo split.

---

## Quick start

**Prerequisites:** Python 3.11+, Node 20+ (or Docker 24+).

```bash
git clone https://github.com/taotechs/NaijaSense-AI.git
cd NaijaSense-AI
cp .env.example .env   # add GROQ_API_KEY
docker compose up --build
```

| Service | URL |
|---------|-----|
| UI | http://localhost:3000 |
| API docs | http://localhost:8000/docs |
| Health | http://localhost:8000/api/v1/health |

**Local dev:** `pip install -r requirements.txt`, `PYTHONPATH=.` + `uvicorn main:app --reload`; in `frontend/`, `npm install && npm run dev`.

**Smoke test:** `python scripts/smoke_api.py http://127.0.0.1:8000`

---

## API examples

**Task A**

```bash
curl -X POST "http://localhost:8000/task-a/user-modeling" \
  -H "Content-Type: application/json" \
  -d '{"user_persona":"Lagos foodie in Yaba, balanced tone.","product_details":"Iya Eba Amala Spot - lunch for two, amala soft, egusi rich, about 2k each, 20 min wait."}'
```

**Task B**

```bash
curl -X POST "http://localhost:8000/task-b/recommendation" \
  -H "Content-Type: application/json" \
  -d '{"user_persona":{"user_id":"demo_user","persona":"UNILAG student in Yaba on a 10k weekly budget. Loves jollof, street food, and weekend Nollywood."}}'
```

Other routes (`/api/v1/*`, `/api/agent/v1`, streaming) are documented in Swagger and [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

---

## Data & corpus

**One corpus for Task A and Task B:** `data/processed/review_corpus.jsonl` (**5,011** rows: Yelp 2,498, Amazon 2,482, Goodreads 31).

Task A uses it for few-shot retrieval; Task B uses the same file via `data/processed/corpus_index.json` (built at deploy / locally).

**Build or refresh the index:**

```bash
python scripts/build_corpus_index.py
```

**Rebuild review_corpus.jsonl** (optional, Hugging Face or offline):

```bash
python scripts/build_review_corpus.py --use_hf --hf_sources yelp,amazon \
  --limit 2500 --extra_jsonl data/offline_review_samples.jsonl
python scripts/build_corpus_index.py --force
```

---

## Evaluation

```bash
pytest -q
python scripts/run_real_benchmark.py --sample_size 20 --all_variants
python scripts/eval_fidelity.py --base-url http://127.0.0.1:8000 --limit 20
```

Metrics and methodology: [`docs/EVAL.md`](docs/EVAL.md) · Results: [`data/benchmark_results.json`](data/benchmark_results.json)

---

## Configuration

| Variable | Purpose |
|----------|---------|
| `ORCHESTRATOR_PROVIDER` | `groq`, `openai`, or `none` |
| `GROQ_API_KEY` | Groq API key (Task A + Task B stage 2) |
| `TASK_B_TOP_K` | Recommendations ranked (default `6`) |

See `.env.example` for the full list.

---

## Project layout

```text
agents/          Task A & B pipelines
api/             FastAPI routes
core/            Orchestrator, corpus index, persona parsing
frontend/        Next.js UI
scripts/         Corpus generation, benchmarks, smoke tests
docs/            Solution paper, deployment, evaluation guides
evals.py         Hackathon KPI helpers
```

---

## Docs

- [Solution paper](docs/SOLUTION_PAPER.md)
- [Deployment](docs/DEPLOYMENT.md)
- [Evaluation](docs/EVAL.md)
