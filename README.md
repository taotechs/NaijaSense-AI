# NaijaSense AI · Behavioral Intelligence Hub

NaijaSense AI is a context-aware, multi-agent system for the **DSN × Bluechip Tech LLM Agent Challenge (DSAS 2026)**. It tackles both competition tasks behind one unified API and a single chat UI branded as the **Behavioral Intelligence Hub**:

- **Task A — User Modeling:** simulate a star rating and a written review for an unseen item, conditioned on a user persona inferred from minimal signals.
- **Task B — Recommendation:** rank items for an individual user, handling cold-start, cross-domain, and multi-turn conversational queries with explicit reasoning traces.

The system intentionally **separates a small fast router model from a strong generator model**, grounds review writing in retrieved corpus examples (RAG), and runs an optional **critique → regenerate** loop to catch generic outputs before they reach the user. Every request also runs a **silent context-retrieval step** that pulls the user's historical ratings/reviews from the normalized corpus by `user_id` *before* any LLM call, so the persona used for generation and ranking reflects real past behaviour rather than a static UI profile.

![Behavioral Intelligence Hub — landing screen with backend status pill, agentic-workflow indicator, and Nigerian quick-start prompts](docs/homescreen.png)

> **Live demo:** <https://naija-sense-ai.vercel.app/unified> · **API:** <https://youthful-wynn-taotechs-6715c87e.koyeb.app/api/v1/health>

---

## Why this submission

- **Honest, reproducible benchmarks.** Ablation numbers in [`data/benchmark_results.json`](data/benchmark_results.json) measured on a held-out slice of Yelp + Goodreads + Amazon reviews, plus a behavioural-fidelity A/B harness ([`scripts/eval_fidelity.py`](scripts/eval_fidelity.py), [`docs/EVAL.md`](docs/EVAL.md)) that quantifies how much the silent history step moves the needle.
- **Streaming agentic UX.** The unified gateway has a non-blocking NDJSON streaming sibling (`POST /api/agent/v1/stream`) that emits each reasoning step as it fires; the UI renders them as an animated timeline so the agent's thinking is visible in real time.
- **Transparent reasoning.** Every response carries `reasoning_steps`, `safety_flags`, `timing_ms`, the routing path, and the critique-pass verdict — all surfaced in the UI as pills, badges, and an expandable trace.
- **Multi-language output.** A `language` field on the persona supports `english`, `pidgin`, and `yoruba_mix`, threaded into the generator prompt as a hard rule that overrides the persona-style preset.
- **Two-model split.** Cheap router model (Groq Llama-3.1-8B) for classification + persona inference + safety critique; strong generator (Llama-3.3-70B) for writing. Cuts cost while keeping output quality high.
- **Advisory safety layer.** Input is scanned for prompt-injection + PII shapes; output is checked for PII leakage and ungrounded numeric specifics. Findings surface as a non-blocking `safety_flags` array, never as a hard block.
- **Feedback loop, built-in.** Thumbs-up / thumbs-down on every result writes to a JSONL log (`POST /api/agent/feedback`) so the team can audit outputs and feed the signal back as few-shot examples.
- **Nigerian contextualisation.** Persona styles support formal global English and `nigerian_twitter` (light pidgin colouring) with hard rules in the prompt to prevent forced slang.
- **Containerised** end-to-end (FastAPI + Next.js + Chroma) with one `docker compose up`, plus a public production deploy on **Koyeb (backend) + Vercel (frontend)**.

---

## Architecture

```mermaid
flowchart LR
    U[User] --> FE[Behavioral Intelligence Hub<br/>Next.js /unified]
    U --> SW[Swagger /docs]
    FE -->|POST| AGW[/api/agent/v1<br/>+ multi-turn buffer]
    FE -->|POST NDJSON| STR[/api/agent/v1/stream<br/>live reasoning steps]
    FE -->|POST thumbs| FB[/api/agent/feedback<br/>JSONL log]
    FE -->|GET| HC[/api/v1/health<br/>status pill + pre-warm]
    SW --> AGW
    AGW --> SAFE[Safety Layer<br/>prompt-injection / PII / ungrounded specifics]
    STR --> SAFE
    SAFE --> IR[Intent Router LLM small/heuristic]
    IR -->|task=review| O[Orchestrator]
    IR -->|task=recommend| O
    O -.optional skip.-> SCR[Silent Context Retrieval<br/>by user_id]
    SCR --> HUS[(Historical User Store<br/>corpus indexed by user_id)]
    SCR --> O
    O --> UMA[User Modeling Agent — router LLM<br/>history baseline + UI override]
    O --> RGA[Review Generation Agent — generator LLM<br/>english / pidgin / yoruba_mix]
    O --> RA[Recommendation Agent — deterministic hybrid scorer<br/>+ chain-of-thought trace]
    RGA --> RAG[(Review Corpus Store / RAG)]
    RGA --> CRI[Critic LLM — router model]
    O --> MEM[(User Memory / Vector Store<br/>warmed from history)]
    O --> TRC[Reasoning Trace + safety_flags + timing_ms]
```

### Stateful agentic workflow (silent context retrieval)

Both task flows start with the same shared step:

0. **Silent context retrieval.** `HistoricalUserStore` (built at startup from `data/processed/review_corpus.jsonl`) is queried by `user_id`. Up to 5 past entries are pulled, normalised to short snippets, and pushed into `UserMemory` so downstream vector retrieval also sees historical behaviour. A `HistoricalPersona` summary (`avg_rating`, `rating_tendency`, `sentiment_bias`, `tone_signal`, `top_domains`, `inferred_interests`) is derived from the same rows. This whole step runs *before any LLM call* and never blocks the request — unknown `user_id`s fall through with an empty persona.

The `UserModelingAgent` then merges this baseline with the UI-supplied persona under explicit **default-vs-override semantics**: history wins by default, UI fields override only when the user actively set them (non-empty for free-text, non-default for enums). The merge result is logged in `merge_meta.overridden_fields` and `merge_meta.source_per_field` and surfaced in the reasoning trace and on the response. Multi-turn for Task B is implemented as a per-`user_id` rolling 6-turn buffer in `api/deps.py`; the most recent prior turns are threaded into `RecommendationRequest.conversation_history` automatically.

### Task A — Review Simulation flow

1. **Silent context retrieval** for `user_id`.
2. `UserModelingAgent` merges historical baseline + UI overrides.
3. `ReviewCorpusStore.search()` retrieves top-3 similar items for style grounding.
4. `ReviewGenerationAgent` builds a structured *fact* prompt (no draft text), adds a few-shot block, a variation token, and a per-call random seed; calls the strong generator with explicit `top_p`, `presence_penalty`, `frequency_penalty`, `seed`.
5. If `REVIEW_CRITIQUE_ENABLED=true`, the router model scores the review on specificity (1–5). Below threshold → one rewrite with the critic's issues injected as rules.
6. Generated review is stored back into `UserMemory` for downstream tasks. Response includes `persona_breakdown.historical_signal`, `persona_breakdown.history_used`, and `persona_breakdown.merge_meta`.

### Task B — Recommendation flow

1. **Silent context retrieval** for `user_id`.
2. Gateway threads the **rolling multi-turn buffer** (prior turns from this same user_id) into `conversation_history`.
3. Chain-of-thought reasoning step: the orchestrator logs *why* this retrieval strategy was chosen (history present?, multi-turn turns count, free-text context).
4. `UserMemory` (now warmed with both in-session and silent-historical snippets) returns relevant prior interactions.
5. `RecommendationAgent` scores each candidate with a hybrid signal:
   - `0.5 × interest_overlap + 0.25 × memory_overlap + 0.2 × context_overlap + 0.2 × domain_alignment + base + bias`
   - Conditional boosts: spicy / budget / relax / cold-start / cross-domain.
   - Template-y items ("starter pack", "bundle") penalised when there's no contextual hook.
6. The agent emits an explicit `chain_of_thought` array in `explainability` naming the path taken (warm vs cold start, cross-domain flag, multi-turn-aware flag, active intent boosts, top pick rationale).
7. Returns ranked recommendations + conversational summary + explainability dict with `historical_signal` + `history_turns_used`.

---

## How to run

> Prerequisites: **Python 3.11+**, **Node.js 20+**, and (for the container path) **Docker 24+ with Docker Compose v2**. A Groq API key (free tier works) or OpenAI key is recommended — without one the app falls back to deterministic heuristics.

> Going live? See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for a step-by-step Vercel (frontend) + Koyeb (backend) deploy, plus drop-in alternatives for Render, Fly.io, Hugging Face Spaces, and Railway. The repo ships with `koyeb.yaml`, `render.yaml`, and a `$PORT`-aware `Dockerfile` so the live deploy is essentially clone → connect → set env vars.

### Option A · Docker Compose (recommended for judges)

Spins up the FastAPI backend, the Next.js Behavioral Intelligence Hub UI, and a Chroma vector store with one command:

```bash
git clone https://github.com/<your-org>/NaijaSense-AI.git
cd NaijaSense-AI
cp .env.example .env            # then edit and add GROQ_API_KEY (or OPENAI_API_KEY)
docker compose up --build
```

When the containers are healthy:

| Service | URL |
|---|---|
| **Behavioral Intelligence Hub (UI)** | <http://localhost:3000> |
| Swagger / OpenAPI docs | <http://localhost:8000/docs> |
| Health probe | <http://localhost:8000/api/v1/health> |
| Chroma vector store | <http://localhost:18000> |

To stop: `docker compose down` (add `-v` to also drop the Chroma volume).

### Option B · Local dev (hot-reload)

Run the backend and frontend in two terminals. From the project root:

```bash
# Terminal 1 — FastAPI
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# macOS / Linux:       source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env             # add GROQ_API_KEY (or OPENAI_API_KEY)

# Windows PowerShell:
$env:PYTHONPATH = (Get-Location).Path
# macOS / Linux:
export PYTHONPATH=$(pwd)

python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

```bash
# Terminal 2 — Next.js UI
cd frontend
npm install                       # first time only
npm run dev                       # serves http://localhost:3000
```

Open <http://localhost:3000> — the home route redirects to `/unified`, which is the Behavioral Intelligence Hub. Swagger lives at <http://localhost:8000/docs>.

### Using the UI

The Behavioral Intelligence Hub gives the judge everything they need from one screen.

![Filling out a query and the behavioral profile in the unified hub](docs/input.png)

1. **Single input field.** Placeholder: *"Simulate a review for a Nigerian spot or ask for personalized recommendations…"* — the LLM intent router decides between Task A and Task B automatically.
2. **Quick-start chips** with Nigerian context (Ikeja suya, late-night Yaba akara/noodles, Iya Eba jollof, Abuja-on-10k). One click fills the textarea.
3. **Output language selector.** Three options — *English*, *Nigerian Pidgin*, *English + Yoruba mix* — threaded into the generator as a hard prompt rule that overrides the persona-style preset.
4. **Behavioral profile (Task A user modeling).** A clearly labelled collapsible section with a **Quick preset** dropdown (Lagos foodie · VI lifestyle critic · Abuja professional · Campus student) plus manual fields for location, interests, sentiment bias, tone notes and history.
5. **Silent history toggle.** When checked (default), the silent corpus retrieval step runs before routing; uncheck only for experimentation. The hub always shows **one** answer per submission.
6. **Backend status pill.** A small pill in the page header pings `/api/v1/health` on mount (which doubles as a free-tier pre-warm) and polls every 60s. States: *checking → waking up… → ready · NNms → unreachable*.
7. **Live agent trace.** While the request streams, an animated timeline fills in step-by-step (silent retrieval → persona strategy → build persona → generate → critique / persist). Each node has its own icon and pulses while active, so the agent's reasoning is visible rather than implied.

![Result card with safety advisories, timing pill, ★ rating, generated review, thumbs feedback, and expandable reasoning trace](docs/output.png)

8. **Result card** — task pill (`Task A · review` or `Task B · recommend`), routing source (`llm` vs `heuristic`), language tag, `NNms` latency, an amber `Critique applied` chip when the critique→regenerate loop fired, and the orchestrator's rationale.
9. **Safety advisories.** Non-blocking flags from the validation layer surface as small amber chips with hover-tooltips (e.g. `prompt_injection_suspected`, `ungrounded_numeric_specifics`, `pii_phone_in_input`).
10. **Thumbs feedback.** Every result card carries 👍 / 👎 buttons that POST to `/api/agent/feedback` and append to a JSONL log for later audit + few-shot fine-tuning fuel.
11. **Agentic reasoning trace** — the same animated timeline, frozen on the final state, with the full numbered list of every reasoning line emitted by the orchestrator.

### Verify the stack in 30 seconds

```bash
# 1. Health probe (should return {"status": "ok"})
curl http://localhost:8000/api/v1/health

# 2. End-to-end smoke (review + recommend + critique scenarios)
python scripts/smoke_api.py http://127.0.0.1:8000
```

If both pass, the submission is functional end-to-end.

---

## Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `ORCHESTRATOR_PROVIDER` | `none` | `groq`, `openai`, or `none` (heuristic + deterministic fallback). |
| `GROQ_API_KEY` | — | Groq Cloud API key. |
| `GROQ_ROUTER_MODEL` | `llama-3.1-8b-instant` | Small/fast model for intent routing, persona inference, and review critique. |
| `GROQ_GENERATOR_MODEL` | `llama-3.3-70b-versatile` | Strong model for review writing. |
| `OPENAI_API_KEY` | — | Used only when `ORCHESTRATOR_PROVIDER=openai`. |
| `ORCHESTRATOR_MODEL` | `gpt-4o-mini` | OpenAI model name when OpenAI is selected. |
| `GEN_TEMPERATURE` | `0.85` | Generator sampling temperature. |
| `GEN_TOP_P` | `0.9` | Generator nucleus sampling. |
| `GEN_PRESENCE_PENALTY` | `0.6` | Discourages topic repetition. |
| `GEN_FREQUENCY_PENALTY` | `0.5` | Discourages token repetition. |
| `GEN_MAX_TOKENS` | `320` | Max tokens per generated review. |
| `REVIEW_CRITIQUE_ENABLED` | `true` | Toggles the critique → regenerate loop. |
| `REVIEW_CRITIQUE_THRESHOLD` | `4` | Specificity score (1–5) below which the review is rewritten. |
| `CHROMA_HOST` / `CHROMA_PORT` | — / `8000` | Optional Chroma service for persistent vector memory. |

---

## API endpoints

| Method | Path | What it does |
|---|---|---|
| `GET` | `/api/v1/health` | Liveness probe; doubles as a cold-start pre-warm for the frontend. |
| `POST` | `/api/v1/simulate-review` | Task A — explicit endpoint. Body: `user_profile`, `item_data`, `persona_style`. |
| `POST` | `/api/v1/recommend` | Task B — explicit endpoint. Body: `user_profile`, `candidate_items`, `context`, `top_k`. |
| `POST` | `/api/agent/v1` | **Unified gateway.** Body: `user_persona`, `query`, `include_history?`, `compare_with_no_history?`. Routes to Task A or B via the LLM intent router (with heuristic fallback). |
| `POST` | `/api/agent/v1/stream` | **Streaming unified gateway.** Same payload as `/v1`, returns `application/x-ndjson` — one JSON event per line (`start` → `route` → `plan` → `step_start`/`step_end` × N → `final`). Powers the live reasoning timeline. |
| `POST` | `/api/agent/feedback` | Thumbs-up/down feedback. Appends to `data/feedback.jsonl`. |
| `GET` | `/api/agent/feedback/stats` | Aggregate over the feedback log (total / positive / negative / positive_pct). |

### New request fields on `/api/agent/v1` and `/v1/stream`

| Field | Type | Purpose |
|---|---|---|
| `user_persona.language` | `english` \| `pidgin` \| `yoruba_mix` | Hard language rule threaded into the generator prompt. Overrides `persona_style`. |
| `include_history` | `bool` (default `true`) | When `false`, skips the silent historical-context step. Useful for A/B isolation. |
| `compare_with_no_history` | `bool` (default `false`) | **Optional — not used by the public hub.** When `true` (and `include_history=true`), the API runs a second pass with history disabled and attaches it as `response.no_history_variant` (for scripts such as `scripts/eval_fidelity.py`). |

### New response fields

| Field | Type | Purpose |
|---|---|---|
| `safety_flags` | `string[]` | Advisory advisories: `prompt_injection_suspected`, `pii_email_in_input`, `ungrounded_numeric_specifics`, etc. Never blocking. |
| `timing_ms` | `int` | End-to-end server latency, surfaced as a UI pill. |
| `language` | `string` | The language actually used (after server-side normalisation). |
| `no_history_variant` | recursive `AgentGatewayResponse` | Present only when `compare_with_no_history=true` in the API request (eval / research use). The Behavioral Intelligence Hub does not request this field. |

### Example — unified gateway (recommended)

```bash
curl -X POST "http://localhost:8000/api/agent/v1" \
  -H "Content-Type: application/json" \
  -d '{
    "user_persona": {
      "user_id": "judge_demo",
      "location": "Lagos",
      "interests": ["street food", "amala"],
      "sentiment_bias": "balanced",
      "tone_notes": "Use Nigerian twitter tone.",
      "language": "pidgin"
    },
    "query": "Review for Iya Eba Amala Spot. Saturday lunch with a friend; amala was soft, egusi rich, 20 min wait, paid about 2k each.",
    "top_k": 4
  }'
```

The response includes `task`, `routing_source` (`llm` or `heuristic`), `review`/`recommendation`, `safety_flags`, `timing_ms`, `language`, and a `reasoning_steps` array including any critique-pass note.

### Example — streaming gateway (live reasoning)

```bash
curl -N -X POST "http://localhost:8000/api/agent/v1/stream" \
  -H "Content-Type: application/json" \
  -d '{ "user_persona": {"user_id":"judge_demo", "language":"pidgin"},
        "query": "Suggest cheap weekend places to eat in Yaba." }'
```

The response is `application/x-ndjson` — one JSON object per line:

```jsonc
{"type":"start","ts":1747345560123}
{"type":"route","task":"recommend","source":"llm","rationale":"…"}
{"type":"plan","flow":"task_b_memory_recommendation","steps":["silent_context_retrieval", "…"]}
{"type":"step_start","flow":"task_b_memory_recommendation","step":"silent_context_retrieval"}
{"type":"step_end","flow":"task_b_memory_recommendation","step":"silent_context_retrieval"}
// …more step pairs…
{"type":"final","result": { /* full AgentGatewayResponse */ }}
```

The frontend parses these line-by-line and animates the reasoning timeline in real time.

---

## Datasets

The retrieval corpus at `data/processed/review_corpus.jsonl` represents **all three datasets named in the brief** (Yelp, Amazon, Goodreads). Build/rebuild with:

```bash
# HuggingFace path (recommended; needs internet)
python scripts/build_review_corpus.py \
  --output data/processed/review_corpus.jsonl \
  --extra_jsonl data/offline_review_samples.jsonl \
  --use_hf --hf_sources yelp,amazon --limit 250

# Fully offline path (uses curated seed only — useful for reproducibility)
python scripts/build_review_corpus.py \
  --output data/processed/review_corpus.jsonl \
  --extra_jsonl data/offline_review_samples.jsonl
```

A Kaggle path is also supported (`--use_kaggle` with API token, or `--kaggle_*_dir` for a manual unzipped download). See `scripts/build_review_corpus.py --help`.

Schemas are normalised through `data_pipeline/normalize.py`. The corpus we evaluated against in `data/benchmark_results.json` is:

| Source | Rows |
|---|---:|
| Yelp (HF + curated Nigerian restaurant seeds) | 274 |
| Goodreads (curated, including 20+ African-lit titles) | 31 |
| Amazon (curated tech/kitchen + HF when available) | 6 |
| **Total** | **311** |

> The HF `amazon_polarity` parquet was unreachable during our benchmark run; numbers reflect what was available. The pipeline supports the full slice when the endpoint is healthy.

---

## Evaluation

### Real-data benchmark with ablations

```bash
# Single variant
python scripts/run_real_benchmark.py --sample_size 20 --variant full --task both

# All variants (full / no_rag / no_critique / no_llm) — writes JSON report
python scripts/run_real_benchmark.py --sample_size 20 --all_variants --output data/benchmark_results.json
```

The harness:
- Samples N rows stratified by gold rating (positive / critical).
- **Task A:** generates a review from `item_name` only (gold review text is never shown) and scores against the gold via ROUGE-1/2/L, BERTScore (or token-F1 fallback), and RMSE on rating.
- **Task B:** builds a 20-item candidate set (target + 19 same-domain distractors) and scores NDCG@10 + Hit Rate@10.

> **BERTScore fallback note.** `bert-score` requires a torch build; on Python 3.14 wheels are unavailable and source builds time out. The evaluation module falls back to a token-F1 lexical proxy (clearly labelled `bertscore_mode=token-f1-fallback` in outputs). Real BERTScore can be enabled by installing `bert-score` in any Python ≤ 3.12 environment.

### Headline ablation findings

See `data/benchmark_results.json` for the full table. Highlights:

- **LLM is the dominant factor for Task A**: removing it drops ROUGE-1 by ~22% (0.161 → 0.126).
- **RAG slightly hurts ROUGE but helps diversity** (a classic lexical-metric blind spot): retrieved few-shots push the model toward more concrete, item-specific phrasings that don't necessarily share n-grams with gold.
- **Critique loop is metric-neutral on lexical scores** — by design, it targets human quality, not n-gram overlap.
- **Task B is identical across variants** because the ranker is fully deterministic; the LLM contributes only the conversational summary. The current Hit Rate@10 of 0.2 (vs 0.5 random baseline on a same-domain candidate set) flags a clear limitation: the hybrid scorer's interest-overlap signal is too narrow when distractors share the target's domain. Future work: add an LLM-driven reranking layer.

### Behavioural fidelity (A/B history harness)

Quantifies how much the silent historical-context step moves the needle. For every eligible user (≥2 corpus entries) it holds out the last review and runs the agent twice — once with history, once without — then scores each generated review for rating error, TF cosine similarity, tone match, and a composite fidelity number.

```bash
# Against a locally running backend
python scripts/eval_fidelity.py --limit 20

# Against the deployed Koyeb backend
python scripts/eval_fidelity.py \
  --base-url https://youthful-wynn-taotechs-6715c87e.koyeb.app \
  --limit 30
```

Outputs go to `data/eval/`:

- `fidelity_results.jsonl` — per-sample raw scores for both modes.
- `fidelity_summary.json`  — aggregate means + the `delta` between modes.

Full methodology, metric definitions, and interpretation guide in [`docs/EVAL.md`](docs/EVAL.md).

### Smoke harnesses (judges can run these in under a minute)

```bash
# Calibration check: vague vs rich input, observe critique pass behaviour
python scripts/smoke_critique.py

# End-to-end HTTP smoke against a running API
python scripts/smoke_api.py http://127.0.0.1:8000
```

---

## Tests

```bash
pytest -q
```

6/6 currently passing on the main branch.

---

## Project layout

```text
.
├── agents/                  # User modeling, review generation (with critique loop), recommendation
├── api/                     # FastAPI app + unified agent gateway route
├── core/                    # Orchestrator + LangChain intent router
├── data/                    # Processed corpus + benchmark results
├── data_pipeline/           # Normalisation schemas (Yelp / Amazon / Goodreads)
├── docs/                    # Solution paper (SOLUTION_PAPER.md) + template
├── evaluation/              # ROUGE / BERTScore-fallback / RMSE / NDCG / Hit Rate metrics
├── frontend/                # Next.js 15 unified chat UI
├── memory/                  # In-memory vector store + corpus store + user memory
├── models/                  # Role-aware LLM wrapper (router vs generator)
├── scripts/                 # Corpus builder, benchmark + ablation runner, smoke tests
├── tests/
├── utils/                   # Config, schemas, logger
├── Dockerfile / Dockerfile.frontend / docker-compose.yml
└── main.py
```

---

## Submission checklist

- ✅ Task A containerised app (API + Web)
- ✅ Task B containerised app (API + Web)
- ✅ Cold-start, cross-domain, multi-turn support
- ✅ Datasets normalisation for Yelp / Amazon / Goodreads (all represented in the eval corpus)
- ✅ Evaluation scripts with ROUGE / BERTScore (token-F1 fallback) / RMSE / NDCG@10 / Hit Rate@10
- ✅ Ablation runner (no-RAG / no-critique / no-LLM) with real numbers
- ✅ Behavioural-fidelity A/B harness ([`scripts/eval_fidelity.py`](scripts/eval_fidelity.py) + [`docs/EVAL.md`](docs/EVAL.md))
- ✅ Streaming reasoning gateway (`POST /api/agent/v1/stream`) + animated live timeline UI
- ✅ Multi-language output: English · Nigerian Pidgin · English + Yoruba mix
- ✅ Advisory safety / validation layer with `safety_flags` on every response
- ✅ User feedback loop (thumbs up/down → JSONL log + stats endpoint)
- ✅ Solution paper at [`docs/SOLUTION_PAPER.md`](docs/SOLUTION_PAPER.md)
- ✅ Reproducible Docker Compose stack
- ✅ Public production deploy: Vercel (frontend) + Koyeb (backend)
- ✅ Nigerian contextualisation in tone + retrieval seed data
