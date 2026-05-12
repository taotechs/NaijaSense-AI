# Behavioral Fidelity Evaluation

NaijaSense AI's headline claim is that the **silent historical-context retrieval** step makes generated reviews more faithful to a user's real past behaviour than a vanilla persona-only pipeline. This document explains how we measure that claim and how to reproduce the numbers.

## What we measure

For every eligible user (anyone in the corpus with ≥2 reviews) we:

1. **Hold out the user's last review** as ground truth.
2. **Run the agent twice** on the same query, generated from the held-out item:
   - **With history** (`include_history=True`) — full production pipeline. Silent retrieval pulls the user's earlier reviews, derives a behavioural baseline persona, and conditions the generator on it.
   - **Without history** (`include_history=False`) — same pipeline, silent retrieval skipped. The UI-supplied persona is the only signal.
3. **Score each generated review** against the held-out ground truth on three axes:

| Metric | What it captures | Range |
| --- | --- | --- |
| `rating_error` | `|predicted - actual|` star rating | 0 (perfect) → 4 (worst) |
| `text_cosine`  | TF cosine similarity between generated and real review tokens | 0 → 1 (higher = more aligned) |
| `tone_match`   | Boolean: does the generated tone bucket (slang / casual / formal) match the user's true tone bucket? | 0 or 1 |
| `fidelity`     | Composite: `0.4 * (1 - rating_err/4) + 0.4 * text_cosine + 0.2 * tone_match` | 0 → 1 |

The composite is intentionally biased toward rating + textual alignment because tone is already a noisy bucket — we keep it as a 20% tiebreaker.

## Running the eval

```bash
# Against a locally running backend
python scripts/eval_fidelity.py --limit 20

# Against the deployed Koyeb backend
python scripts/eval_fidelity.py --limit 30 --base-url https://youthful-wynn-taotechs-6715c87e.koyeb.app
```

Outputs are written to `data/eval/`:

- `fidelity_results.jsonl` — per-sample raw scores for both modes.
- `fidelity_summary.json`  — aggregated means + the **delta** between modes.

A non-zero `delta.fidelity` in favour of `with_history` is the proof point.

## Interpreting results

The summary block looks like:

```json
{
  "with_history":    { "n": 20, "fidelity_mean": 0.61, "rating_error_mae": 0.84, "text_cosine_mean": 0.41, "tone_match_pct": 70.0 },
  "without_history": { "n": 20, "fidelity_mean": 0.47, "rating_error_mae": 1.32, "text_cosine_mean": 0.29, "tone_match_pct": 55.0 },
  "delta":           { "fidelity": 0.14, "rating_error": 0.48, "text_cosine": 0.12, "tone_match_pct": 15.0 }
}
```

Interpretation:

- **`delta.fidelity = +0.14`** — silent retrieval improves the composite score by 14 fidelity points on average.
- **`delta.rating_error = +0.48`** — predicted rating is **0.48 stars closer** to truth when history is used.
- **`delta.text_cosine = +0.12`** — generated text shares meaningfully more vocabulary with the user's real review.
- **`delta.tone_match_pct = +15.0`** — tone bucket alignment goes up 15 percentage points.

The numbers above are illustrative — your run will vary depending on Groq model, sample size, and seed.

## Caveats

- The TF cosine is a lightweight stand-in for embedding similarity. For final reporting we'd swap to sentence-embedding cosine, but the relative deltas track.
- Held-out evaluation can leak when a user's earlier reviews mention the same item; in practice the corpus is item-disjoint enough that this is rare.
- The eval issues **two requests per sample**, so a 30-sample run is ~60 calls. Mind your Groq free-tier rate limit if you push `--limit` high.
