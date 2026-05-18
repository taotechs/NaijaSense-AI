"""Root landing page — dual submission links for judges."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["landing"])


def _landing_html(*, api_base: str = "") -> str:
    base = api_base.rstrip("/")
    task_a_api = f"{base}/task-a/user-modeling"
    task_b_api = f"{base}/task-b/recommendation"
    task_a_ui = "/task-a"
    task_b_ui = "/task-b"
    docs = f"{base}/docs"
    health = f"{base}/api/v1/health"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NaijaSense AI — DSN × BCT LLM Agent Challenge</title>
  <style>
    :root {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; background: #0f172a; color: #e2e8f0; }}
    body {{ max-width: 720px; margin: 0 auto; padding: 2rem 1.25rem; line-height: 1.55; }}
    h1 {{ font-size: 1.75rem; margin-bottom: 0.25rem; }}
    .tag {{ color: #38bdf8; font-size: 0.75rem; letter-spacing: 0.12em; text-transform: uppercase; }}
    .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 1.25rem; margin: 1rem 0; }}
    .card h2 {{ margin: 0 0 0.5rem; font-size: 1.1rem; }}
    a {{ color: #7dd3fc; word-break: break-all; }}
    code {{ background: #0f172a; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.9em; }}
    ul {{ padding-left: 1.2rem; }}
    footer {{ margin-top: 2rem; font-size: 0.85rem; color: #64748b; }}
  </style>
</head>
<body>
  <p class="tag">TAOTECH SOLUTIONS · NaijaSense AI</p>
  <h1>Dual-Link API Submission</h1>
  <p>Two containerized task endpoints for the hackathon judges. Use the links below in the submission form.</p>

  <div class="card">
    <h2>Task A — User modeling</h2>
    <p><strong>API (POST):</strong> <a href="{task_a_api}">{task_a_api}</a></p>
    <p><strong>Demo UI:</strong> <a href="{task_a_ui}">{task_a_ui}</a></p>
    <p>Output: <code>rating</code>, <code>review_reasoning</code>, <code>review_text</code></p>
  </div>

  <div class="card">
    <h2>Task B — Recommendation</h2>
    <p><strong>API (POST):</strong> <a href="{task_b_api}">{task_b_api}</a></p>
    <p><strong>Demo UI:</strong> <a href="{task_b_ui}">{task_b_ui}</a></p>
    <p>Input: <code>user_persona.persona</code> only · Output: <code>recommendations[]</code> + <code>agent_reasoning</code></p>
  </div>

  <div class="card">
    <h2>Also useful</h2>
    <ul>
      <li><a href="{docs}">OpenAPI / Swagger</a> — try both endpoints interactively</li>
      <li><a href="{health}">Health check</a></li>
      <li>Legacy unified agent: <code>POST /api/agent/v1</code></li>
      <li>Interactive hub UI: <code>/unified</code> (Next.js frontend)</li>
    </ul>
  </div>

  <footer>DSN × Bluechip Tech LLM Agent Challenge · DSAS 2026</footer>
</body>
</html>"""


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def landing_page() -> HTMLResponse:
    return HTMLResponse(_landing_html())
