"""Browser-friendly GET pages for hackathon POST endpoints."""

from __future__ import annotations

import json


def task_endpoint_html(
    *,
    task_name: str,
    path: str,
    method: str = "POST",
    description: str,
    example_body: dict,
    docs_path: str = "/docs",
) -> str:
    example_json = json.dumps(example_body, indent=2)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NaijaSense AI — {task_name}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0;
           max-width: 720px; margin: 0 auto; padding: 1.5rem; line-height: 1.5; }}
    .ok {{ color: #4ade80; font-weight: 600; }}
    .warn {{ background: #422006; border: 1px solid #b45309; padding: 0.75rem 1rem;
             border-radius: 8px; margin: 1rem 0; }}
    pre {{ background: #1e293b; padding: 1rem; overflow-x: auto; border-radius: 8px;
          font-size: 0.85rem; }}
    a {{ color: #7dd3fc; }}
    code {{ background: #1e293b; padding: 0.1rem 0.35rem; border-radius: 4px; }}
  </style>
</head>
<body>
  <p class="ok">✓ Endpoint is live</p>
  <h1>{task_name}</h1>
  <p>{description}</p>
  <div class="warn">
    <strong>This URL is an API endpoint.</strong> Opening it in a browser sends
    <code>GET</code>, but the hackathon contract requires <code>{method}</code> with a JSON body.
    Judges should use Swagger, curl, or their evaluation harness — not the address bar alone.
  </div>
  <h2>How to call it</h2>
  <p><code>{method} {path}</code><br/>Content-Type: <code>application/json</code></p>
  <h2>Example body</h2>
  <pre>{example_json}</pre>
  <p><a href="{docs_path}">Open Swagger → try it interactively</a> ·
     <a href="/">← Submission home</a></p>
</body>
</html>"""
