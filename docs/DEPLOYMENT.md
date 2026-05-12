# Live deployment guide

Goal: get a public URL judges can hit, with the FastAPI backend on one
host and the Next.js frontend on another. Both legs are free.

| Layer | Host | Reason |
|---|---|---|
| Frontend (Next.js) | **Vercel** | Made by the Next.js team; zero-config deploy from GitHub. |
| Backend (FastAPI + Groq) | **Koyeb** | Free Docker web service, no credit card, always-on (no cold starts), GitHub-driven. |

The pieces this repo already ships for you:

- `koyeb.yaml` — Koyeb app definition, deploys the existing `Dockerfile` (primary path).
- `render.yaml` — Render blueprint, also deploys the existing `Dockerfile` (alternative, in case you prefer Render).
- `Dockerfile` — honors `$PORT`, portable across Koyeb / Render / Fly / HF Spaces / local compose.
- CORS in `api/app.py` — driven by `CORS_ORIGINS` env so you can whitelist the Vercel domain.

---

## 0. Prerequisites

1. The repo is already on GitHub. If not yet pushed, run `git push -u origin main`.
2. Free Groq API key: <https://console.groq.com/keys> (sign in with Google → "Create API Key" → copy `gsk_…`).

You'll need three accounts. All free, all sign in with GitHub:

- GitHub (already have it): <https://github.com>
- Vercel (frontend): <https://vercel.com/signup>
- Koyeb (backend): <https://app.koyeb.com/auth/signup>

---

## 1. Deploy the backend on Koyeb (≈ 5 minutes)

1. Go to <https://app.koyeb.com/auth/signup> and sign in with GitHub.
2. From the dashboard click **Create App** → **GitHub** → authorize → pick this repo, branch `main`.
3. In the build step, choose **Dockerfile** (Koyeb auto-detects the root `Dockerfile`).
4. Instance type: **Free** (Eco Nano). Region: pick closest to your judges (Washington `was` or Frankfurt `fra`).
5. **Exposed port**: `8000` (Koyeb also injects `$PORT` automatically; the Dockerfile handles either). Health-check path: `/api/v1/health`.
6. Open the **Environment variables** section and add the values listed in `koyeb.yaml` — the only secrets you must set yourself are:
   - `GROQ_API_KEY` → paste your `gsk_…`
   - `CORS_ORIGINS` → leave empty for now; we'll fill it once Vercel gives us a URL.
   The other variables (`ORCHESTRATOR_PROVIDER=groq`, model names, sampling, critique flags) you can either type in manually or paste the YAML from `koyeb.yaml` into **App Settings → Definition**.
7. Click **Deploy**. After the build is green Koyeb gives you a public URL like `https://naijasense-ai-api-<org>.koyeb.app`. **Copy it.**
8. Sanity check it: `https://<your-koyeb-url>/api/v1/health` should return `{"status":"ok"}`.

> Free-tier note: Koyeb's Eco Nano is **always-on** (no cold starts), so judges won't wait. The trade-off vs Render: lower CPU/RAM headroom, but more than enough for this app since Groq does all the heavy LLM work.

> Already have a Render slot free? You can skip this section and use `render.yaml` instead — see "Alternative hosts" below. Either way works.

---

## 2. Deploy the frontend on Vercel (≈ 3 minutes)

1. Go to <https://vercel.com/new>.
2. Click **Import** on this repo.
3. In the import wizard:
   - **Root Directory**: click **Edit** → select `frontend` (this is the only non-default field).
   - **Framework Preset**: Vercel auto-detects Next.js — leave as-is.
4. Open **Environment Variables** and add one:
   - `NEXT_PUBLIC_AGENT_API_URL` = `https://<your-koyeb-url>/api/agent/v1`
     (e.g. `https://naijasense-ai-api-<org>.koyeb.app/api/agent/v1`)
5. Click **Deploy**.
6. Vercel gives you a URL like `https://naijasense-ai.vercel.app`. **Copy it.**

---

## 3. Wire the two together (1 minute)

Go back to Koyeb → your API service → **Settings → Environment variables**:

- `CORS_ORIGINS` = `https://naijasense-ai.vercel.app` (the Vercel URL from step 2)

Hit **Save & Redeploy**. Koyeb redeploys in ~30 s.

That's it. Open the Vercel URL, send a query, watch the agentic workflow indicator, see the response from your live Groq-backed API.

---

## 4. Smoke test the live system

```bash
# Backend health
curl https://<your-koyeb-url>/api/v1/health

# Full Task A round-trip via the agent gateway
curl -X POST https://<your-koyeb-url>/api/agent/v1 \
  -H "Content-Type: application/json" \
  -d '{
    "user_persona": {
      "user_id": "demo-user",
      "location": "Lagos",
      "interests": ["food", "nightlife"]
    },
    "query": "Review the new Suya spot in Ikeja — went on Friday, queue was long but yaji on point."
  }'
```

Expect `task: "review"`, a paragraph of `review_text`, and a `reasoning_steps` array that includes `silent_context_retrieval`.

---

## 5. Submit the URLs

In the hackathon submission form you'll typically be asked for:

| Field | Value |
|---|---|
| GitHub repo | `https://github.com/<you>/NaijaSense-AI` |
| Live demo (frontend) | `https://naijasense-ai.vercel.app` |
| API endpoint | `https://<your-koyeb-url>/api/agent/v1` |
| Docker setup | Already covered by `Dockerfile` + `docker-compose.yml` in the repo |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Vercel build fails with "Cannot find module" | You forgot to set Root Directory to `frontend`. Re-import or edit project settings. |
| Browser console: `CORS policy: No 'Access-Control-Allow-Origin'` | `CORS_ORIGINS` on the backend doesn't match your Vercel URL **exactly** (no trailing slash, https vs http). |
| Frontend posts to `127.0.0.1:8000` in production | `NEXT_PUBLIC_AGENT_API_URL` not set on Vercel. Vercel needs a redeploy after adding env vars. |
| Koyeb service shows "deploy failed: no open ports" | Make sure you pulled the updated `Dockerfile` — the CMD must use `${PORT:-8000}`. |
| LLM responses sound flat / repetitive | `GROQ_API_KEY` missing or invalid. Check the service Logs for `LLMWrapper provider=none`. |

---

## Alternative backend hosts

The backend is just a Docker image, so any of these work **without code changes** — the `$PORT`-aware Dockerfile already in the repo is portable. Pick whichever matches your account situation:

### Render (if you have free quota available)

`render.yaml` is already in the repo. Steps:
1. <https://dashboard.render.com/select-repo?type=blueprint> → connect GitHub → pick this repo → **Apply**.
2. Set `GROQ_API_KEY` and `CORS_ORIGINS` in the **Environment** tab.
3. Copy the `*.onrender.com` URL into Vercel's `NEXT_PUBLIC_AGENT_API_URL`.

Caveat: Render free-tier services sleep after ~15 min idle and cold-start in 30–50 s. Ping `/api/v1/health` before judging.

### Fly.io (most production-like, requires CC on file)

1. Install flyctl: `iwr https://fly.io/install.ps1 -useb | iex` (Windows) or `curl -L https://fly.io/install.sh | sh`.
2. From the repo root: `fly launch --copy-config --no-deploy` → answer the prompts (org, region, no Postgres, no Redis).
3. `fly secrets set GROQ_API_KEY=gsk_... CORS_ORIGINS=https://your-app.vercel.app`
4. `fly deploy`. URL is `https://<app>.fly.dev`.

### Hugging Face Spaces (Docker SDK, no CC, no cold starts)

1. <https://huggingface.co/new-space> → SDK = **Docker** → blank template.
2. Add this YAML frontmatter to the top of `README.md` on the Space repo:
   ```yaml
   ---
   title: NaijaSense AI
   emoji: "🇳🇬"
   colorFrom: green
   colorTo: white
   sdk: docker
   app_port: 8000
   ---
   ```
3. Push this repo to the Space's git remote, set `GROQ_API_KEY` + `CORS_ORIGINS` as Space secrets.
4. URL is `https://<user>-<space>.hf.space`.

### Railway ($5/mo trial credit)

<https://railway.com/new> → "Deploy from GitHub repo" → set the same env vars. Railway auto-detects the Dockerfile and injects `$PORT`. Easiest UX but uses your trial credit ~$3/mo for a small always-on service.

---

## Alternative frontend hosts

**Netlify** (<https://app.netlify.com/start>) is the obvious Vercel alternative: same flow, set base directory to `frontend/`, add `NEXT_PUBLIC_AGENT_API_URL`.

**Cloudflare Pages** also works for Next.js; pick the "Next.js" preset, root `frontend/`.
