# Live deployment guide

Goal: publish a public URL with FastAPI on one host and Next.js on another. Both tiers can be deployed on free plans.

| Layer | Host | Reason |
|---|---|---|
| Frontend (Next.js) | **Vercel** | Made by the Next.js team; zero-config deploy from GitHub. |
| Backend (FastAPI + Groq) | **Koyeb** | Free Docker web service, no credit card, always-on (no cold starts), GitHub-driven. |

This repository already includes:

- `koyeb.yaml` — Koyeb app definition, deploys the existing `Dockerfile` (primary path).
- `render.yaml` — Render blueprint, also deploys the existing `Dockerfile` (alternative path for Render deployments).
- `Dockerfile` — honors `$PORT`, portable across Koyeb / Render / Fly / HF Spaces / local compose.
- CORS in `api/app.py` — driven by `CORS_ORIGINS` env so deployment domains can be whitelisted.

---

## 0. Prerequisites

1. Ensure the repository is available on GitHub. If not yet pushed, run `git push -u origin main`.
2. Free Groq API key: <https://console.groq.com/keys> (sign in with Google → "Create API Key" → copy `gsk_…`).

Required accounts (all support free tiers and GitHub sign-in):

- GitHub: <https://github.com>
- Vercel (frontend): <https://vercel.com/signup>
- Koyeb (backend): <https://app.koyeb.com/auth/signup>

---

## 1. Deploy the backend on Koyeb (≈ 5 minutes)

1. Go to <https://app.koyeb.com/auth/signup> and sign in with GitHub.
2. From the dashboard click **Create App** → **GitHub** → authorize → pick this repo, branch `main`.
3. In the build step, choose **Dockerfile** (Koyeb auto-detects the root `Dockerfile`).
4. Instance type: **Free** (Eco Nano). Region: select the closest region for expected users (for example Washington `was` or Frankfurt `fra`).
5. **Exposed port**: `8000` (Koyeb also injects `$PORT` automatically; the Dockerfile handles either). Health-check path: `/api/v1/health`.
6. Open **Environment variables** and add values listed in `koyeb.yaml`. Required secrets are:
   - `GROQ_API_KEY` → add a valid `gsk_...` key
   - `CORS_ORIGINS` → leave empty initially; set it after the frontend URL is available
   Other variables (`ORCHESTRATOR_PROVIDER=groq`, model names, sampling, critique flags) can be entered manually or via **App Settings → Definition** from `koyeb.yaml`.
7. Click **Deploy**. After a successful build, Koyeb provides a public URL like `https://naijasense-ai-api-<org>.koyeb.app`. Save this value.
8. Validate backend health: `https://<koyeb-url>/api/v1/health` should return `{"status":"ok"}`.

> Free-tier note: Koyeb's Eco Nano is **always-on** (no cold starts). Compared with Render free tier, it offers lower CPU/RAM headroom but is typically sufficient for this architecture because LLM inference runs on Groq.

> If Render is preferred, use `render.yaml` instead. See "Alternative hosts" below.

---

## 2. Deploy the frontend on Vercel (≈ 3 minutes)

1. Go to <https://vercel.com/new>.
2. Click **Import** on this repo.
3. In the import wizard:
   - **Root Directory**: click **Edit** → select `frontend` (this is the only non-default field).
   - **Framework Preset**: Vercel auto-detects Next.js — leave as-is.
4. Open **Environment Variables** and add:
   - `NEXT_PUBLIC_AGENT_API_URL` = `https://<koyeb-url>/api/agent/v1`
     (e.g. `https://youthful-wynn-taotechs-6715c87e.koyeb.app/api/agent/v1`)
   - Optional: `NEXT_PUBLIC_API_BASE_URL` — leave **unset** on Vercel so the home page
     shows same-origin hackathon URLs (`/task-a/...`, `/task-b/...` proxied to Koyeb).
5. Click **Deploy**.
6. Vercel provides a URL like `https://naijasense-ai.vercel.app`. Save this value.

---

## 3. Wire the two together (1 minute)

In Koyeb, open the API service → **Settings → Environment variables**:

- `CORS_ORIGINS` = `https://naijasense-ai.vercel.app` (the Vercel URL from step 2)

Hit **Save & Redeploy**. Koyeb redeploys in ~30 s.

Open the Vercel URL and send a test query to confirm end-to-end connectivity with the live Groq-backed API.

---

## 4. Smoke test the live system

```bash
# Backend health
curl https://<koyeb-url>/api/v1/health

# Full Task A round-trip via the agent gateway
curl -X POST https://<koyeb-url>/api/agent/v1 \
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

## 5. Deployment handoff values

Record these values for documentation and operational handoff:

| Field | Value |
|---|---|
| GitHub repo | `https://github.com/taotechs/NaijaSense-AI.git` |
| Live demo (frontend) | `https://naija-sense-ai.vercel.app` |
| Hackathon Task A (POST) | `https://naija-sense-ai.vercel.app/task-a/user-modeling` |
| Hackathon Task B (POST) | `https://naija-sense-ai.vercel.app/task-b/recommendation` |
| API endpoint (unified demo) | `https://<koyeb-url>/api/agent/v1` |
| Docker setup | Already covered by `Dockerfile` + `docker-compose.yml` in the repo |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Vercel build fails with "Cannot find module" | Set Root Directory to `frontend`. Re-import or edit project settings. |
| Browser console: `CORS policy: No 'Access-Control-Allow-Origin'` | `CORS_ORIGINS` on the backend does not exactly match the Vercel URL (no trailing slash; correct protocol). |
| Frontend posts to `127.0.0.1:8000` in production | `NEXT_PUBLIC_AGENT_API_URL` not set on Vercel. Vercel needs a redeploy after adding env vars. |
| Koyeb service shows "deploy failed: no open ports" | Ensure the latest `Dockerfile` is in use; the CMD must use `${PORT:-8000}`. |
| LLM responses sound flat / repetitive | `GROQ_API_KEY` missing or invalid. Check the service Logs for `LLMWrapper provider=none`. |

---

## Alternative backend hosts

The backend is a Docker image, so these options work **without code changes**. The `$PORT`-aware Dockerfile in this repository is portable across providers.

### Render (when free quota is available)

`render.yaml` is already in the repo. Steps:
1. <https://dashboard.render.com/select-repo?type=blueprint> → connect GitHub → pick this repo → **Apply**.
2. Set `GROQ_API_KEY` and `CORS_ORIGINS` in the **Environment** tab.
3. Copy the `*.onrender.com` URL into Vercel's `NEXT_PUBLIC_AGENT_API_URL`.

Caveat: Render free-tier services sleep after ~15 minutes of idle time and may cold-start in 30–50 seconds. Call `/api/v1/health` before demonstrations.

### Fly.io (most production-like, requires CC on file)

1. Install flyctl: `iwr https://fly.io/install.ps1 -useb | iex` (Windows) or `curl -L https://fly.io/install.sh | sh`.
2. From the repo root: `fly launch --copy-config --no-deploy` → answer the prompts (org, region, no Postgres, no Redis).
3. `fly secrets set GROQ_API_KEY=gsk_... CORS_ORIGINS=https://<vercel-app>.vercel.app`
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

<https://railway.com/new> → "Deploy from GitHub repo" → set the same env vars. Railway auto-detects the Dockerfile and injects `$PORT`. This path has a straightforward setup UX but consumes trial credit (~$3/month for a small always-on service).

---

## Alternative frontend hosts

**Netlify** (<https://app.netlify.com/start>) is the obvious Vercel alternative: same flow, set base directory to `frontend/`, add `NEXT_PUBLIC_AGENT_API_URL`.

**Cloudflare Pages** also works for Next.js; pick the "Next.js" preset, root `frontend/`.
