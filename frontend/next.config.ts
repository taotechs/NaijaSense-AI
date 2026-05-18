import type { NextConfig } from "next";

/** FastAPI host (Koyeb/Render/local). Used to proxy hackathon task routes on Vercel. */
function backendOrigin(): string {
  const agent = process.env.NEXT_PUBLIC_AGENT_API_URL;
  if (agent) {
    try {
      return new URL(agent).origin;
    } catch {
      /* fall through */
    }
  }
  const base = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_API_BASE_URL;
  if (base) {
    return base.replace(/\/api\/v1\/?$/, "").replace(/\/$/, "");
  }
  return "http://127.0.0.1:8000";
}

const backend = backendOrigin();

const nextConfig: NextConfig = {
  reactStrictMode: true,
  output: "standalone",
  async rewrites() {
    return [
      {
        source: "/task-a/user-modeling",
        destination: `${backend}/task-a/user-modeling`,
      },
      {
        source: "/task-b/recommendation",
        destination: `${backend}/task-b/recommendation`,
      },
      { source: "/docs", destination: `${backend}/docs` },
      { source: "/docs/:path*", destination: `${backend}/docs/:path*` },
      { source: "/openapi.json", destination: `${backend}/openapi.json` },
      { source: "/api/v1/health", destination: `${backend}/api/v1/health` },
    ];
  },
};

export default nextConfig;
