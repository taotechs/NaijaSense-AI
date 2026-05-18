/**
 * API base URL for browser fetch calls.
 *
 * On Vercel (and local Next dev), always use same-origin paths so
 * `next.config.ts` rewrites proxy to Koyeb. Calling Koyeb directly from
 * the browser fails with CORS, cold-start, or 404 when env URLs are wrong.
 */
export function getApiRoot(): string {
  if (typeof window !== "undefined") {
    return "";
  }
  const raw = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "").trim();
  if (!raw) return "";
  return raw.replace(/\/api\/v1\/?$/i, "").replace(/\/$/, "");
}

/** Origin shown in UI / README (submission links). */
export function getPublicOrigin(): string {
  if (typeof window !== "undefined") {
    return window.location.origin;
  }
  return (
    process.env.NEXT_PUBLIC_SITE_URL?.replace(/\/$/, "") ||
    "https://naija-sense-ai.vercel.app"
  );
}

export function publicTaskUrl(path: string): string {
  const p = path.startsWith("/") ? path : `/${path}`;
  return `${getPublicOrigin()}${p}`;
}
