/** Backend origin for hackathon task routes (no /api/v1 suffix). */
export function getApiRoot(): string {
  const raw = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "").trim();
  if (!raw) return "";
  return raw.replace(/\/api\/v1\/?$/i, "").replace(/\/$/, "");
}
