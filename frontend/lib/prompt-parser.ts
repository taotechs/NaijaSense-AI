type ChatMode = "review" | "recommend";

/** Reject keyboard-mash / non-word tokens from interest extraction. */
export function isPlausibleWord(token: string): boolean {
  const t = token.toLowerCase();
  if (t.length < 3 || t.length > 22) return false;
  if (!/[aeiouy]/.test(t)) return false;
  const vowels = (t.match(/[aeiouy]/g) ?? []).length;
  if (t.length >= 6 && vowels / t.length < 0.18) return false;
  let cons = 0;
  let maxCons = 0;
  for (const ch of t) {
    if (/[aeiouy]/.test(ch)) {
      cons = 0;
    } else {
      cons += 1;
      maxCons = Math.max(maxCons, cons);
    }
  }
  if (maxCons > 4) return false;
  return true;
}

/** Pick a human-readable item name for review simulation from chat text + interests. */
export function extractReviewItemName(prompt: string, interests: string[]): string {
  const trimmed = prompt.trim();
  if (!trimmed) return "General Product";

  const quoted = trimmed.match(/["']([^"']{2,60})["']/);
  if (quoted?.[1]) {
    const inner = quoted[1].trim();
    if (inner.length >= 2) return inner.charAt(0).toUpperCase() + inner.slice(1);
  }

  const patterns: RegExp[] = [
    /\b(?:review|rate)\s+(?:for|about|on)\s+(.+?)(?:[.?!]|$)/i,
    /\b(?:try(?:ing)?|tried)\s+(.+?)(?:\s+and|\s+at|\s+in\b|[.?!]|$)/i,
    /\babout\s+(?:the\s+)?(.+?)(?:[.?!]|$)/i,
    /\bfor\s+(.+?)\s+(?:restaurant|spot|place|buka|product)\b/i
  ];
  for (const p of patterns) {
    const m = trimmed.match(p);
    if (m?.[1]) {
      let cand = m[1].replace(/\s+/g, " ").trim().replace(/^(the|a|an)\s+/i, "").slice(0, 80);
      if (cand.length >= 2) {
        cand = cand.replace(/\s+option$/i, "");
        return cand.charAt(0).toUpperCase() + cand.slice(1);
      }
    }
  }

  const firstLine = trimmed.split(/\n/)[0] ?? trimmed;
  const words = firstLine
    .split(/\s+/)
    .map((w) => w.replace(/[^a-zA-Z']/g, "").toLowerCase())
    .filter((w) => w.length >= 3 && !STOPWORDS.has(w));
  const concrete = words.find((w) => isPlausibleWord(w) && !GENERIC_INTERESTS.has(w));
  if (concrete) {
    return concrete.charAt(0).toUpperCase() + concrete.slice(1);
  }

  const goodInterest = interests.find(
    (i) =>
      i !== "general lifestyle" &&
      !GENERIC_INTERESTS.has(i) &&
      (/\s/.test(i) || isPlausibleWord(i))
  );
  if (goodInterest) {
    return goodInterest.charAt(0).toUpperCase() + goodInterest.slice(1);
  }

  return "General Product";
}

export function inferChatMode(prompt: string): ChatMode {
  const text = prompt.toLowerCase();
  if (
    text.includes("recommend") ||
    text.includes("suggest") ||
    text.includes("cheap") ||
    text.includes("restaurant") ||
    text.includes("where should i")
  ) {
    return "recommend";
  }
  return "review";
}

export function extractInterests(prompt: string): string[] {
  const text = normalize(prompt);
  const mapped: string[] = [];

  const dictionary: Array<[string, string]> = [
    ["amala", "amala"],
    ["gbegiri", "gbegiri"],
    ["spicy", "spicy food"],
    ["food", "food"],
    ["restaurant", "restaurants"],
    ["buka", "restaurants"],
    ["tech", "tech"],
    ["gadget", "gadgets"],
    ["fashion", "fashion"],
    ["music", "music"],
    ["travel", "travel"]
  ];
  for (const [needle, interest] of dictionary) {
    if (text.includes(needle)) mapped.push(interest);
  }

  const tokenInterests = tokenize(text).filter(
    (t) => t.length > 3 && !STOPWORDS.has(t) && /^[a-z]+$/.test(t) && isPlausibleWord(t)
  );
  const combined = [...mapped, ...tokenInterests].slice(0, 8);
  return combined.length > 0 ? [...new Set(combined)] : ["general lifestyle"];
}

export function inferSentimentBias(prompt: string): "positive" | "balanced" | "critical" {
  const text = prompt.toLowerCase();
  if (text.includes("bad") || text.includes("no try") || text.includes("terrible")) {
    return "critical";
  }
  if (text.includes("love") || text.includes("great") || text.includes("slap")) {
    return "positive";
  }
  return "balanced";
}

export function inferPersonaStyle(prompt: string): "nigerian_twitter" | "formal" {
  const text = prompt.toLowerCase();
  if (text.includes("omo") || text.includes("abeg") || text.includes("sha")) {
    return "nigerian_twitter";
  }
  return "nigerian_twitter";
}

export function extractCandidateItems(prompt: string): string[] {
  const text = normalize(prompt);
  const candidates: string[] = [];

  if (text.includes("amala") || text.includes("gbegiri")) {
    candidates.push(
      "Iya Aladura Amala Spot",
      "Gbegiri Joint Express",
      "Local Buka Combo",
      "Abula Kitchen Hub"
    );
  }

  if (text.includes("spicy") || text.includes("restaurant") || text.includes("food")) {
    candidates.push(
      "Pepper Grill Corner",
      "Budget Buka",
      "Street Shawarma Hub",
      "Mama Put Special"
    );
  }

  if (text.includes("cheap") || text.includes("budget")) {
    candidates.push("Wallet Saver Combo", "Daily Budget Picks");
  }

  if (text.includes("tech") || text.includes("gadget")) {
    candidates.push(
      "Budget Earbuds",
      "Smartwatch Pro",
      "Portable Charger",
      "Mechanical Keyboard"
    );
  }

  if (candidates.length === 0) {
    const tokens = tokenize(text).filter((t) => t.length > 3 && !STOPWORDS.has(t) && isPlausibleWord(t));
    for (const token of tokens.slice(0, 4)) {
      candidates.push(`${capitalize(token)} Popular Pick`);
    }
  }

  if (candidates.length === 0) {
    candidates.push("Daily Essentials Pack", "Urban Saver Choice", "Popular Local Pick");
  }

  return [...new Set(candidates)].slice(0, 8);
}

const STOPWORDS = new Set([
  "i",
  "like",
  "want",
  "need",
  "with",
  "and",
  "the",
  "for",
  "that",
  "this",
  "from",
  "cheap",
  "good",
  "best"
]);

/** Too vague to use alone as an item name (prefer a noun from the message). */
const GENERIC_INTERESTS = new Set([
  "food",
  "tech",
  "music",
  "travel",
  "fashion",
  "gadgets",
  "restaurants",
  "general",
  "lifestyle"
]);

function normalize(text: string): string {
  return text.toLowerCase().trim();
}

function tokenize(text: string): string[] {
  return normalize(text)
    .split(/[^a-z0-9]+/g)
    .map((t) => t.trim())
    .filter(Boolean);
}

function capitalize(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}
