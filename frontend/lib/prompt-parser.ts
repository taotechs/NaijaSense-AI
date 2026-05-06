type ChatMode = "review" | "recommend";

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
    (t) => t.length > 3 && !STOPWORDS.has(t) && /^[a-z]+$/.test(t)
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
    const tokens = tokenize(text).filter((t) => t.length > 3 && !STOPWORDS.has(t));
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
