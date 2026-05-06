import Link from "next/link";

const cards = [
  {
    title: "Chat Copilot (Recommended)",
    href: "/chat",
    description: "ChatGPT-style experience: type naturally, get recommendations/reviews plus reasoning."
  },
  {
    title: "Review Simulation",
    href: "/review",
    description: "Generate persona-aware product reviews with rating and cultural tone."
  },
  {
    title: "Recommendations",
    href: "/recommend",
    description: "Get ranked items with explainability and conversational recommendation style."
  },
  {
    title: "Agent Reasoning",
    href: "/reasoning",
    description: "Inspect step-by-step reasoning traces from backend orchestration."
  }
];

export default function HomePage() {
  return (
    <section className="space-y-8">
      <div className="glass rounded-3xl p-8 md:p-12">
        <p className="text-xs uppercase tracking-[0.2em] text-brand-500">Welcome</p>
        <h2 className="mt-2 text-3xl font-bold md:text-5xl">
          Modern Frontend for <span className="text-brand-500">NaijaSense AI</span>
        </h2>
        <p className="mt-4 max-w-2xl text-slate-300">
          Explore simulation, recommendation, and transparent agent reasoning in a responsive,
          demo-ready interface wired to your FastAPI backend.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {cards.map((card) => (
          <Link key={card.href} href={card.href} className="glass rounded-2xl p-5 transition hover:-translate-y-1">
            <h3 className="text-lg font-semibold text-slate-100">{card.title}</h3>
            <p className="mt-2 text-sm text-slate-400">{card.description}</p>
            <p className="mt-4 text-sm font-medium text-brand-500">Open page -&gt;</p>
          </Link>
        ))}
      </div>
    </section>
  );
}
