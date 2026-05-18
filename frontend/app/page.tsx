import Link from "next/link";

// On Vercel, leave NEXT_PUBLIC_API_BASE_URL unset so links use same-origin
// rewrites (see frontend/next.config.ts). Local dev: set to http://127.0.0.1:8000
const apiBase =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? "";

export default function HomePage() {
  const taskA = `${apiBase}/task-a/user-modeling`;
  const taskB = `${apiBase}/task-b/recommendation`;
  const docs = `${apiBase}/docs`;

  return (
    <div className="space-y-8">
      <section>
        <p className="text-[11px] uppercase tracking-[0.22em] text-brand-500">
          Dual-link submission
        </p>
        <h2 className="mt-2 text-xl font-semibold text-slate-100">
          Two endpoints for the hackathon form
        </h2>
        <p className="mt-2 text-sm text-slate-400">
          Submit separate URLs for Task A (user modeling) and Task B
          (recommendation). Each accepts a straightforward JSON body and returns
          the expected output shape.
        </p>
      </section>

      <div className="grid gap-4 sm:grid-cols-2">
        <article className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
          <h3 className="font-semibold text-slate-100">Task A — User modeling</h3>
          <p className="mt-2 text-xs text-slate-400">
            Input: user persona + product details. Output: rating + review.
          </p>
          <p className="mt-3 break-all font-mono text-[11px] text-brand-400">
            POST {taskA}
          </p>
          <Link
            href={docs}
            className="mt-3 inline-block text-sm text-brand-500 hover:underline"
            target="_blank"
            rel="noopener noreferrer"
          >
            Try in Swagger →
          </Link>
        </article>

        <article className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
          <h3 className="font-semibold text-slate-100">Task B — Recommendation</h3>
          <p className="mt-2 text-xs text-slate-400">
            Input: user persona. Output: ranked list + chain-of-thought.
          </p>
          <p className="mt-3 break-all font-mono text-[11px] text-brand-400">
            POST {taskB}
          </p>
          <Link
            href={docs}
            className="mt-3 inline-block text-sm text-brand-500 hover:underline"
            target="_blank"
            rel="noopener noreferrer"
          >
            Try in Swagger →
          </Link>
        </article>
      </div>

      <section className="rounded-xl border border-dashed border-slate-700 bg-slate-950/50 p-5">
        <h3 className="text-sm font-medium text-slate-300">Interactive demo</h3>
        <p className="mt-1 text-sm text-slate-500">
          Full agentic hub with live reasoning timeline (unified gateway).
        </p>
        <Link
          href="/unified"
          className="mt-3 inline-flex items-center rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-500"
        >
          Open Behavioral Intelligence Hub
        </Link>
      </section>
    </div>
  );
}
