import Link from "next/link";
import { publicTaskUrl } from "@/lib/api-root";

export default function HomePage() {
  const taskAApi = publicTaskUrl("/task-a/user-modeling");
  const taskBApi = publicTaskUrl("/task-b/recommendation");

  return (
    <div className="space-y-8">
      <section>
        <p className="text-[11px] uppercase tracking-[0.22em] text-brand-500">
          Dual-link submission
        </p>
        <h2 className="mt-2 text-xl font-semibold text-slate-100">
          Two tasks — each with its own demo screen
        </h2>
        <p className="mt-2 text-sm text-slate-400">
          Use the API URLs as your agent links. Open the demo screens below to
          test Task A and Task B in the browser (no Swagger required).
        </p>
      </section>

      <div className="grid gap-4 sm:grid-cols-2">
        <article className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
          <h3 className="font-semibold text-slate-100">Task A — User modeling</h3>
          <p className="mt-2 text-xs text-slate-400">
            Persona + product → rating + review.
          </p>
          <p className="mt-2 break-all font-mono text-[10px] text-slate-500">
            POST {taskAApi || "/task-a/user-modeling"}
          </p>
          <Link
            href="/task-a"
            className="mt-4 inline-flex rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-500"
          >
            Open Task A demo →
          </Link>
        </article>

        <article className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
          <h3 className="font-semibold text-slate-100">Task B — Recommendation</h3>
          <p className="mt-2 text-xs text-slate-400">
            Persona → ranked list + chain-of-thought.
          </p>
          <p className="mt-2 break-all font-mono text-[10px] text-slate-500">
            POST {taskBApi || "/task-b/recommendation"}
          </p>
          <Link
            href="/task-b"
            className="mt-4 inline-flex rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-500"
          >
            Open Task B demo →
          </Link>
        </article>
      </div>

      <section className="rounded-xl border border-dashed border-slate-700 bg-slate-950/50 p-5">
        <h3 className="text-sm font-medium text-slate-300">Unified agent (optional)</h3>
        <p className="mt-1 text-sm text-slate-500">
          One chat that auto-routes between Task A and B with a live reasoning timeline.
        </p>
        <Link
          href="/unified"
          className="mt-3 inline-block text-sm text-brand-500 hover:underline"
        >
          Open Behavioral Intelligence Hub →
        </Link>
      </section>
    </div>
  );
}
