"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/", label: "Home" },
  { href: "/chat", label: "Chat Copilot" },
  { href: "/review", label: "Review Simulation" },
  { href: "/recommend", label: "Recommendations" },
  { href: "/reasoning", label: "Agent Reasoning" }
];

export function NavBar() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-10 border-b border-slate-800/80 bg-slate-950/80 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-brand-500">NaijaSense AI</p>
          <h1 className="text-sm font-semibold text-slate-200">Context-Aware Agent UI</h1>
        </div>
        <nav className="flex flex-wrap items-center gap-2">
          {links.map((link) => {
            const active = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                  active
                    ? "bg-brand-500 text-slate-950"
                    : "bg-slate-900 text-slate-300 hover:bg-slate-800"
                }`}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
