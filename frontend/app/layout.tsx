import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NaijaSense AI: Behavioral Intelligence Hub",
  description:
    "Behavioral user modeling and contextual recommendation for the DSN \u00d7 BCT LLM Agent Challenge. Team: TAOTECH SOLUTIONS.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <header className="border-b border-slate-800/70 bg-slate-950/80 backdrop-blur-md">
          <div className="mx-auto flex max-w-4xl flex-col gap-2 px-4 py-5 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p className="text-[11px] uppercase tracking-[0.28em] text-brand-500">
                NaijaSense AI
              </p>
              <h1 className="text-2xl font-bold text-slate-100 sm:text-[1.65rem]">
                Behavioral Intelligence Hub
              </h1>
            </div>
            <p className="text-xs text-slate-400 sm:max-w-[18rem] sm:text-right">
              <span className="font-medium text-slate-300">Built for</span>{" "}
              DATA &amp; AI SUMMIT · HACKATHON 3.0{" "}
              <span className="text-slate-600">|</span>{" "}
              <span className="text-brand-500">
                DSN &times; BCT LLM Agent Challenge
              </span>
              <span className="mt-2 block text-[11px] uppercase tracking-[0.18em] text-slate-500">
                Team · TAOTECH SOLUTIONS
              </span>
            </p>
          </div>
        </header>
        <main className="mx-auto max-w-4xl px-4 py-8">{children}</main>
        <footer className="mx-auto max-w-4xl px-4 pb-8 text-center text-xs text-slate-600">
          User modeling (Task A) &amp; contextual recommendation (Task B), behind
          one agentic gateway.
          <span className="mt-2 block text-slate-500">
            Team · TAOTECH SOLUTIONS
          </span>
        </footer>
      </body>
    </html>
  );
}
