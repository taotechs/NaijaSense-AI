import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NaijaSense AI Frontend",
  description: "Modern interface for NaijaSense AI multi-agent backend."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <main className="mx-auto max-w-4xl px-4 py-8">{children}</main>
      </body>
    </html>
  );
}
