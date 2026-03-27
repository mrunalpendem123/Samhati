import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Samhati — Open Intelligence for Everyone, Forever",
  description: "The BitTorrent of Intelligence. Every device that uses AI also provides AI. No corporation in the middle.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="antialiased">
      <body>{children}</body>
    </html>
  );
}
