"use client";
import Particles from "./Particles";

export default function Home() {
  return (
    <div className="relative w-screen h-screen overflow-hidden" style={{ background: "#F0EEEB" }}>
      <Particles />

      <div className="fixed inset-0 z-10 pointer-events-none flex flex-col justify-between p-8 md:p-12">
        <div className="flex items-center justify-between pointer-events-auto">
          <span></span>
          <a href="https://github.com/mrunalpendem123/Samhati" target="_blank" rel="noopener noreferrer" className="text-[12px] text-[#999] hover:text-[#1A1A1A] transition-colors">
            GitHub →
          </a>
        </div>

        <div className="flex items-end justify-between pointer-events-auto">
          <div className="flex gap-6 text-[12px] text-[#AAA]">
            <a href="https://github.com/mrunalpendem123/Samhati" className="hover:text-[#1A1A1A] transition-colors">About</a>
            <a href="https://github.com/mrunalpendem123/Samhati#readme" className="hover:text-[#1A1A1A] transition-colors">Docs</a>
          </div>
          <p className="text-[11px] text-[#BBB]">Solana · TOPLOC · MIT</p>
        </div>
      </div>
    </div>
  );
}
