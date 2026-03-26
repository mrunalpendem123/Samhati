"use client";

import { motion, useScroll, useTransform } from "framer-motion";
import { useRef } from "react";

const fadeUp = {
  hidden: { opacity: 0, y: 40 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.8 } },
};

const stagger = {
  visible: { transition: { staggerChildren: 0.15 } },
};

function Section({ children, className = "", id }: { children: React.ReactNode; className?: string; id?: string }) {
  return (
    <motion.section
      id={id}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, margin: "-100px" }}
      variants={stagger}
      className={`relative max-w-5xl mx-auto px-6 md:px-12 py-24 md:py-40 ${className}`}
    >
      {children}
    </motion.section>
  );
}

function Terminal({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <motion.div variants={fadeUp} className="terminal-window glow-purple">
      <div className="terminal-titlebar">
        <div className="w-3 h-3 rounded-full bg-red-500/80" />
        <div className="w-3 h-3 rounded-full bg-yellow-500/80" />
        <div className="w-3 h-3 rounded-full bg-green-500/80" />
        <span className="ml-2 text-xs text-[#8888A0] font-mono">{title}</span>
      </div>
      <div className="p-6 font-mono text-sm leading-relaxed text-[#E8E8ED] overflow-x-auto">
        {children}
      </div>
    </motion.div>
  );
}

function SectionNumber({ n }: { n: string }) {
  return (
    <motion.div variants={fadeUp} className="flex flex-col items-center mb-12">
      <span className="inline-block px-3 py-1 border border-[#8B5CF6]/30 rounded-full text-xs font-mono text-[#8B5CF6] mb-4">
        {n}
      </span>
      <div className="w-px h-12 bg-gradient-to-b from-[#8B5CF6]/40 to-transparent" />
    </motion.div>
  );
}

export default function Home() {
  const heroRef = useRef(null);
  const { scrollYProgress } = useScroll({ target: heroRef, offset: ["start start", "end start"] });
  const heroOpacity = useTransform(scrollYProgress, [0, 0.5], [1, 0]);
  const heroY = useTransform(scrollYProgress, [0, 0.5], [0, -100]);

  return (
    <div className="grid-bg min-h-screen">
      {/* ═══ NAV ═══ */}
      <nav className="fixed top-0 left-0 right-0 z-50 backdrop-blur-md bg-[#0A0A0F]/80 border-b border-white/5">
        <div className="max-w-6xl mx-auto px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-[#8B5CF6] animate-pulse" />
            <span className="font-semibold text-sm tracking-wide">SAMHATI</span>
          </div>
          <div className="hidden md:flex items-center gap-8 text-xs text-[#8888A0]">
            <a href="#how" className="hover:text-white transition-colors">How it works</a>
            <a href="#swarm" className="hover:text-white transition-colors">Swarm</a>
            <a href="#toploc" className="hover:text-white transition-colors">TOPLOC</a>
            <a href="#solana" className="hover:text-white transition-colors">Solana</a>
            <a href="#models" className="hover:text-white transition-colors">Models</a>
          </div>
          <a
            href="https://github.com/mrunalpendem123/Samhati"
            target="_blank"
            rel="noopener noreferrer"
            className="px-4 py-1.5 text-xs font-medium border border-[#8B5CF6]/40 rounded-full text-[#8B5CF6] hover:bg-[#8B5CF6]/10 transition-colors"
          >
            GitHub
          </a>
        </div>
      </nav>

      {/* ═══ HERO ═══ */}
      <motion.div ref={heroRef} style={{ opacity: heroOpacity, y: heroY }} className="relative min-h-screen flex flex-col items-center justify-center pt-14">
        {/* Ambient glow */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="w-[600px] h-[600px] rounded-full bg-[#8B5CF6]/5 blur-[120px]" />
        </div>

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.2 }}
          className="relative z-10 text-center px-6"
        >
          <p className="text-xs font-mono text-[#8B5CF6] tracking-[0.3em] uppercase mb-8">
            Decentralized AI Inference
          </p>

          <h1 className="text-5xl md:text-7xl lg:text-8xl font-bold tracking-tight leading-[0.9] mb-8">
            <span className="block text-white">Free AI for</span>
            <span className="block bg-gradient-to-r from-[#8B5CF6] to-[#A78BFA] bg-clip-text text-transparent">
              Everyone, Forever.
            </span>
          </h1>

          <p className="max-w-xl mx-auto text-lg md:text-xl text-[#8888A0] leading-relaxed mb-12">
            Every user is a node. Every device contributes inference.
            Like BitTorrent for AI — nobody pays because everyone shares.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <a
              href="https://github.com/mrunalpendem123/Samhati"
              className="px-8 py-3 bg-[#8B5CF6] text-white font-medium rounded-full hover:bg-[#7C3AED] transition-colors text-sm"
            >
              Get Started
            </a>
            <a
              href="#how"
              className="px-8 py-3 border border-white/10 text-white/70 font-medium rounded-full hover:border-[#8B5CF6]/40 hover:text-white transition-colors text-sm"
            >
              Read the Paper
            </a>
          </div>
        </motion.div>

        {/* Install command */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.8 }}
          className="relative z-10 mt-16 w-full max-w-2xl px-6"
        >
          <div className="bg-[#111118] border border-white/5 rounded-xl px-6 py-4 font-mono text-sm flex items-center justify-between">
            <code className="text-[#8888A0]">
              <span className="text-[#8B5CF6]">$</span>{" "}
              curl -sSL https://raw.githubusercontent.com/mrunalpendem123/Samhati/main/install.sh | bash
            </code>
          </div>
        </motion.div>

        {/* Scroll indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.5 }}
          className="absolute bottom-8 flex flex-col items-center gap-2"
        >
          <span className="text-[10px] text-[#8888A0] tracking-widest uppercase">Scroll</span>
          <motion.div
            animate={{ y: [0, 8, 0] }}
            transition={{ duration: 1.5, repeat: Infinity }}
            className="w-px h-8 bg-gradient-to-b from-[#8B5CF6]/50 to-transparent"
          />
        </motion.div>
      </motion.div>

      <div className="hr-purple" />

      {/* ═══ TERMINAL SHOWCASE ═══ */}
      <Section>
        <SectionNumber n="00" />
        <motion.h2 variants={fadeUp} className="text-3xl md:text-5xl font-bold text-center mb-6">
          A TUI That Runs a<br />
          <span className="text-[#8B5CF6]">Decentralized Network</span>
        </motion.h2>
        <motion.p variants={fadeUp} className="text-center text-[#8888A0] max-w-2xl mx-auto mb-16 text-lg">
          Five tabs. One identity. Chat, monitor, earn — all from your terminal.
          The same UI renders in your browser via WebAssembly.
        </motion.p>

        <Terminal title="samhati — Chat Tab">
          <div className="text-[#8B5CF6] mb-2">┌─ Samhati ──────────────────────────────────────────────┐</div>
          <div className="text-white mb-1">│ <span className="text-[#8B5CF6] font-bold underline">1 Chat</span>  2 Dashboard  3 Models  4 Wallet  5 Settings │</div>
          <div className="text-[#8B5CF6] mb-3">├────────────────────────────────────────────────────────┤</div>
          <div className="text-[#22C55E] mb-1">│ user (14:23)                                           │</div>
          <div className="text-white mb-3">│ Explain how TOPLOC proofs prevent cheating              │</div>
          <div className="text-[#A78BFA] mb-1">│ assistant (14:23)                                      │</div>
          <div className="text-white mb-1">│ TOPLOC captures BLAKE3 hashes of intermediate layer     │</div>
          <div className="text-white mb-1">│ activations during inference. Different model weights    │</div>
          <div className="text-white mb-1">│ produce different activations → different hashes. A     │</div>
          <div className="text-white mb-3">│ node that fakes inference can&apos;t produce matching proofs. │</div>
          <div className="text-[#8888A0] mb-1">│ [Hard | 5 nodes + debate | General | 92% conf | 3.2s]  │</div>
          <div className="text-[#8B5CF6] mb-2">├────────────────────────────────────────────────────────┤</div>
          <div className="text-white mb-1">│ {">"} _                                                    │</div>
          <div className="text-[#8B5CF6] mb-2">├────────────────────────────────────────────────────────┤</div>
          <div className="text-[#8888A0]">│ ● ELO:1847 │ Peers:12 │ 2.1SOL │ 5EZ...8Qk │ Qwen-7B │</div>
          <div className="text-[#8B5CF6]">└────────────────────────────────────────────────────────┘</div>
        </Terminal>
      </Section>

      <div className="hr-purple" />

      {/* ═══ THE BITTORENT ANALOGY ═══ */}
      <Section id="how">
        <SectionNumber n="01" />
        <motion.h2 variants={fadeUp} className="text-4xl md:text-6xl font-bold text-center mb-8">
          BitTorrent for AI
        </motion.h2>
        <motion.p variants={fadeUp} className="text-center text-[#8888A0] text-xl max-w-2xl mx-auto mb-16 leading-relaxed">
          In BitTorrent, you download and upload simultaneously. Nobody pays for bandwidth
          because everyone contributes it. Samhati does the same for intelligence.
        </motion.p>

        <motion.div variants={fadeUp} className="grid md:grid-cols-2 gap-8 mb-16">
          <div className="bg-[#111118] border border-white/5 rounded-2xl p-8">
            <div className="text-xs font-mono text-[#8888A0] mb-4 tracking-widest">BITTORRENT</div>
            <div className="font-mono text-sm space-y-2 text-[#8888A0]">
              <div><span className="text-white">download</span> (consume) + <span className="text-white">upload</span> (seed)</div>
              <div>→ nobody pays for bandwidth</div>
            </div>
          </div>
          <div className="bg-[#111118] border border-[#8B5CF6]/20 rounded-2xl p-8 glow-purple">
            <div className="text-xs font-mono text-[#8B5CF6] mb-4 tracking-widest">SAMHATI</div>
            <div className="font-mono text-sm space-y-2 text-[#8888A0]">
              <div><span className="text-white">query AI</span> (use) + <span className="text-white">serve others</span> (node)</div>
              <div>→ nobody pays for intelligence</div>
              <div className="text-[#8B5CF6]">+ SMTI tokens reward contributors</div>
            </div>
          </div>
        </motion.div>

        <motion.p variants={fadeUp} className="text-center text-[#8888A0] max-w-3xl mx-auto text-lg leading-relaxed">
          The core mechanism is <span className="text-white font-semibold">swarm intelligence</span>: queries
          fan out to N nodes via <span className="text-[#8B5CF6]">iroh QUIC</span>. Each runs
          llama.cpp with a <span className="text-[#8B5CF6]">TOPLOC</span> cryptographic proof.
          Nodes peer-rank answers. <span className="text-[#8B5CF6]">BradleyTerry</span> picks
          the winner — achieving <span className="text-white font-semibold">85.90% on GPQA Diamond</span>{" "}
          vs 68.69% for majority voting.
        </motion.p>
      </Section>

      <div className="hr-purple" />

      {/* ═══ SWARM FLOW ═══ */}
      <Section id="swarm">
        <SectionNumber n="02" />
        <motion.h2 variants={fadeUp} className="text-4xl md:text-6xl font-bold text-center mb-6">
          Swarm Consensus
        </motion.h2>
        <motion.p variants={fadeUp} className="text-center text-[#8888A0] text-lg max-w-2xl mx-auto mb-16">
          Your question goes through 5 phases. The answer you see is the statistically
          verified winner across multiple independent nodes.
        </motion.p>

        <div className="space-y-6">
          {[
            { phase: "Classify", desc: "Complexity classifier routes your query: Easy (3 nodes), Medium (3), Hard (5 + debate). Domain classifier tags Code, Math, Reasoning, or General.", color: "#22C55E" },
            { phase: "Fan-Out", desc: "Your prompt dispatches to N nodes in parallel over iroh QUIC. Each node runs llama.cpp inference independently and generates a TOPLOC proof.", color: "#8B5CF6" },
            { phase: "Debate", desc: "Hard queries only. Each node sees all other answers and rewrites its own. Multi-agent debate (arXiv:2305.14325) improves accuracy by +12.8pp on math.", color: "#F59E0B" },
            { phase: "Peer Rank", desc: "Every node judges every other node's answer in pairwise comparisons. 50-100 token reasoning chains per comparison. Nodes can't judge their own answer.", color: "#EC4899" },
            { phase: "Aggregate", desc: "BradleyTerry MLE converts pairwise preferences into a global ranking. The MM algorithm runs up to 100 iterations with 1e-6 tolerance. Winner + confidence returned.", color: "#06B6D4" },
          ].map((step, i) => (
            <motion.div
              key={step.phase}
              variants={fadeUp}
              className="flex gap-6 items-start bg-[#111118] border border-white/5 rounded-2xl p-6 md:p-8 hover:border-white/10 transition-colors"
            >
              <div className="flex-shrink-0 w-12 h-12 rounded-xl flex items-center justify-center font-mono text-sm font-bold" style={{ background: `${step.color}15`, color: step.color }}>
                {i + 1}
              </div>
              <div>
                <h3 className="text-xl font-semibold text-white mb-2">{step.phase}</h3>
                <p className="text-[#8888A0] leading-relaxed">{step.desc}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </Section>

      <div className="hr-purple" />

      {/* ═══ TOPLOC ═══ */}
      <Section id="toploc">
        <SectionNumber n="03" />
        <motion.h2 variants={fadeUp} className="text-4xl md:text-6xl font-bold text-center mb-6">
          TOPLOC Proofs
        </motion.h2>
        <motion.p variants={fadeUp} className="text-center text-[#8888A0] text-lg max-w-2xl mx-auto mb-16">
          Proof of honest inference. A node that fakes computation can&apos;t produce matching
          activation hashes — caught 100% of the time.
        </motion.p>

        <Terminal title="TOPLOC Proof Generation">
          <div className="text-[#8888A0]">{`// During inference, each layer's activations are hashed`}</div>
          <div className="mt-2">
            <span className="text-[#22C55E]">Layer  0</span>: attention + MLP → hidden_state → <span className="text-[#8B5CF6]">BLAKE3</span> hash
          </div>
          <div>
            <span className="text-[#22C55E]">Layer  1</span>: attention + MLP → hidden_state → <span className="text-[#8B5CF6]">BLAKE3</span> hash
          </div>
          <div className="text-[#8888A0]">  ...</div>
          <div>
            <span className="text-[#22C55E]">Layer 31</span>: attention + MLP → hidden_state → <span className="text-[#8B5CF6]">BLAKE3</span> hash
          </div>
          <div className="mt-3 text-[#F59E0B]">→ Chain 32 hashes → final proof hash</div>
          <div className="text-[#F59E0B]">→ <span className="text-[#8B5CF6]">Ed25519</span> sign (bound to node_pubkey)</div>
          <div className="mt-3 text-[#8888A0]">{`// Verifier checks: model hash, token count, chunk count,`}</div>
          <div className="text-[#8888A0]">{`// freshness (5 min), node binding, Ed25519 signature`}</div>
        </Terminal>

        <motion.div variants={fadeUp} className="grid md:grid-cols-3 gap-6 mt-12">
          {[
            { label: "Model Binding", desc: "BLAKE3(model_id) must match. Different weights = different proof." },
            { label: "Node Binding", desc: "Proof contains the prover's Ed25519 pubkey. Can't steal another node's proof." },
            { label: "Freshness", desc: "5-minute maximum age. Prevents replay of old proofs for new queries." },
          ].map((item) => (
            <div key={item.label} className="bg-[#111118] border border-white/5 rounded-xl p-6">
              <h4 className="text-[#8B5CF6] font-semibold mb-2 text-sm">{item.label}</h4>
              <p className="text-[#8888A0] text-sm leading-relaxed">{item.desc}</p>
            </div>
          ))}
        </motion.div>
      </Section>

      <div className="hr-purple" />

      {/* ═══ UNIFIED IDENTITY ═══ */}
      <Section>
        <SectionNumber n="04" />
        <motion.h2 variants={fadeUp} className="text-4xl md:text-6xl font-bold text-center mb-6">
          One Key for Everything
        </motion.h2>
        <motion.p variants={fadeUp} className="text-center text-[#8888A0] text-lg max-w-2xl mx-auto mb-16">
          A single Ed25519 keypair stored at <code>~/.samhati/identity.json</code> derives
          your Solana wallet, P2P identity, and proof signer.
        </motion.p>

        <motion.div variants={fadeUp} className="grid md:grid-cols-3 gap-6">
          {[
            { use: "Solana Wallet", format: "base58(pubkey)", example: "5EZ7Q..." },
            { use: "iroh P2P NodeId", format: "hex(pubkey)", example: "a3b4c5d6..." },
            { use: "TOPLOC Signer", format: "Ed25519 sign()", example: "Bound to proof" },
          ].map((id) => (
            <div key={id.use} className="bg-[#111118] border border-white/5 rounded-xl p-6 text-center">
              <div className="text-[#8B5CF6] font-semibold mb-2">{id.use}</div>
              <div className="font-mono text-xs text-[#8888A0] mb-1">{id.format}</div>
              <div className="font-mono text-xs text-white/40">{id.example}</div>
            </div>
          ))}
        </motion.div>
      </Section>

      <div className="hr-purple" />

      {/* ═══ SOLANA ═══ */}
      <Section id="solana">
        <SectionNumber n="05" />
        <motion.h2 variants={fadeUp} className="text-4xl md:text-6xl font-bold text-center mb-6">
          Settled on <span className="text-[#8B5CF6]">Solana</span>
        </motion.h2>
        <motion.p variants={fadeUp} className="text-center text-[#8888A0] text-lg max-w-2xl mx-auto mb-16">
          Node registry, ELO scores, round results, and domain demand — all on-chain.
          Anchor program with 5 instructions and 3 PDAs.
        </motion.p>

        <motion.div variants={fadeUp} className="space-y-4">
          {[
            { name: "ProtocolConfig", seeds: '[b"config"]', desc: "Authority, emission rate, domain demand counters (Code, Math, Reasoning, General)" },
            { name: "NodeAccount", seeds: '[b"node", pubkey]', desc: "Operator, ELO (starts 1500), model name, rounds played, wins, pending rewards" },
            { name: "RoundAccount", seeds: '[b"round", round_id]', desc: "Participants, TOPLOC proof hashes, ELO deltas, winner, domain, SMTI emitted" },
          ].map((pda) => (
            <div key={pda.name} className="bg-[#111118] border border-white/5 rounded-xl p-6 flex flex-col md:flex-row md:items-center gap-4">
              <div className="flex-shrink-0">
                <div className="text-white font-semibold">{pda.name}</div>
                <div className="font-mono text-xs text-[#8B5CF6]">{pda.seeds}</div>
              </div>
              <div className="text-[#8888A0] text-sm leading-relaxed">{pda.desc}</div>
            </div>
          ))}
        </motion.div>

        <motion.div variants={fadeUp} className="mt-8 text-center">
          <code className="text-xs text-[#8888A0]">
            Program ID: AB7cSMLv1J7J28DKLMbzo2tyNp1kZSmE67a6Heoa5Mkr (devnet)
          </code>
        </motion.div>
      </Section>

      <div className="hr-purple" />

      {/* ═══ MODELS ═══ */}
      <Section id="models">
        <SectionNumber n="06" />
        <motion.h2 variants={fadeUp} className="text-4xl md:text-6xl font-bold text-center mb-6">
          21 Models, 4 Domains
        </motion.h2>
        <motion.p variants={fadeUp} className="text-center text-[#8888A0] text-lg max-w-2xl mx-auto mb-16">
          From 0.5B to 14B parameters. Specialists earn <span className="text-[#8B5CF6] font-semibold">1.5x SMTI</span> on
          matched queries. RAM auto-detection recommends what fits your device.
        </motion.p>

        <Terminal title="samhati — Models Tab">
          <div className="text-[#8888A0] text-xs mb-3">{'  Model                    Params  Domain     Size   RAM    Bonus  Status'}</div>
          <div className="text-[#8888A0] text-xs mb-2">{'  ─────────────────────────────────────────────────────────────────────'}</div>
          <div className="text-white text-xs"><span className="text-[#22C55E]">▸</span> Qwen2.5-7B              7B      General    4.4G   8.8G   1.0x   <span className="text-[#22C55E]">● Active</span></div>
          <div className="text-[#8888A0] text-xs">  Qwen2.5-Coder-7B        7B      Code       4.4G   8.8G   <span className="text-[#F59E0B]">1.5x</span>   Installed</div>
          <div className="text-[#8888A0] text-xs">  DeepSeek-Coder-V2-Lite  7B      Code       4.4G   8.8G   <span className="text-[#F59E0B]">1.5x</span>   —</div>
          <div className="text-[#8888A0] text-xs">  Qwen2.5-Math-7B         7B      Math       4.4G   8.8G   <span className="text-[#F59E0B]">1.5x</span>   —</div>
          <div className="text-[#8888A0] text-xs">  Llama-3.1-8B            8B      General    4.7G   9.4G   1.0x   —</div>
          <div className="text-[#8888A0] text-xs">  Qwen2.5-14B             14B     General    8.9G   17.8G  1.0x   —</div>
          <div className="text-[#8888A0] text-xs">  DeepSeek-R1-Distill-14B 14B     Reasoning  8.9G   17.8G  <span className="text-[#F59E0B]">1.5x</span>   —</div>
          <div className="mt-3 text-xs text-[#8888A0]">  Enter: Install + Activate │ s: Add to Swarm │ r: Connect Peer</div>
        </Terminal>

        <motion.div variants={fadeUp} className="mt-12">
          <Terminal title="samhati — Dashboard Tab">
            <div className="text-white mb-2">  <span className="text-[#8B5CF6] font-bold">ELO: 1847</span>  ████████████████████████████████████████ ▲</div>
            <div className="text-[#8888A0] mb-4">  Rounds: 142  │  Wins: 89  │  Win Rate: 62.7%</div>
            <div className="text-white mb-2">  Network Demand:</div>
            <div className="text-xs">  <span className="text-[#22C55E]">Code</span>      <span className="text-[#8B5CF6]">████████████████████</span> 42%  ← run Coder for 1.5x SMTI</div>
            <div className="text-xs">  <span className="text-[#22C55E]">Math</span>      <span className="text-[#8B5CF6]">████████</span>             16%</div>
            <div className="text-xs">  <span className="text-[#22C55E]">Reasoning</span> <span className="text-[#8B5CF6]">████</span>                 11%</div>
            <div className="text-xs">  <span className="text-[#22C55E]">General</span>   <span className="text-[#8B5CF6]">████████████</span>         31%</div>
          </Terminal>
        </motion.div>
      </Section>

      <div className="hr-purple" />

      {/* ═══ ARCHITECTURE ═══ */}
      <Section>
        <SectionNumber n="07" />
        <motion.h2 variants={fadeUp} className="text-4xl md:text-6xl font-bold text-center mb-6">
          The Stack
        </motion.h2>
        <motion.p variants={fadeUp} className="text-center text-[#8888A0] text-lg max-w-2xl mx-auto mb-16">
          13 Rust crates. One Solana program. Patched llama.cpp.
          SDKs in Rust, Python, and TypeScript.
        </motion.p>

        <motion.div variants={fadeUp} className="grid md:grid-cols-2 gap-4">
          {[
            { name: "samhati-tui", desc: "Terminal + browser UI (ratatui + ratzilla WASM)" },
            { name: "mesh-node", desc: "HTTP API (axum) + iroh QUIC + gossip protocol" },
            { name: "inference-coordinator", desc: "Distributed shard execution + KV cache + failover replay" },
            { name: "samhati-toploc", desc: "TOPLOC proof generation, verification, calibration" },
            { name: "samhati-swarm", desc: "Swarm consensus: fan-out, debate, classification" },
            { name: "samhati-ranking", desc: "ELO engine + BradleyTerry MLE + SMTI rewards" },
            { name: "proximity-router", desc: "Peer selection by latency, VRAM, layers (zero deps)" },
            { name: "shard-store", desc: "Content-addressed weight cache (BLAKE3)" },
            { name: "llama-toploc", desc: "Patched llama.cpp with activation-level TOPLOC proofs (C++)" },
            { name: "samhati-protocol", desc: "Solana Anchor program: 5 instructions, 3 PDAs" },
          ].map((crate) => (
            <div key={crate.name} className="bg-[#111118] border border-white/5 rounded-xl p-5 hover:border-[#8B5CF6]/20 transition-colors">
              <div className="font-mono text-sm text-[#8B5CF6] mb-1">{crate.name}</div>
              <div className="text-[#8888A0] text-sm">{crate.desc}</div>
            </div>
          ))}
        </motion.div>
      </Section>

      <div className="hr-purple" />

      {/* ═══ RESEARCH ═══ */}
      <Section>
        <SectionNumber n="08" />
        <motion.h2 variants={fadeUp} className="text-4xl md:text-6xl font-bold text-center mb-16">
          Research Foundation
        </motion.h2>

        <motion.div variants={fadeUp} className="space-y-4">
          {[
            { paper: "Fortytwo (arXiv:2510.24801)", use: "LLM peer-ranking + BradleyTerry aggregation" },
            { paper: "Multi-Agent Debate (arXiv:2305.14325)", use: "Debate round for hard queries (+12.8pp on math)" },
            { paper: "TOPLOC (arXiv:2501.16007)", use: "Activation-level proofs in patched llama.cpp" },
            { paper: "Agent Diversity (arXiv:2602.03794)", use: "Diverse models beat copies (2 diverse ≥ 16 same)" },
            { paper: "RouteLLM (arXiv:2406.18665)", use: "Complexity classifier (Easy / Medium / Hard)" },
            { paper: "EAGLE-3 (arXiv:2503.01840)", use: "Future: 4-6x per-node speedup via speculative decoding" },
          ].map((ref) => (
            <div key={ref.paper} className="flex flex-col md:flex-row gap-4 bg-[#111118] border border-white/5 rounded-xl p-5">
              <div className="text-white font-medium text-sm flex-shrink-0 md:w-72">{ref.paper}</div>
              <div className="text-[#8888A0] text-sm">{ref.use}</div>
            </div>
          ))}
        </motion.div>
      </Section>

      <div className="hr-purple" />

      {/* ═══ CTA ═══ */}
      <Section className="text-center">
        <motion.h2 variants={fadeUp} className="text-4xl md:text-6xl font-bold mb-6">
          Start Contributing<br />
          <span className="text-[#8B5CF6]">Intelligence</span>
        </motion.h2>
        <motion.p variants={fadeUp} className="text-[#8888A0] text-lg max-w-xl mx-auto mb-12">
          One command. Any device. Your laptop becomes part of the world&apos;s
          free AI network. Earn SMTI while your machine runs inference.
        </motion.p>
        <motion.div variants={fadeUp} className="flex flex-col sm:flex-row gap-4 justify-center">
          <a
            href="https://github.com/mrunalpendem123/Samhati"
            className="px-8 py-3 bg-[#8B5CF6] text-white font-medium rounded-full hover:bg-[#7C3AED] transition-colors text-sm"
          >
            View on GitHub
          </a>
          <a
            href="https://github.com/mrunalpendem123/Samhati/blob/main/README.md"
            className="px-8 py-3 border border-white/10 text-white/70 font-medium rounded-full hover:border-[#8B5CF6]/40 hover:text-white transition-colors text-sm"
          >
            Read the Docs
          </a>
        </motion.div>
      </Section>

      {/* ═══ FOOTER ═══ */}
      <footer className="border-t border-white/5 py-12 px-6">
        <div className="max-w-5xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-[#8B5CF6]" />
            <span className="font-semibold text-sm">SAMHATI</span>
            <span className="text-[#8888A0] text-xs ml-2">Free AI for Everyone, Forever</span>
          </div>
          <div className="flex items-center gap-6 text-xs text-[#8888A0]">
            <span>Built on Solana</span>
            <span>Powered by TOPLOC</span>
            <span>MIT License</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
