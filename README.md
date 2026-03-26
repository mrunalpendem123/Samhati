# Samhati — Free AI for Everyone, Forever

**Samhati** is a decentralized AI inference network that makes frontier-level artificial
intelligence permanently free for every person on earth. Every user is simultaneously a node
operator — installing Samhati turns any device into both an AI consumer and an inference
contributor, exactly like BitTorrent where downloading and uploading happen simultaneously.

```
BitTorrent:  download (consume)  +  upload (seed)    →  nobody pays
Samhati:     query AI (use)      +  serve others (node) →  nobody pays
             SMTI tokens reward node operators from protocol emission
```

The core mechanism is **swarm intelligence**: queries fan out to N independent nodes via
**iroh QUIC**. Each node runs **llama.cpp** inference and attaches a **TOPLOC** cryptographic
proof. The same N nodes peer-rank each other's answers. **BradleyTerry** aggregation selects the
winner — achieving **85.90% on GPQA Diamond** vs 68.69% for majority voting.

> Built on **Solana** · Powered by **TOPLOC**

---

## Quick Start

### Terminal (native)

```bash
curl -sSL https://raw.githubusercontent.com/mrunalpendem123/Samhati/main/install.sh | bash
cd ~/Samhati && cargo run -p samhati-tui --bin samhati
```

### Browser (WASM) — experimental

```bash
cd crates/samhati-tui
rustup target add wasm32-unknown-unknown
cargo install trunk
trunk serve
# Opens at http://localhost:8080
```

The browser build uses **ratzilla** (DomBackend) to render the same ratatui UI in your browser.
Chat works via remote API — set the endpoint in the Settings tab.

### Inside the TUI

| Key | Action |
|-----|--------|
| `Tab` / `1-5` | Switch tabs |
| `Enter` | Send message (Chat) / Install + activate model (Models) |
| `s` | Add model as swarm node (Models) |
| `r` | Connect to a friend by NodeId (Models) |
| `a` | Request SOL airdrop (Wallet) |
| `q` | Quit |

### Tabs

- **Chat** — AI chat with swarm consensus. Shows difficulty, domain, confidence, node count
- **Dashboard** — ELO score, network demand bars, swarm nodes, last round stats
- **Models** — 21 models (Code, Math, Reasoning specialists), auto-detect RAM, download from HuggingFace
- **Wallet** — Solana devnet wallet: balance, transactions, airdrop
- **Settings** — API endpoint, VRAM config

---

## How It Works

```
You run the TUI (terminal or browser)
  ↓
Identity loaded (~/.samhati/identity.json — one Ed25519 key for everything)
  ↓
Reads Solana → finds all registered nodes → bootstraps iroh gossip
  ↓
Auto-registers on Solana if new (auto-airdrop if needed)
  ↓
You pick a model → llama-server starts (TOPLOC-enabled)
  ↓
Announces to gossip → other nodes discover you automatically
  ↓
You ask a question
  ↓
Complexity classifier → Easy (3 nodes) / Medium (3 nodes) / Hard (5 nodes + debate)
  ↓
Domain classifier → Code / Math / Reasoning / General
  ↓
Phase 1: Fan out to N nodes in parallel
  Each runs llama-server → answer + TOPLOC proof hash
  ↓
Phase 1.5 (hard queries only): Debate round
  Each node sees others' answers, rewrites its own (arXiv:2305.14325)
  ↓
Phase 2: LLM peer-ranking
  Each node judges others' answers (excluded from judging own)
  50-100 token reasoning chains per comparison
  ↓
Phase 3: BradleyTerry aggregation
  Statistical winner from real pairwise preferences
  ↓
Phase 4: ELO update + persist to disk
  ↓
Phase 5: Solana settlement (async)
  submit_round → proof hashes + ELO deltas + domain on-chain
  ↓
Best answer shown to user
  [Hard | 5 nodes + debate | Code | 85% conf | 4200ms]
```

---

## Core Architecture

```
                    ┌─────────────────────────┐
                    │  Shared Rendering Code   │
                    │  ui.rs, tabs/*, app.rs   │
                    │  (pure ratatui widgets)  │
                    └─────────┬───────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
    ┌─────────▼──────────┐         ┌──────────▼──────────┐
    │    main.rs          │         │   web_main.rs       │
    │    (native)         │         │   (browser/WASM)    │
    │                     │         │                     │
    │  crossterm backend  │         │  ratzilla DomBackend│
    │  tokio runtime      │         │  Rc<RefCell> state  │
    │  filesystem I/O     │         │  browser fetch API  │
    │  P2P networking     │         │  (remote node only) │
    │  local llama exec   │         │                     │
    └─────────────────────┘         └─────────────────────┘

    cargo run -p samhati-tui        trunk serve
    → terminal binary               → browser WASM app
```

| Component | Purpose |
|-----------|---------|
| `samhati-tui` | Terminal + browser UI — chat, models, wallet, dashboard, settings |
| `mesh-node` | Network node — HTTP API (axum), iroh QUIC, gossip, rate limiting |
| `inference-coordinator` | Distributed inference — RPC protocol, shard execution, KV cache |
| `samhati-toploc` | TOPLOC proof system — prover, verifier, calibration |
| `samhati-swarm` | Swarm consensus — fan-out, classification, coordination |
| `samhati-ranking` | ELO engine + BradleyTerry aggregation |
| `proximity-router` | Peer selection — latency, VRAM, layer ranges (std-only, zero deps) |
| `cluster-manager` | Cluster constraints and node selection |
| `model-config` | Model config parsing (params, quantization, architecture) |
| `shard-store` | Content-addressed weight cache (BLAKE3) |
| `llama-toploc/` | Patched llama.cpp with activation-level TOPLOC proofs |
| `programs/samhati-protocol/` | Solana Anchor program (5 instructions, 3 PDAs) |
| `sdk/` | Client SDKs — Rust, Python, TypeScript |
| `app-tauri/` | Tauri v2 desktop app |

### Unified Identity

One Ed25519 keypair (`~/.samhati/identity.json`) for everything:

| Use | Format |
|-----|--------|
| Solana wallet | base58(pubkey) |
| iroh P2P NodeId | hex(pubkey) |
| TOPLOC proof signer | Ed25519 signature (bound to node_pubkey) |
| On-chain PDA | seeds=[b"node", pubkey] |

### Security Hardening

| Feature | Implementation |
|---------|---------------|
| API auth | Constant-time comparison (`subtle::ConstantTimeEq`) |
| Rate limiting | Per-client by TCP socket address (not spoofable headers) |
| TOPLOC verification | Node-to-key binding, 5-min freshness, reject when no keys registered |
| RPC deserialization | 256 MB bincode size limit on all message types |
| Gossip parsing | 64 KB max message size before JSON deserialization |
| Identity file | 0600 permissions (Unix), warning on Windows |
| P2P encryption | QUIC/TLS 1.3 via iroh (all traffic encrypted) |
| Solana RPC | Configurable via `SOLANA_RPC_URL` env var (defaults to devnet) |
| Supply chain | Cargo.lock tracked in git for reproducible builds |

### Auto-Mesh via Solana

Nodes discover each other automatically — no manual IP entry:

1. On startup → read all NodeAccount PDAs from Solana (free RPC call)
2. Each operator pubkey = iroh NodeId (same key)
3. Bootstrap iroh gossip with all NodeIds
4. iroh connects via QUIC (relay handles NAT)
5. Gossip propagates announcements — model name, inference URL

### TOPLOC — Proof of Honest Inference

Patched llama.cpp captures intermediate layer activations during inference:

```
Layer 0: attention + MLP → hidden_state → BLAKE3 hash
Layer 1: attention + MLP → hidden_state → BLAKE3 hash
...
Layer 31: attention + MLP → hidden_state → BLAKE3 hash
→ Chain all 32 hashes → final proof hash → Ed25519 sign (bound to node_pubkey)
→ Verifier checks: model hash, token count, chunk count, freshness, node binding, signature
```

Build: `./llama-toploc/build.sh`

### Adaptive Swarm

| Query type | Nodes | Method | Example |
|------------|-------|--------|---------|
| Easy | 3 | Peer rank | "What is photosynthesis?" |
| Medium | 3 | Peer rank | "Write a Python web scraper" |
| Hard | 5 | Debate + peer rank | "Prove the Pythagorean theorem step by step" |

### Domain Demand (Solana)

ProtocolConfig on-chain tracks domain counters per round. Dashboard shows live demand:

```
Code      ████████████████████ 42%  ← run Coder model for 1.5x SMTI
Math      ████████             16%
Reasoning ████                 11%
General   ████████████         31%
```

New operators see what the network needs → pick specialist models → earn more.

---

## Solana Integration

| On-chain | What |
|----------|------|
| `ProtocolConfig` PDA | Authority, emission rate, domain demand counters |
| `NodeAccount` PDA | Operator pubkey, ELO, model name, rounds, wins, rewards |
| `RoundAccount` PDA | Participants, proof hashes, ELO deltas, winner, domain |
| 5 instructions | register_node, calibrate_node, submit_round, slash_node, emit_rewards |

Program ID: `AB7cSMLv1J7J28DKLMbzo2tyNp1kZSmE67a6Heoa5Mkr` (devnet)

---

## Models

21 real models, all downloadable from HuggingFace (Q4_K_M GGUF):

| Size | General | Code | Math | Reasoning |
|------|---------|------|------|-----------|
| 0.5B | Qwen2.5-0.5B | Qwen2.5-Coder-0.5B | | |
| 1B | Gemma-3-1B | | | |
| 1.5B | Qwen2.5-1.5B | Qwen2.5-Coder-1.5B | Qwen2.5-Math-1.5B | |
| 1.7B | SmolLM2-1.7B | | | |
| 3B | Qwen2.5-3B, Llama-3.2-3B | Qwen2.5-Coder-3B | | |
| 3.8B | | | | Phi-4-mini |
| 4B | Gemma-3-4B | | | |
| 7B | Qwen2.5-7B, Mistral-7B | Qwen2.5-Coder-7B, DeepSeek-Coder-V2-Lite | Qwen2.5-Math-7B | |
| 8B | Llama-3.1-8B | | | |
| 14B | Qwen2.5-14B | Qwen2.5-Coder-14B | | DeepSeek-R1-Distill-14B |

Domain specialists earn **1.5x SMTI** on matched queries.

---

## Project Structure

```
Samhati/
├── crates/
│   ├── samhati-tui/                # Terminal + browser UI
│   │   └── src/
│   │       ├── main.rs             # Native TUI (crossterm backend, tokio)
│   │       ├── web_main.rs         # Browser WASM (ratzilla DomBackend)
│   │       ├── app.rs              # App state, 5 tabs, 21 models
│   │       ├── ui.rs               # Shared rendering (pure ratatui)
│   │       ├── swarm.rs            # Swarm: classify → fan-out → debate → rank → BT → ELO
│   │       ├── identity.rs         # Unified Ed25519 identity
│   │       ├── network.rs          # iroh P2P: gossip, peer discovery
│   │       ├── registry.rs         # Solana: fetch nodes, register, submit_round
│   │       ├── settlement.rs       # Round payloads, Solana settlement
│   │       ├── node_runner.rs      # Spawns TOPLOC llama-server
│   │       ├── model_download.rs   # HuggingFace GGUF downloads
│   │       ├── wallet.rs           # Solana devnet wallet
│   │       ├── api.rs              # HTTP client for inference API
│   │       ├── events.rs           # Keyboard event handling
│   │       └── tabs/               # Chat, Dashboard, Models, Wallet, Settings
│   │   └── index.html              # Trunk build config for WASM
│   ├── mesh-node/                  # Network node (axum HTTP, iroh QUIC, gossip)
│   ├── inference-coordinator/      # Distributed inference (RPC, shards, KV cache)
│   ├── samhati-toploc/             # TOPLOC proofs (prover, verifier, calibration)
│   ├── samhati-swarm/              # Swarm consensus traits
│   ├── samhati-ranking/            # ELO + BradleyTerry engine
│   ├── proximity-router/           # Peer selection (std-only, zero deps)
│   ├── cluster-manager/            # Cluster constraints
│   ├── model-config/               # Model config parsing
│   └── shard-store/                # Content-addressed weight cache (BLAKE3)
├── llama-toploc/                   # Patched llama.cpp with TOPLOC proofs
│   ├── toploc-proof.h              # BLAKE3 activation hashing (~140 lines C++)
│   ├── toploc.patch                # Diff to apply on llama.cpp
│   └── build.sh                    # One command: clone → patch → build
├── programs/
│   └── samhati-protocol/           # Solana Anchor program (5 instructions, 3 PDAs)
├── sdk/
│   ├── rust/                       # Rust SDK (reqwest, async streaming)
│   ├── python/                     # Python SDK (httpx, sync + async)
│   └── typescript/                 # TypeScript SDK (native fetch)
├── app-tauri/                      # Tauri v2 desktop app
├── pipeline/                       # Self-evolving training (Python)
├── scripts/
│   └── init_protocol.py            # Initialize ProtocolConfig on Solana
├── install.sh                      # One-command install for new users
└── README.md
```

---

## Research Foundation

| Domain | Paper | How we use it |
|--------|-------|---------------|
| Swarm inference | [Fortytwo (arXiv:2510.24801)](https://arxiv.org/abs/2510.24801) | LLM peer-ranking + BradleyTerry aggregation |
| Multi-agent debate | [arXiv:2305.14325](https://arxiv.org/abs/2305.14325) | Debate round for hard queries (+12.8pp on math) |
| Proof of compute | [TOPLOC (arXiv:2501.16007)](https://arxiv.org/abs/2501.16007) | Activation-level proofs in patched llama.cpp |
| Agent diversity | [arXiv:2602.03794](https://arxiv.org/abs/2602.03794) | Use diverse models, not copies (2 diverse ≥ 16 same) |
| Mixture of agents | [arXiv:2406.04692](https://arxiv.org/abs/2406.04692) | MoA-style refinement in debate round |
| Adaptive routing | [RouteLLM (arXiv:2406.18665)](https://arxiv.org/abs/2406.18665) | Complexity classifier (Easy/Medium/Hard) |
| Speculative decoding | [EAGLE-3 (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) | Future: 4-6x per-node speedup |

---

## Roadmap

### Done

- Terminal UI with 5 tabs + browser WASM target (ratzilla)
- 21 real models with HuggingFace downloads
- llama-server auto-start (TOPLOC-enabled)
- Swarm inference: fan-out → debate → LLM peer-rank → BradleyTerry
- Complexity classifier (Easy/Medium/Hard) + Domain classifier
- Adaptive node count (3 easy, 5 hard + debate)
- ELO tracking + persistence
- Unified identity (one key for Solana + iroh + TOPLOC)
- iroh P2P + gossip + NAT traversal
- Auto-mesh via Solana node registry
- Auto-airdrop + auto-register for new users
- TOPLOC llama.cpp patch (activation-level proofs)
- Solana Anchor program (5 instructions + domain counters)
- Security hardening (constant-time auth, per-IP rate limit, proof freshness, gossip bounds)
- Rust, Python, TypeScript SDKs
- Tauri v2 desktop app

### Next

- [ ] SMTI token mint on devnet
- [ ] EAGLE-3 speculative decoding (4-6x speedup)
- [ ] Trained difficulty classifier (RouteLLM-style)
- [ ] React Native mobile app
- [ ] Full WASM browser build (gate remaining native deps)
- [ ] Domain AMM pools on Solana
- [ ] Mainnet migration

---

## Contributing

1. Fork and clone
2. `cargo build` — must compile clean
3. `cargo test` — all 48 tests pass
4. One feature/fix per PR

---

## License

MIT
