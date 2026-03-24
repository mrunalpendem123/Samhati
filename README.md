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

> Built on **Solana** · Powered by **TOPLOC** · Inspired by **Fortytwo**

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Core Architecture](#core-architecture)
3. [Swarm Inference](#swarm-inference)
4. [TOPLOC — Proof of Honest Inference](#toploc--proof-of-honest-inference)
5. [Rewards — ELO Not Stake](#rewards--elo-not-stake)
6. [Self-Evolving Pipeline](#self-evolving-pipeline)
7. [Solana Integration](#solana-integration)
8. [Token Economics — SMTI](#token-economics--smti)
9. [Samhati vs Fortytwo vs Bittensor](#samhati-vs-fortytwo-vs-bittensor)
10. [Project Structure](#project-structure)
11. [Quick Start](#quick-start)
12. [CLI Reference](#cli-reference)
13. [Apps & SDK](#apps--sdk)
14. [Roadmap](#roadmap)
15. [Research Foundation](#research-foundation)

---

## How It Works

### The Request Lifecycle

```
1. User sends POST /v1/chat/completions to their local samhati-node
2. Complexity classifier scores the prompt → assigns N (1, 3, 5, 7, or 9 nodes)
3. ModelRegistry selects top-N nodes by ELO + domain speciality
4. iroh QUIC opens N parallel connections — fan-out
5. Phase 1: each node runs llama.cpp independently, attaches TOPLOC proof
6. Fan-in: N (answer, proof) pairs return. Bad proofs excluded and slashed
7. Phase 2: all N nodes receive all N answers → C(N,2) pairwise rankings
8. BradleyTerry aggregates → winner selected → SMTI emitted → Solana settled async
```

```
              User sends prompt
                    │
                    ▼
       ┌────────────────────────┐
       │  Complexity Classifier │
       │  (< 1ms, on-device)   │
       └───────────┬────────────┘
                   │  N = 1, 3, 5, 7, or 9
                   ▼
       ┌────────────────────────┐
       │  ModelRegistry         │
       │  top-N by ELO + domain │
       └───────────┬────────────┘
                   │
   ════════════════╪════════════════ Phase 1: Generate ════
                   │
           ┌───────┼───────┬───────────┐
           ▼       ▼       ▼           ▼
       ┌──────┐┌──────┐┌──────┐   ┌──────┐
       │Node 1││Node 2││Node 3│...│Node N│
       │3B    ││7B    ││14B   │   │3B    │
       │Hindi ││Code  ││Math  │   │Gen.  │
       └──┬───┘└──┬───┘└──┬───┘   └──┬───┘
          │       │       │           │
          │answer │answer │answer     │answer
          │+proof │+proof │+proof     │+proof
          ▼       ▼       ▼           ▼
       ┌─────────────────────────────────────┐
       │  TOPLOC Verification                 │
       │  CPU-only, < 1ms per proof           │
       │  Invalid → discard + slash on-chain  │
       └──────────────────┬──────────────────┘
                          │
   ═══════════════════════╪════════════ Phase 2: Rank ════
                          │
       ┌──────────────────▼──────────────────┐
       │  All N nodes receive all N answers   │
       │  Each produces C(N,2) pairwise       │
       │  rankings + 50-100 token reasoning   │
       └──────────────────┬──────────────────┘
                          │
       ┌──────────────────▼──────────────────┐
       │  BradleyTerry Aggregation            │
       │  ELO-weighted MLE → winner           │
       └──────────┬──────────────┬───────────┘
                  │              │
                  ▼              ▼
       ┌────────────────┐ ┌──────────────────┐
       │ Winner returned │ │ Solana settlement │
       │ to user         │ │ (async, off       │
       │ immediately     │ │  critical path)   │
       └────────────────┘ └──────────────────┘
```

The user receives the winning answer **before** Solana settlement completes. Settlement is
always asynchronous and never on the critical path.

---

## Core Architecture

| Component | Purpose | Implementation |
|-----------|---------|----------------|
| **samhati-node** | Main binary — HTTP API, iroh P2P, model runner, wallet | Rust · axum · llama.cpp |
| **SwarmCoordinator** | Fan-out, collect, verify, rank, aggregate | `crates/samhati-swarm` |
| **TOPLOC verifier** | Cryptographic proof of honest inference | `crates/samhati-toploc` |
| **BradleyTerry** | Pairwise ranking aggregation, ELO update | `crates/samhati-ranking` |
| **ReputationStore** | ELO history, domain speciality, uptime | SQLite → Solana PDAs |
| **Solana program** | Identity, slash, rewards, settlement | Anchor framework |

Everything is Rust. Fully peer-to-peer — no Samhati-operated servers on the critical path. The
Solana program is the only shared state, immutable and public.

### Per-Node Services

Every Samhati node runs three iroh endpoints simultaneously:

| Endpoint | Purpose |
|----------|---------|
| **Inference** | Receives request → runs llama.cpp → returns answer + TOPLOC proof |
| **Ranking** | Receives N answers → returns pairwise preference scores with reasoning |
| **DERP relay** | Contributes NAT traversal connectivity, earns small SMTI bonus |

DERP relay means nodes earn SMTI even when hardware is too slow to win inference — every device
contributes something.

### Actual Inference Stack

```
samhati-node (Rust orchestrator)
  ├── iroh QUIC (peer discovery + RPC)
  ├── SwarmCoordinator (fan-out → verify → rank → aggregate)
  │     ├── Each peer: llama-server subprocess (llama.cpp)
  │     ├── TOPLOC proof verification (CPU-only)
  │     └── BradleyTerry winner selection
  └── Solana settlement (async, off critical path)
```

Inference runs through **llama.cpp** (via `llama-server` or `llama-cli` subprocess). The Rust
layer handles orchestration, networking, ranking, and settlement — not GPU compute.

---

## Swarm Inference

Implements [Fortytwo: Swarm Inference with Peer-Ranked Consensus](https://arxiv.org/abs/2510.24801).

### Why Swarm Beats Single Model

**85.90% on GPQA Diamond** vs **68.69%** for majority voting (+17.21pp):

- **Diversity cancels errors.** Five different models have different blind spots.
- **Peer ranking adds reasoning.** 50-100 token reasoning chains per comparison — these become
  training data for the self-evolving pipeline.
- [arXiv:2602.03794](https://arxiv.org/abs/2602.03794) proves **2 diverse agents ≥ 16 homogeneous**.

### Complexity Routing

| Query type | N nodes | Example | Latency |
|------------|---------|---------|---------|
| Trivial | 1 (local) | "What is 2+2?" | < 1s |
| Conversational | 3 | "Explain photosynthesis" | 3-8s |
| Reasoning | 5 | "Debug this Rust code" | 5-15s |
| Hard | 7 | "Solve this AIME problem" | 10-30s |
| Expert | 9 | "GPQA Diamond level" | 15-45s |

Users can override: **Quick** (N=3) vs **Best** (N=7).

### Peer Ranking

In Phase 2, the same N nodes that generated answers receive all N answers and produce pairwise
rankings. For N=5: C(5,2) = 10 comparisons, each with:

- **Preference score:** P(A beats B) ∈ [0, 1]
- **Reasoning chain:** 50-100 tokens explaining the preference
- **Domain tags:** auto-extracted topic classification

Reasoning chains are **not discarded** — they become training data.

### BradleyTerry Aggregation

```
P(i beats j) = exp(β_i) / (exp(β_i) + exp(β_j))

Winner = argmax(β_i)
ELO Δ  = K × (actual - expected),  K = 32 new / 16 established
```

High-ELO nodes' rankings carry more weight — self-reinforcing quality signal.

---

## TOPLOC — Proof of Honest Inference

[TOPLOC](https://arxiv.org/abs/2501.16007) (PrimeIntellect) cryptographically proves a node ran
real inference — without requiring any GPU on the verifier side.

| Metric | Value |
|--------|-------|
| Proof size | 258 bytes per 32 tokens |
| Detection rate | 100% |
| False positive rate | 0% |
| Verifier GPU | None (CPU-only) |
| Verification time | < 1ms |

A node claiming 70B but running 1B produces a proof that fails — 100% detection, 0 false positives.

### Sybil Resistance

| Layer | Mechanism | Attacker cost | Honest user cost |
|-------|-----------|---------------|------------------|
| 1 | TOPLOC calibration (10 test prompts) | Real GPU × N fake nodes | 2 min electricity |
| 2 | ELO quality gates | Hundreds of honest rounds | Zero |
| 3 | iroh NodeID fingerprinting | Unique hardware per node | Zero |

No financial stake required from honest users. Physical computation **is** the stake.

---

## Rewards — ELO Not Stake

A [2025 Bittensor analysis](https://arxiv.org/abs/2507.02951) found rewards "overwhelmingly
driven by stake." Samhati eliminates this — rewards are proportional to **demonstrated quality**.

```
winner_share      = base_emission × 0.60
participant_share = base_emission × 0.40
each participant  = participant_share × (elo_i / sum(elo_all))
domain bonus      = × 1.5 if specialist model matches query domain
```

### Domain-Weighted Emissions

| Scenario | Multiplier | Effect |
|----------|------------|--------|
| Generic node, Hindi query | 1.0x | Standard |
| Hindi SLM, Hindi query | 1.5x | 50% bonus |
| Oversaturated domain | 1.5x shared many ways | Equilibrium |
| Undersaturated domain | 1.5x, few competitors | Strong signal to specialize |

Operators see earnings in their dashboard and rationally switch to specialist models.

### ELO Mechanics

- New nodes: K=32 · Established (>1000 rounds): K=16 · Floor: 100
- TOPLOC slash: **-200 ELO + 10% stake burn**
- Two slashes → below routing threshold · Three → effectively fatal
- Asymmetry (slow to earn, fast to lose) makes honesty dominant

---

## Self-Evolving Pipeline

Every round with BradleyTerry confidence ≥ 0.70 emits a training example:

```
More users → more queries → more training data
  ↓
Better specialist SLMs (monthly QLoRA fine-tune)
  ↓
Higher accuracy → higher peer-rank scores → higher ELO
  ↓
More SMTI rewards → more operators run specialists
  ↓
Better answers → more users → [repeat]
```

| Parameter | Value |
|-----------|-------|
| Base model | Qwen2.5-3B or Llama-3.2-3B |
| Method | QLoRA (r=64, α=128) |
| Hardware | Single RTX 3090 |
| Time | 2-6h (10K examples) / 6-12h (100K) |
| Cost | ~$20-50 on RunPod |
| Output | GGUF Q4_K_M for llama.cpp |
| Release | HuggingFace, Apache 2.0 |
| Cadence | Monthly per domain |

Implemented in `pipeline/`: collector → trainer (SFT + DPO) → exporter (GGUF) → scheduler.

---

## Solana Integration

Every identity, proof, ranking, reward, and slash is on-chain — permanent and publicly
verifiable.

| Property | Solana | Ethereum | Monad (Fortytwo) |
|----------|--------|----------|-------------------|
| Tx cost | $0.00025 | $5-50 | Pre-mainnet |
| Block time | 400ms | 12s | Pre-mainnet |
| DePIN projects | 250+ | Few | None |
| Rust SDK | Native (Anchor) | Solidity | EVM |

Per-round settlement: ~$0.00065 (7 ELO updates + 1 proof hash).

### Anchor Program — 5 Instructions

Deployed in `programs/samhati-protocol/`:

| Instruction | Effect |
|-------------|--------|
| `register_node` | Create NodeAccount PDA. ELO=1500, calibrated=false |
| `calibrate_node` | Set calibrated=true after TOPLOC calibration round |
| `submit_round` | Store proof hashes, rankings, ELO deltas, winner, SMTI emitted |
| `slash_node` | ELO -= 200, burn 10% stake |
| `emit_rewards` | Permissionless — transfer SMTI from vault to operator wallet |

### On-Chain Data

```rust
// NodeAccount PDA — seeds=[b"node", pubkey]
elo_score:       i32,     // [100, ∞), starts 1500
calibrated:      bool,    // must pass TOPLOC calibration
model_name:      String,  // "samhati-rust-coder-14b-v2"
domain_tags:     u64,     // bitmask
total_rounds:    u64,
rounds_won:      u64,
slash_count:     u8,
staked:          u64,
pending_rewards: u64,

// RoundAccount PDA — seeds=[b"round", round_id]
participant_nodes: Vec<Pubkey>,
proof_hashes:      Vec<[u8; 32]>,
peer_rankings:     Vec<PairScore>,
elo_deltas:        Vec<i32>,
winner:            Pubkey,
smti_emitted:      u64,
settled:           bool,
```

---

## Token Economics — SMTI

Solana SPL token. Fixed supply: **1,000,000,000** (1B).

| Allocation | % | Amount | Purpose |
|------------|---|--------|---------|
| Node emission | 60% | 600M | Protocol rewards for inference |
| Ecosystem | 20% | 200M | Grants, partnerships, dev incentives |
| Team | 15% | 150M | 4-year vest, 1-year cliff |
| Liquidity | 5% | 50M | Raydium/Jupiter DEX liquidity |

**Emission:** ~164,000 SMTI/day initially, halving every ~2.5 years over 10 years.

### Domain Pool AMM

Each domain (Hindi, Rust, Math, DeFi, General) has a Solana liquidity pool. Emissions
proportional to net SMTI staked — self-regulating market for compute allocation:

```
Hindi undersupplied → users stake into Hindi pool
→ Hindi emission ↑ → Hindi nodes earn more
→ more operators run Hindi SLM → supply ↑ → equilibrium
```

---

## Samhati vs Fortytwo vs Bittensor

| Property | Samhati | Fortytwo | Bittensor |
|----------|---------|----------|-----------|
| Chain | **Solana** (live) | Monad (pre-mainnet) | Substrate |
| Reward signal | **ELO** (quality) | Reputation (opaque) | Stake (capital) |
| Proof of compute | **TOPLOC** (crypto) | Capability tests | None |
| P2P transport | **iroh QUIC** (open) | Undisclosed | Custom |
| Inference | Swarm + BradleyTerry | Swarm + BradleyTerry | Subnet competition |
| Self-evolving | Domain SLMs | Strand Rust Coder | SN9 |
| User cost | **Free** | API pricing | Subnet fees |
| Sybil resistance | TOPLOC proof | Stake + calibration | Stake |

---

## Project Structure

```
Samhati/
├── crates/
│   ├── samhati-swarm/              # Swarm consensus engine
│   │   └── src/
│   │       ├── coordinator.rs      # SwarmCoordinator: fan-out → verify → rank → aggregate
│   │       ├── classifier.rs       # ComplexityClassifier: prompt → N nodes (1-9)
│   │       ├── registry.rs         # ModelRegistry: ELO-ranked node selection
│   │       └── types.rs            # InferenceRequest, NodeResponse, SwarmResult
│   ├── samhati-ranking/            # ELO + BradleyTerry
│   │   └── src/
│   │       ├── elo.rs              # EloRating, adaptive K-factor, per-domain
│   │       ├── bradley_terry.rs    # MLE aggregation from pairwise outcomes
│   │       └── rewards.rs          # SMTI emission calculator, domain bonuses
│   ├── samhati-toploc/             # Proof of honest inference
│   │   └── src/
│   │       ├── prover.rs           # BLAKE3 logit hashes + Ed25519 signing
│   │       ├── verifier.rs         # CPU-only proof validation
│   │       └── calibration.rs      # 10-prompt new-node verification
│   ├── samhati-tui/                # Terminal UI (ratatui)
│   │   └── src/
│   │       ├── app.rs              # 5-tab app: Chat, Dashboard, Models, Wallet, Settings
│   │       ├── node_runner.rs      # Spawns llama-server subprocess
│   │       └── swarm.rs            # Multi-node swarm orchestration
│   ├── mesh-node/                  # Network node binary
│   │   └── src/
│   │       ├── main.rs             # CLI: gossip, serve, dist-run, api, etc.
│   │       ├── api.rs              # OpenAI-compatible REST API (axum)
│   │       ├── swarm_bridge.rs     # Wires SwarmCoordinator to iroh transport
│   │       ├── inference.rs        # HTTP proxy + llama.cpp subprocess backends
│   │       ├── server.rs           # iroh QUIC inference endpoint
│   │       └── protocol.rs         # Gossip message types
│   ├── inference-coordinator/      # Distributed execution planner
│   ├── proximity-router/           # Peer scoring + SwarmRegistry
│   ├── cluster-manager/            # VRAM-aware constraint solver
│   ├── model-config/               # Config parsing + VRAM estimation
│   └── shard-store/                # Content-addressed weight cache (blake3)
├── programs/
│   └── samhati-protocol/           # Solana Anchor program (5 instructions)
├── app-frontend/                   # React/Vite web UI
├── app-tauri/                      # Tauri desktop app
├── sdk/rust/                       # OpenAI-compatible Rust SDK
├── pipeline/                       # Self-evolving training pipeline (Python)
│   ├── collector.py                # Swarm round → training examples
│   ├── trainer.py                  # QLoRA SFT + DPO
│   ├── exporter.py                 # LoRA merge → GGUF export
│   └── scheduler.py                # Monthly per-domain trigger
└── scripts/                        # Deployment utilities
```

---

## Quick Start

### Prerequisites

- Rust 1.82+ (`rustup update stable`)
- llama.cpp (`brew install llama.cpp` on macOS)

```bash
git clone https://github.com/mrunalpendem123/Samhati.git
cd Samhati
cargo build
```

### Run

```bash
cargo run -p samhati-tui
```

That's it. The TUI handles everything:

1. **Models tab** → pick a model → press **Enter** → downloads GGUF from HuggingFace
2. Once downloaded, `llama-server` starts automatically on port 8000
3. **Chat tab** → type a message → get AI response (single node)
4. **Swarm mode** → go back to Models → press **s** on another model → starts a second
   `llama-server` on port 8001 and adds it to the swarm
5. Now chat goes through **BradleyTerry ranking** — you see winner, confidence %, node count

### Keyboard

| Key | Action |
|-----|--------|
| `Tab` / `1-5` | Switch tabs |
| `Enter` | Send message (Chat) / Install + activate model (Models) |
| `s` | Add model as swarm node (Models) |
| `a` | Request SOL airdrop (Wallet) |
| `q` | Quit |

### Tabs

- **Chat** — AI chat, routes through swarm when multiple nodes are running
- **Dashboard** — ELO score, SMTI balance, inferences served, peers, last swarm round
- **Models** — 24+ models (Qwen, Llama, Phi, domain specialists), download with progress
- **Wallet** — Solana wallet (devnet): balance, transactions, airdrop
- **Settings** — API endpoint, VRAM, device detection

---

## Roadmap

### Phase 1 — Core Swarm (Weeks 1-4)

- [x] samhati-node: axum HTTP + iroh QUIC + llama.cpp backend
- [x] SwarmCoordinator: fan-out, fan-in, TOPLOC verify, peer rank, aggregate
- [x] BradleyTerry: pairwise aggregation, ELO update
- [ ] ReputationStore: SQLite-backed ELO persistence
- [ ] 3-node testnet with measurable quality improvement

### Phase 2 — App and Incentives (Weeks 5-8)

- [x] Tauri desktop app: chat UI, model runner, SMTI wallet
- [x] Solana Anchor program: register, calibrate, submit_round, slash, emit
- [x] TOPLOC calibration round as Sybil gate
- [ ] SMTI emission to node operators
- [ ] Public dashboard: live node count, inferences/24h

### Phase 3 — Speed and Quality (Weeks 9-12)

- [ ] EAGLE-3 speculative decoding via SGLang (3-6.5x speedup)
- [ ] Cascade classifier: trivial queries answered locally
- [ ] Domain routing: specialist model preference
- [ ] Devnet → Mainnet Solana migration
- [ ] iroh DERP relay nodes

### Phase 4 — First Domain SLM (Month 4-6)

- [x] Swarm data collection pipeline (Python)
- [ ] First domain SLM: QLoRA on Qwen2.5-3B from swarm data
- [ ] HuggingFace open-source release
- [ ] Domain bonus emission
- [ ] React Native mobile app

### Phase 5 — Full Ecosystem (Month 7+)

- [ ] Domain AMM pools on Solana
- [ ] Multi-coordinator geographic sharding
- [ ] SMTI on Raydium/Jupiter
- [ ] Developer API with SLA tiers
- [ ] Continuous monthly SLM pipeline

---

## Scaling

| Network size | Mechanism | Latency | Quality |
|-------------|-----------|---------|---------|
| 1K nodes | Flat ModelRegistry | Baseline | Baseline |
| 100K nodes | Kademlia DHT (iroh) | Same | Better (more specialists) |
| 1M nodes | Geographic ELO sharding | Reduced | Better |
| 10M nodes | 40% answered locally (cascade) | Much reduced | Best |

---

## Research Foundation

| Domain | Paper | Key finding |
|--------|-------|-------------|
| Swarm inference | [arXiv:2510.24801](https://arxiv.org/abs/2510.24801) | 85.90% GPQA vs 68.69% majority vote |
| Proof of compute | [arXiv:2501.16007](https://arxiv.org/abs/2501.16007) | 258B/32 tokens, 100% detection, CPU verify |
| Speculative decoding | [arXiv:2503.01840](https://arxiv.org/abs/2503.01840) | 3-6.5x speedup (EAGLE-3) |
| Heterogeneous swarm | [arXiv:2602.03794](https://arxiv.org/abs/2602.03794) | 2 diverse agents ≥ 16 homogeneous |
| Mixture of agents | [arXiv:2406.04692](https://arxiv.org/abs/2406.04692) | 65.1% AlpacaEval, beats GPT-4o |
| Bittensor analysis | [arXiv:2507.02951](https://arxiv.org/abs/2507.02951) | Stake-reward misalignment |

---

## Contributing

1. Fork and clone
2. `cargo build` — must compile clean
3. `cargo test` — all tests pass
4. One feature/fix per PR

---

## License

MIT
