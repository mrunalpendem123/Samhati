# Samhati — Open Intelligence for Everyone, Forever

A peer-to-peer network where every device that uses AI also provides AI. Queries fan out to multiple nodes running small language models. Nodes peer-rank each other's answers. BradleyTerry aggregation selects the best response. No staking, no special hardware, no permission to join.

> [Whitepaper](whitepaper.pdf) · [GitHub](https://github.com/mrunalpendem123/Samhati)

## Quick Start

```bash
curl -sSL https://raw.githubusercontent.com/mrunalpendem123/Samhati/main/install.sh | bash
cd ~/Samhati && cargo run -p samhati-tui --bin samhati
```

## How It Works

```
You ask a question
  ↓
Complexity classifier → Easy (3 nodes) / Hard (5 nodes + debate)
  ↓
Fan-out to N nodes in parallel, each running llama.cpp
  ↓
Each node returns answer + Proof of Inference (signed logprob hashes)
  ↓
Random judge assignment (anti-collusion, BLAKE3 seed)
  ↓
Peer ranking via LLM — judges evaluate factual correctness only
  ↓
BradleyTerry MLE aggregation → winner
  ↓
EMA reputation update (dual-track: answer quality + ranking honesty)
  ↓
Settlement on Solana (proof hashes, reputation deltas, domain demand)
  ↓
Best answer shown to you
```

## Architecture

| Crate | What it does |
|-------|-------------|
| `samhati-tui` | Terminal UI — chat, dashboard, models, wallet, settings |
| `samhati-ranking` | BradleyTerry + EMA reputation + rewards (pure computation) |
| `samhati-toploc` | Proof of Inference — logprob hashing, Ed25519 signing, verification |
| `mesh-node` | HTTP API (axum) + iroh QUIC + gossip protocol |
| `inference-coordinator` | Distributed shard execution + KV cache + failover |
| `proximity-router` | Peer selection (std-only, zero external deps) |
| `shard-store` | Content-addressed weight cache (BLAKE3) |

## Proof of Inference

Not TOPLOC (which hashes internal layer activations). PoI hashes **output logprobs** — the top-K token probability distributions from each generation step:

1. For each token: record token ID + top-8 (token_id, logprob) pairs
2. Group into 32-token chunks → BLAKE3 hash each chunk
3. Sign `model_hash || token_count || chunk_hashes || timestamp || node_pubkey` with Ed25519
4. Verifier checks: model hash, token count, chunks, freshness (5 min), node binding, signature

**What it proves:** A model produced specific output distributions, signed by this node, recently.
**What it doesn't prove:** Which specific model ran internally. That's handled by peer ranking.

## Reputation System

Dual-track EMA (not ELO):

```
R = 0.4 × R_rank + 0.6 × R_gen

R_gen  = 0.85 × R_gen  + 0.15 × (1 if won, 0 if lost)     ← answer quality
R_rank = 0.85 × R_rank + 0.15 × accuracy                    ← ranking honesty
```

- **Uncertainty (σ):** Starts 0.4, decays 5%/round → 0.05. TrueSkill-inspired.
- **Adaptive audit:** New nodes ~50%, trusted nodes ~2%. Hyperbolic PoSP-inspired.
- **Domain-specific:** Separate R_gen per Code, Math, Reasoning, General.
- **Slash:** Failed proof → reputation 0, must recalibrate.
- **No staking.** Barrier is compute, not capital.

## Attack Resistance

| Attack | Defense |
|--------|---------|
| Sybil (fake nodes) | PoI requires real inference. Bad answers lose reputation. |
| Collusion | Random judge assignment via BLAKE3(timestamp + prompt + N). Changes every round. |
| Adversarial inputs | 21 models, 4 architectures. Debate round for hard queries. |
| Self-promotion | Ranking prompt ignores persuasion. Judges can't rank own answer. |
| Freeloading | Weak models lose ranking → reputation decays → selection drops. |

## Settlement

Solana devnet. Program: `AB7cSMLv1J7J28DKLMbzo2tyNp1kZSmE67a6Heoa5Mkr`

| PDA | Seeds | Stores |
|-----|-------|--------|
| ProtocolConfig | `[b"config"]` | emission rate, domain demand counters |
| NodeAccount | `[b"node", pubkey]` | reputation, model, rounds, rewards |
| RoundAccount | `[b"round", id]` | participants, proof hashes, deltas, winner |

## Incentive

- Winner: 60% of base emission
- All participants: 40% split by reputation
- Domain specialists: 1.5× on matching queries
- No staking. No fee to join. Earn immediately.

## Models

21 models, 0.5B–14B parameters, 4 domains:

| Domain | Models |
|--------|--------|
| General | Qwen2.5 (0.5B–14B), Llama-3.2-3B, Llama-3.1-8B, Gemma-3 (1B–4B), Mistral-7B, SmolLM2-1.7B |
| Code | Qwen2.5-Coder (0.5B–14B), DeepSeek-Coder-V2-Lite |
| Math | Qwen2.5-Math (1.5B–7B) |
| Reasoning | Phi-4-mini, DeepSeek-R1-Distill-14B |

## Network

- **Discovery:** Solana on-chain registry + iroh QUIC gossip
- **Identity:** One Ed25519 keypair → Solana address + iroh NodeId + PoI signer
- **Transport:** iroh QUIC with relay-assisted NAT traversal
- **Broadcast:** Model name + inference endpoint every 10 seconds

## Tests

```bash
cargo test                                        # 80+ unit tests
cargo test --test deep_e2e -- --ignored           # 9 proof/security tests (needs llama-server)
cargo test --test full_swarm_live -- --ignored    # full swarm round (needs 3 llama-servers)
cargo test --test reputation_live -- --ignored    # 10-round reputation test
cargo test --test solana_submit -- --ignored      # Solana devnet verification
```

## What's Real, What's Not

| Component | Status |
|-----------|--------|
| Swarm inference (fan-out → debate → rank → aggregate) | ✓ Tested against live servers |
| Proof of Inference (logprob hashing + Ed25519) | ✓ Deterministic, tamper-proof, tested |
| BradleyTerry aggregation (MM algorithm) | ✓ Converges in 10–25 iterations |
| EMA reputation (dual-track + uncertainty + audit) | ✓ Tested over 10 live rounds |
| Random judge assignment (BLAKE3 seed) | ✓ Implemented, tested |
| Solana registration | ✓ Live on devnet |
| Solana submit_round | ⚠ Code complete, prerequisites verified, needs TUI flow |
| iroh P2P gossip | ⚠ Code works, needs two physical machines to test |
| Browser WASM build | ⚠ Configured, not fully compiled to wasm32 |

## Contributing

```bash
cargo build    # must compile clean
cargo test     # all tests pass
```

## License

MIT
