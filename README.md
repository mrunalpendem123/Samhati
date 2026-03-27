# Samhati — Open Intelligence for Everyone, Forever

**Samhati** is a decentralized AI inference network — the BitTorrent of Intelligence. Every device that uses AI also provides AI. No corporation in the middle. No meter running. Just a global mesh of machines thinking together.

```
BitTorrent:  download (consume)  +  upload (seed)    →  open bandwidth
Samhati:     query AI (use)      +  serve others (node) →  open intelligence
             SMTI tokens reward node operators from protocol emission
```

The core mechanism is **swarm intelligence**: queries fan out to N independent nodes via **iroh QUIC**. Each node runs **llama.cpp** inference and attaches a **TOPLOC** cryptographic proof. The same N nodes peer-rank each other's answers. **BradleyTerry** aggregation selects the winner — achieving **85.90% on GPQA Diamond** vs 68.69% for majority voting.

> Built on **Solana** · Powered by **TOPLOC**

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [How It Works](#how-it-works)
3. [Swarm Inference](#swarm-inference)
4. [TOPLOC Proof System](#toploc-proof-system)
5. [ELO + BradleyTerry Ranking](#elo--bradleyterry-ranking)
6. [iroh P2P Networking](#iroh-p2p-networking)
7. [Solana Smart Contract](#solana-smart-contract)
8. [Unified Identity System](#unified-identity-system)
9. [Distributed Inference Coordinator](#distributed-inference-coordinator)
10. [mesh-node HTTP API](#mesh-node-http-api)
11. [Model System](#model-system)
12. [Settlement Flow](#settlement-flow)
13. [Security Hardening](#security-hardening)
14. [Architecture](#architecture)
15. [Project Structure](#project-structure)
16. [Research Foundation](#research-foundation)
17. [Roadmap](#roadmap)
18. [Contributing](#contributing)

---

## Quick Start

### Terminal (native)

```bash
curl -sSL https://raw.githubusercontent.com/mrunalpendem123/Samhati/main/install.sh | bash
cd ~/Samhati && cargo run -p samhati-tui --bin samhati
```

### Browser (WASM — experimental)

```bash
cd crates/samhati-tui
rustup target add wasm32-unknown-unknown
cargo install trunk
trunk serve   # opens at http://localhost:8080
```

The browser build uses **ratzilla** (DomBackend) to render the same ratatui UI in your browser via WebAssembly. Chat works via remote API — set the endpoint in Settings tab.

### TUI Keys

| Key | Action |
|-----|--------|
| `Tab` / `1-5` | Switch tabs |
| `Enter` | Send message (Chat) / Install + activate model (Models) |
| `s` | Add model as swarm node (Models) |
| `r` | Connect to friend by NodeId (Models) |
| `a` | Request SOL airdrop (Wallet) |
| `q` | Quit |

### Tabs

- **Chat** — AI chat with swarm consensus. Shows difficulty, domain, confidence, node count
- **Dashboard** — ELO score, sparkline, network demand bars, swarm nodes, last round
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

## Swarm Inference

**Files:** `crates/samhati-swarm/src/coordinator.rs`, `crates/samhati-tui/src/swarm.rs`

### Query Lifecycle

**Step 1 — Classify difficulty.** The complexity classifier examines the prompt and assigns Easy, Medium, or Hard. Easy/Medium queries use 3 nodes. Hard queries use 5 nodes plus a debate round.

**Step 2 — Select nodes.** The coordinator queries the SwarmOrchestrator for the top-N nodes by ELO score. Nodes are filtered to those running a model compatible with the query's domain.

**Step 3 — Fan-out.** The coordinator dispatches the prompt to all N nodes in parallel using `tokio::task::JoinSet`. Each node runs llama-server inference and returns a `NodeAnswer`:

```rust
pub struct NodeAnswer {
    pub node_id: String,
    pub answer: String,
    pub latency_ms: u64,
    pub model: String,
    pub proof_hash: Option<[u8; 32]>,   // BLAKE3 of TOPLOC proof
}
```

If a node times out (120s default), it's skipped. The round continues as long as at least 1 response arrives.

**Step 4 — Debate round (hard queries only).** Each node sees all other nodes' answers and rewrites its own. This implements multi-agent debate from arXiv:2305.14325, which improves accuracy by +12.8 percentage points on math benchmarks. Nodes self-correct without re-running full inference — they refine their reasoning based on peer perspectives.

**Step 5 — Peer ranking.** Every node ranks every other node's answer in pairwise comparisons. For N nodes, this produces C(N,2) rankings. Each ranking contains:

```rust
pub struct PairwiseRanking {
    pub ranker_node_id: String,
    pub node_a: String,
    pub node_b: String,
    pub preference: f32,          // 0.0 = prefer A, 1.0 = prefer B
    pub reasoning: String,        // 50-100 token explanation
    pub domain_tags: Vec<String>,
}
```

Nodes cannot judge their own answers (excluded from self-ranking).

**Step 6 — BradleyTerry aggregation.** The MM algorithm converts pairwise preferences into a global ranking with strength estimates and a confidence score. The winner is the node with highest estimated strength.

**Step 7 — ELO update.** Winner gets +16 ELO (K/2 where K=32). Each loser gets -16/(N-1) ELO. Updates persist to disk. Domain-specific ELO is tracked separately (a node can be ELO 1800 for Code but 1400 for Math).

### SwarmConfig

```rust
pub struct SwarmConfig {
    min_responses: usize,                  // minimum verified responses to proceed
    toploc_required: bool,                 // reject nodes without valid proof
    training_confidence_threshold: f32,    // capture training data above this confidence
    max_fan_out_timeout: Duration,         // per-node timeout
}
```

---

## TOPLOC Proof System

**Files:** `crates/samhati-toploc/src/`

TOPLOC (Top-K Logit Output Commitment) is a cryptographic proof that a node actually ran inference with the correct model. It works by committing to the top-K logit values at each token generation step, then signing the commitment chain.

### How Proofs Are Created (Prover)

During inference, the prover captures the top 8 logit values after each token is generated:

```rust
pub struct TokenLogits {
    pub token_id: u32,                    // the token that was actually generated
    pub top_k: Vec<(u32, f32)>,          // top 8 (token_id, logit_value) pairs
}
```

After inference completes, the prover:

1. **Chunks the logits** into groups of 32 tokens (DEFAULT_CHUNK_SIZE = 32)
2. **Hashes each chunk** using BLAKE3 over a deterministic byte serialization (tokens sorted by ID, big-endian fixed-width encoding)
3. **Builds the proof structure** with model hash, token count, chunk hashes, timestamp, and the prover's Ed25519 public key
4. **Signs everything**: the signable message is `model_hash || token_count || chunk_hashes || timestamp || node_pubkey`
5. **Returns the complete proof:**

```rust
pub struct ToplocProof {
    pub model_hash: [u8; 32],            // BLAKE3(model_id_string)
    pub token_count: u32,
    pub chunk_proofs: Vec<LogitChunkProof>,  // one per 32-token chunk
    pub timestamp: u64,                   // Unix seconds
    pub node_pubkey: [u8; 32],           // Ed25519 public key of the prover
    pub node_signature: [u8; 64],        // Ed25519 signature over signable message
}
```

**Why it works:** Different models, quantization methods, or weight files produce different intermediate activations, which produce different top-K logit values, which produce different BLAKE3 hashes. A node that fakes inference cannot produce matching hashes without actually running the model.

### How Proofs Are Verified (Verifier)

The verifier runs 8 checks in order:

1. **Key registration** — reject if no public keys are registered (prevents misconfigured verifiers from accepting anything)
2. **Model hash** — `BLAKE3(claimed_model_id)` must match `proof.model_hash`
3. **Token count** — must match the actual output token count
4. **Chunk count** — must equal `ceil(token_count / 32)`
5. **Chunk indices** — must be sequential: 0, 1, 2, ...
6. **Freshness** — `now - proof.timestamp` must be ≤ `max_proof_age_secs` (default 300 seconds / 5 minutes)
7. **Node binding** — `proof.node_pubkey` must be in the registered key set (prevents node A from claiming node B's proof)
8. **Signature** — Ed25519 verify against the specific `node_pubkey` declared in the proof

```rust
pub enum VerificationResult {
    Valid,
    InvalidModelHash { expected: [u8; 32], got: [u8; 32] },
    TokenCountMismatch { expected: u32, got: u32 },
    InvalidSignature,
    NoKeysRegistered,
    MissingChunks { expected: usize, got: usize },
    UnknownModel(String),
    InvalidProofStructure(String),
    StaleProof { proof_ts: u64, now_ts: u64, max_age_secs: u64 },
    UnknownNode,
}
```

### Calibration

Before a new node is trusted, it must pass a calibration round. The verifier sends 10 prompts from a built-in template pool (20 templates covering capitals, math, translations, chemistry, etc.). The node generates responses with TOPLOC proofs. All 10 proofs must verify correctly. On first calibration, any valid proof is accepted. On subsequent calibrations, the proof hashes must match previous runs (deterministic inference = deterministic proofs).

---

## ELO + BradleyTerry Ranking

**Files:** `crates/samhati-ranking/src/`

### ELO System

Every node starts at ELO 1500. After each swarm round, the winner gains ELO and losers lose ELO. The system uses adaptive K-factors:

| Node experience | K-factor | Effect |
|----------------|----------|--------|
| < 1000 rounds | K = 32 | Large swings — new nodes find their level quickly |
| ≥ 1000 rounds | K = 16 | Smaller swings — established nodes change slowly |

**Expected score formula** (standard ELO):
```
E(A vs B) = 1 / (1 + 10^((ELO_B - ELO_A) / 400))
```

**Update formula:**
```
delta = K * (actual_score - expected_score)
new_elo = max(old_elo + delta, 100)    // ELO floor = 100
```

**Slash penalty:** If a node fails TOPLOC verification, it loses 200 ELO immediately.

**Domain-specific ELO:** Each node tracks separate ELO scores per domain (Code, Math, Reasoning, General). A node might be ELO 1800 for Code queries but 1400 for Math queries.

### BradleyTerry Aggregation

Given N nodes with C(N,2) pairwise rankings from peer evaluation, the BradleyTerry model estimates each node's latent "strength" using Maximum Likelihood Estimation via the MM (Majorization-Minimization) algorithm:

**Model:** The probability that node i beats node j is:
```
P(i > j) = exp(β_i) / (exp(β_i) + exp(β_j))
```

**MM Algorithm (up to 100 iterations, tolerance 1e-6):**

1. Initialize all β_i = 0
2. For each node i:
   - numerator = sum of (weight × wins) across all comparisons involving i
   - denominator = sum of weight / (exp(β_i) + exp(β_j)) for all comparisons
   - β_i_new = ln(numerator / denominator)
3. Normalize: subtract mean(β) from all values (prevents drift)
4. Check convergence: max|β_new - β_old| < 1e-6
5. Winner = argmax(β). Confidence = softmax(β_winner) / sum(softmax(β_all))

**Ranker ELO weighting:** Rankings from higher-ELO nodes carry more weight. A 1800-ELO node's preference counts more than a 1200-ELO node's preference.

### Reward Distribution

```
base_emission = 1000 SMTI per round

Winner pool:      60% × base_emission
Participant pool: 40% × base_emission

Winner reward = winner_pool + (participant_pool × elo_share)
Loser reward  = participant_pool × elo_share

elo_share(node) = node_elo / sum(all_participant_elos)

Domain bonus: if node's model specializes in query domain → reward × 1.5
```

Domain specialists (Code, Math, Reasoning models) earn 50% more on matched queries. This incentivizes operators to run what the network needs.

---

## iroh P2P Networking

**Files:** `crates/samhati-tui/src/network.rs`, `crates/mesh-node/src/server.rs`, `crates/mesh-node/src/protocol.rs`

### How Nodes Find Each Other

Samhati uses a two-layer discovery system: **Solana** for bootstrap and **iroh gossip** for real-time updates.

**Layer 1 — Solana bootstrap (on startup):**
1. Read all `NodeAccount` PDAs from Solana (free RPC call, no transaction needed)
2. Each NodeAccount's `operator` pubkey is the node's iroh NodeId (same Ed25519 key)
3. Bootstrap iroh with all discovered NodeIds
4. iroh connects via QUIC with relay-assisted NAT traversal

**Layer 2 — Gossip (continuous):**
Every 10 seconds, each node broadcasts its `NodeAnnouncement` on the gossip topic:

```rust
pub struct NodeAnnouncement {
    pub solana_pubkey: String,     // base58 Solana address
    pub iroh_node_id: String,     // hex-encoded Ed25519 pubkey
    pub inference_url: String,    // empty = presence-only (no localhost broadcast for SSRF safety)
    pub model_name: String,       // e.g. "Qwen2.5-7B"
    pub port: u16,
}
```

### Protocol Messages

All gossip messages are JSON, parsed with a 64 KB size guard to prevent OOM from malicious peers:

```rust
// Capability announcement (peer → all)
pub struct CapabilityPayload {
    pub node_id: String,
    pub free_vram_gb: f64,
    pub bandwidth_mbps: f64,
    pub reliability: f64,
    pub gpu_capacity_score: f64,
    pub rtt_ms: f64,
    pub kv_bits: u8,
    pub context: usize,
    pub quant_bits: u8,
    pub role: NodeRole,                // Inference | Routing | Cache
    pub layers_hosted: Vec<HostedLayerRange>,
    pub load_score: f64,               // 0.0 = idle, 1.0 = saturated
    pub uptime_secs: u64,
    pub cached_shard_hashes: Vec<String>,
}

// Ping/Pong for RTT measurement
pub struct PingMessage { from_id, target_id, nonce }
pub struct PongMessage { from_id, target_id, nonce }

// Model list announcement
pub struct ModelsAnnouncement { node_id, models: Vec<ModelSummary> }
```

### QUIC RPC Protocol

For actual inference requests between nodes, Samhati uses a binary RPC protocol over iroh QUIC streams:

- **ALPN identifier:** `mesh-inference/2`
- **Wire format:** 4-byte big-endian length prefix + bincode-serialized payload
- **Message types:** `0x00` = Inference request, `0x01` = Replay request (fault recovery)
- **Size limit:** 256 MB max per deserialized message

### Network Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Node A     │     │   Node B     │     │   Node C     │
│  (Qwen-7B)  │     │  (Llama-8B)  │     │  (Coder-3B)  │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       └────────────────────┼────────────────────┘
                            │
                    iroh gossip (QUIC)
                    NAT traversal via relay
                            │
                    ┌───────▼────────┐
                    │ Solana devnet  │
                    │ (node registry │
                    │  + settlement) │
                    └────────────────┘
```

---

## Solana Smart Contract

**Files:** `programs/samhati-protocol/src/`

**Program ID:** `AB7cSMLv1J7J28DKLMbzo2tyNp1kZSmE67a6Heoa5Mkr` (devnet)

### On-Chain Accounts

**ProtocolConfig** (PDA: `[b"config"]`) — global settings:
- `authority` — coordinator signer (can submit rounds, slash nodes)
- `smti_mint` / `reward_vault` — token accounts for SMTI
- `base_emission_per_round` — SMTI emitted per round (default 1000)
- `domain_code/math/reasoning/general` — running counters of queries per domain
- `total_rounds` / `total_smti_emitted` — protocol-wide stats

**NodeAccount** (PDA: `[b"node", operator_pubkey]`) — per-node:
- `operator` — Solana pubkey (= iroh NodeId when hex-encoded)
- `elo_score` — starts 1500, floor 100
- `calibrated` — passed TOPLOC calibration?
- `model_name` — e.g. "Qwen2.5-7B" (max 64 bytes)
- `total_rounds` / `rounds_won` — participation stats
- `pending_rewards` — unclaimed SMTI lamports
- `slash_count` — number of times slashed

**RoundAccount** (PDA: `[b"round", round_id_le_bytes]`) — per-round:
- `participants` — up to 256 node pubkeys
- `proof_hashes` — BLAKE3 of each node's TOPLOC proof
- `elo_deltas` — signed i32 per participant
- `winner` — must be in participants list
- `domain` — 0=General, 1=Code, 2=Math, 3=Reasoning

### Instructions

| Instruction | Who | What |
|------------|-----|------|
| `initialize` | Deployer (once) | Create ProtocolConfig, set authority + mint + vault |
| `register_node` | Any node | Create NodeAccount PDA, set ELO=1500 |
| `calibrate_node` | Authority | Mark node as TOPLOC-verified |
| `submit_round` | Authority | Create RoundAccount, update all participant ELOs, credit rewards, increment domain counters |
| `slash_node` | Authority | Reduce ELO by 200, burn 10% of stake |
| `emit_rewards` | Anyone | Transfer pending rewards from vault to node operator |

**submit_round validation:**
- Array lengths must match (participants = proof_hashes = elo_deltas)
- Winner must be in participants list
- Participant count ≤ 256
- Each participant's NodeAccount passed via `remaining_accounts`

---

## Unified Identity System

**File:** `crates/samhati-tui/src/identity.rs`

One Ed25519 keypair stored at `~/.samhati/identity.json` derives every identity Samhati needs:

```
                    ┌─────────────────────┐
                    │  Ed25519 Secret Key  │
                    │    (32 bytes)        │
                    └─────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
    ┌─────────▼──────┐ ┌─────▼──────┐ ┌──────▼─────────┐
    │ Solana Wallet  │ │ iroh P2P   │ │ TOPLOC Signer  │
    │ base58(pubkey) │ │ hex(pubkey)│ │ Ed25519 sign() │
    │ "5EZ7Q..."     │ │ "a3b4c5.." │ │                │
    └────────────────┘ └────────────┘ └────────────────┘
```

**Storage format:** JSON array of 64 bytes (first 32 = secret, last 32 = public). Same format as `solana-keygen` output.

**Permissions:** 0600 on Unix (owner read/write only). Windows: logged warning to set permissions manually.

**Load priority:** `~/.samhati/identity.json` → `~/.config/solana/id.json` → `~/.local/share/samhati/wallet.json` → generate new.

**Error handling:** If HOME directory is unset, returns an error instead of silently writing to a wrong path.

---

## Distributed Inference Coordinator

**Files:** `crates/inference-coordinator/src/`

The inference coordinator splits a large model across multiple nodes (model parallelism). Each node runs a "shard" — a consecutive range of transformer layers.

### Shard Execution Flow

```
Prompt: "What is AI?"
  ↓
Coordinator tokenizes → [token_ids as f32]
  ↓
TensorFrame { shape: [1, seq_len], data: token_bytes, seq_offset: 0 }
  ↓
Shard 0 (layers 0-15, Node A) → hidden_states [1, seq_len, hidden_size]
  ↓
Shard 1 (layers 16-31, Node B) → next_token_id [1, 1]
  ↓
Decode loop: feed next_token back through all shards
  ↓
Repeat until EOS or max_tokens
  ↓
Final output: byte-level decode back to UTF-8
```

### RPC Messages

```rust
pub struct RpcRequest {
    pub session_id: String,       // unique per generation
    pub layer_start: u32,         // first layer for this shard
    pub layer_end: u32,           // one-past-last layer
    pub total_layers: u32,        // model total (detects first/last shard)
    pub max_tokens: u32,
    pub temperature: f32,
    pub tensor: TensorFrame,      // activation to forward
}

pub struct RpcResponse {
    pub tensor: TensorFrame,      // output activation
    pub error: Option<String>,
}
```

**Deserialization safety:** All `from_bytes()` methods use `bincode::Options::with_limit(256 MB)` to prevent OOM from crafted payloads.

### Fault Recovery (Replay)

If a shard node fails mid-generation, the coordinator replays cached activations to a replacement node:

1. Client maintains an `ActivationRingBuffer` (512 entries) of all activations sent
2. On failure, coordinator queries `SwarmRegistry` for an alternative peer with the same layer range
3. Sends `ReplayRequest` with all prior frames — the replacement rebuilds its KV cache
4. Normal inference continues from where it left off

---

## mesh-node HTTP API

**File:** `crates/mesh-node/src/api.rs`

### Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | No | Returns `{"status": "ok"}` |
| GET | `/v1/models` | Yes | List allowed models |
| POST | `/v1/chat/completions` | Yes | OpenAI-compatible chat completion |

**Authentication:** Bearer token or `x-api-key` header. Compared using constant-time comparison (`subtle::ConstantTimeEq`) to prevent timing attacks.

**Rate limiting:** Per-client token bucket keyed by actual TCP peer address (`ConnectInfo<SocketAddr>`), not the spoofable `X-Forwarded-For` header. Configurable via `--rate-limit-rps` and `--rate-limit-burst` flags. Stale client entries evicted after 10 minutes at 1000+ clients.

### Inference Backends

The API server supports three inference backends:

1. **Local execution** — runs llama.cpp binary with command-line arguments. User prompts are safely escaped (whitespace, quotes, single quotes, backslashes) and passed via `Command::new().args()` (no shell invocation).

2. **HTTP delegation** — forwards to an upstream OpenAI-compatible API with optional bearer auth.

3. **Swarm ranked** — fans out to P2P mesh nodes via iroh QUIC, runs full swarm consensus (BradleyTerry + ELO).

---

## Model System

**Files:** `crates/samhati-tui/src/app.rs`, `crates/samhati-tui/src/model_download.rs`

### Auto-Detection

On startup, Samhati detects available RAM via `sysinfo` and recommends models that fit:

| RAM | Recommended models |
|-----|-------------------|
| < 2 GB | 0.5B models only |
| 2-4 GB | Up to 1.7B |
| 4-8 GB | Up to 3-4B |
| 8-16 GB | Up to 7-8B |
| 16+ GB | Up to 14B |

### Available Models

21 models across 4 domains, all Q4_K_M GGUF from HuggingFace:

| Size | General | Code (1.5x SMTI) | Math (1.5x SMTI) | Reasoning (1.5x SMTI) |
|------|---------|-------------------|-------------------|----------------------|
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

### Download Flow

1. Model name → HuggingFace (repo, filename) mapping
2. Stream download from `https://huggingface.co/{repo}/resolve/main/{filename}`
3. Save to `~/.cache/samhati/models/{model_name}.gguf`
4. Progress callback updates the Models tab gauge
5. On complete → auto-start llama-server → announce to gossip

---

## Settlement Flow

**Files:** `crates/samhati-tui/src/settlement.rs`, `crates/samhati-tui/src/registry.rs`

After each swarm round, results are settled both locally and on-chain:

### Local Settlement

1. **Build payload** from `SwarmRoundResult`:
   - Extract participant Solana pubkeys
   - Compute ELO deltas
   - Capture TOPLOC proof hashes (BLAKE3 of proof bytes)
   - Map domain string to u64: Code→1, Math→2, Reasoning→3, General→0

2. **Save to disk** at `~/.samhati/pending_rounds/round_{id}.json`
   - Allows offline queuing if Solana is unreachable
   - Round IDs are monotonically increasing per node (`~/.samhati/round_counter`)

### On-Chain Settlement

3. **Submit to Solana** via `submit_round` instruction:
   - Creates `RoundAccount` PDA with all round data
   - Updates each participant's `NodeAccount` (ELO, rounds, wins, rewards)
   - Increments `ProtocolConfig` domain demand counters
   - Authority signature required (coordinator key)

### Domain Demand Display

The Dashboard tab reads domain counters from Solana and shows live demand:

```
Code      ████████████████████ 42%  ← run Coder model for 1.5x SMTI
Math      ████████             16%
Reasoning ████                 11%
General   ████████████         31%
```

Operators see what the network needs and choose specialist models accordingly.

---

## Security Hardening

| Layer | Protection | Implementation |
|-------|-----------|---------------|
| **API Auth** | Timing-attack resistant | `subtle::ConstantTimeEq` for token comparison |
| **Rate Limiting** | Per-client by real IP | `ConnectInfo<SocketAddr>` (not spoofable XFF) |
| **TOPLOC Proofs** | Node binding | `node_pubkey` field ties proof to specific node |
| **TOPLOC Proofs** | Freshness | 5-minute max age (configurable), prevents replay |
| **TOPLOC Proofs** | Key enforcement | Rejects when no public keys registered |
| **RPC Messages** | Size limit | 256 MB bincode deserialization cap |
| **Gossip Messages** | Size limit | 64 KB max before JSON parsing |
| **Identity File** | Restricted perms | 0600 on Unix, warning on Windows |
| **P2P Transport** | Encryption | QUIC/TLS 1.3 via iroh (all traffic encrypted) |
| **Local Exec** | Argument safety | Shell metacharacters escaped, `Command::args()` (no shell) |
| **Solana RPC** | Configurable | `SOLANA_RPC_URL` env var (defaults to devnet) |
| **Supply Chain** | Reproducible | Cargo.lock tracked in git |
| **Peer Validation** | On-chain check | Only Solana-registered nodes accepted into swarm |
| **SSRF Protection** | URL validation | Blocks private IPs, localhost, link-local in swarm |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User Layer                           │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ samhati-tui  │    │  app-tauri   │    │  SDK (Rust/  │  │
│  │ (terminal +  │    │  (desktop)   │    │  Python/TS)  │  │
│  │  browser)    │    │              │    │              │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
│         └───────────────────┼───────────────────┘          │
│                             │                               │
├─────────────────────────────┼───────────────────────────────┤
│                      Network Layer                          │
│                             │                               │
│  ┌──────────────────────────▼──────────────────────────┐   │
│  │                    mesh-node                         │   │
│  │  HTTP API (axum) ←→ iroh QUIC ←→ gossip protocol   │   │
│  └──────────────────────────┬──────────────────────────┘   │
│                             │                               │
├─────────────────────────────┼───────────────────────────────┤
│                    Inference Layer                           │
│                             │                               │
│  ┌──────────────────────────▼──────────────────────────┐   │
│  │              inference-coordinator                   │   │
│  │  shard planning → RPC dispatch → KV cache → failover│   │
│  └──────────────────────────┬──────────────────────────┘   │
│                             │                               │
│  ┌──────────────────────────▼──────────────────────────┐   │
│  │               llama-toploc (C++)                     │   │
│  │  llama.cpp + BLAKE3 activation hashing + proof API  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                    Consensus Layer                           │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │ samhati-swarm│  │samhati-ranking│  │  samhati-toploc  │ │
│  │ fan-out +    │  │ ELO + Bradley│  │  proof + verify + │ │
│  │ classify +   │  │ Terry + SMTI │  │  calibration     │ │
│  │ debate       │  │ rewards      │  │                   │ │
│  └──────────────┘  └──────────────┘  └──────────────────┘ │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                    Settlement Layer                          │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Solana (devnet)                          │  │
│  │  ProtocolConfig + NodeAccount + RoundAccount PDAs    │  │
│  │  register / calibrate / submit_round / slash / emit  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
Samhati/
├── crates/
│   ├── samhati-tui/                # Terminal + browser UI
│   │   └── src/
│   │       ├── main.rs             # Native entry (crossterm, tokio)
│   │       ├── web_main.rs         # Browser WASM entry (ratzilla)
│   │       ├── app.rs              # App state, 5 tabs, 21 models, RAM detection
│   │       ├── ui.rs               # Shared ratatui rendering (works both targets)
│   │       ├── swarm.rs            # Swarm: classify → fan-out → debate → rank → BT → ELO
│   │       ├── identity.rs         # Unified Ed25519 identity
│   │       ├── network.rs          # iroh P2P: gossip thread, peer discovery
│   │       ├── registry.rs         # Solana: fetch nodes, register, submit_round
│   │       ├── settlement.rs       # Round payloads, local persist, Solana settle
│   │       ├── node_runner.rs      # Spawns TOPLOC llama-server processes
│   │       ├── model_download.rs   # HuggingFace GGUF streaming download
│   │       ├── wallet.rs           # Solana devnet wallet (balance, airdrop, txs)
│   │       ├── api.rs              # HTTP client for inference API
│   │       ├── events.rs           # Keyboard event routing
│   │       └── tabs/               # Chat, Dashboard, Models, Wallet, Settings
│   │   └── index.html              # Trunk build config for WASM
│   ├── mesh-node/                  # Network node
│   │   └── src/
│   │       ├── api.rs              # HTTP API (axum): /health, /v1/models, /v1/chat/completions
│   │       ├── server.rs           # QUIC RPC server (accepts inference requests)
│   │       ├── inference.rs        # Local exec + HTTP delegation + streaming
│   │       ├── protocol.rs         # Gossip message types + parsing
│   │       ├── swarm_bridge.rs     # Bridges samhati-swarm to mesh-node
│   │       └── main.rs             # CLI args, node startup, gossip loop
│   ├── inference-coordinator/      # Distributed inference
│   │   └── src/
│   │       ├── coordinator.rs      # Shard orchestration + decode loop
│   │       ├── iroh_executor.rs    # QUIC RPC client + failover + replay
│   │       ├── rpc.rs              # Wire protocol (bincode, size-limited)
│   │       ├── swarm_planner.rs    # Layer-to-peer assignment
│   │       ├── kv_cache.rs         # Layer-local KV cache with TTL
│   │       └── tensor_frame.rs     # Typed tensor serialization (f32/f16)
│   ├── samhati-toploc/             # TOPLOC proof system
│   │   └── src/
│   │       ├── proof.rs            # ToplocProof struct + serialization
│   │       ├── prover.rs           # Proof generation (BLAKE3 + Ed25519)
│   │       ├── verifier.rs         # 8-step verification pipeline
│   │       ├── calibration.rs      # New-node calibration rounds
│   │       └── types.rs            # Constants (chunk size, top-K, model hash)
│   ├── samhati-swarm/              # Swarm consensus
│   │   └── src/
│   │       ├── coordinator.rs      # SwarmCoordinator: fan-out, debate, rank, aggregate
│   │       ├── classifier.rs       # Complexity classifier (Easy/Medium/Hard)
│   │       ├── registry.rs         # ModelRegistry + NodeInfo
│   │       └── types.rs            # SwarmConfig, NodeResponse, PairwiseRanking
│   ├── samhati-ranking/            # ELO + BradleyTerry
│   │   └── src/
│   │       ├── elo.rs              # EloStore: update, slash, domain tracking
│   │       ├── bradley_terry.rs    # MM algorithm for MLE ranking
│   │       ├── rewards.rs          # SMTI reward distribution
│   │       └── types.rs            # RoundOutcome, PairwiseComparison
│   ├── proximity-router/           # Peer selection (std-only, zero external deps)
│   ├── cluster-manager/            # Cluster constraints
│   ├── model-config/               # Model config parsing
│   └── shard-store/                # Content-addressed weight cache (BLAKE3)
├── llama-toploc/                   # Patched llama.cpp
│   ├── toploc-proof.h              # BLAKE3 activation hashing (~140 lines C++)
│   ├── toploc.patch                # Diff to apply on llama.cpp
│   └── build.sh                    # One command: clone → patch → build
├── programs/
│   └── samhati-protocol/           # Solana Anchor program
├── sdk/
│   ├── rust/                       # Rust SDK (reqwest, async streaming)
│   ├── python/                     # Python SDK (httpx, sync + async)
│   └── typescript/                 # TypeScript SDK (native fetch)
├── app-tauri/                      # Tauri v2 desktop app
├── pipeline/                       # Self-evolving training (Python)
├── install.sh                      # One-command install
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
- Complexity + domain classifiers with adaptive node count
- ELO tracking + BradleyTerry ranking + SMTI rewards
- Unified identity (one Ed25519 key for Solana + iroh + TOPLOC)
- iroh P2P + gossip + NAT traversal + auto-mesh via Solana
- TOPLOC proof system with node binding, freshness, calibration
- Solana Anchor program (5 instructions + domain counters)
- Security hardening (constant-time auth, per-IP rate limit, gossip bounds, RPC limits)
- Rust, Python, TypeScript SDKs + Tauri desktop app
- One-command install script

### Next

- [ ] SMTI token mint on devnet
- [ ] EAGLE-3 speculative decoding (4-6x speedup)
- [ ] Trained difficulty classifier (RouteLLM-style)
- [ ] Full WASM browser build (gate remaining native deps)
- [ ] React Native mobile app
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
