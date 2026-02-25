# Samhati — Decentralized Peer-to-Peer LLM Inference Mesh

**Samhati** is a Rust-native, model-agnostic framework for running large language models
collaboratively across a peer-to-peer mesh network.  Each node contributes GPU/CPU memory and
compute; together they host models that no single machine could fit alone.  A proximity-aware
gossip protocol discovers peers, measures real network latency, and dynamically forms the
optimal cluster for each model request — with mid-token failover if a peer drops.

> **Status:** Active development — networking, shard planning, swarm routing, SSE streaming, and
> simulated tensor inference are all working.  Full LlamaShard forward-pass validation against
> real checkpoints is in progress.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Feature Status](#feature-status)
3. [Comparison with Prior Art](#comparison-with-prior-art)
4. [Crates](#crates)
5. [Quick Start](#quick-start)
6. [CLI Reference](#cli-reference)
7. [Cargo Features](#cargo-features)
8. [Configuration](#configuration)
9. [Roadmap](#roadmap)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Samhati Node (mesh-node)                      │
│                                                                      │
│  ┌─────────────┐   ┌─────────────────┐   ┌────────────────────────┐ │
│  │  Gossip /   │   │  SwarmRegistry  │   │  Inference             │ │
│  │  Discovery  │──▶│  (reputation +  │──▶│  Coordinator           │ │
│  │  (iroh)     │   │   TTL eviction) │   │  (shard planner)       │ │
│  └─────────────┘   └─────────────────┘   └────────┬───────────────┘ │
│                                                    │                 │
│  ┌─────────────┐   ┌─────────────────┐   ┌────────▼───────────────┐ │
│  │  Proximity  │   │  Shard Store    │   │  Burn Backend          │ │
│  │  Router     │   │  (content-addr) │   │  (NdArray / WGPU)      │ │
│  └─────────────┘   └─────────────────┘   └────────────────────────┘ │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  OpenAI-Compatible REST API  (GET /v1/models, POST /v1/chat) │   │
│  │  Streaming SSE  ·  Auth  ·  Rate limiting                    │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
                  ▲                         ▲
     iroh QUIC    │   gossip/DHT            │   iroh QUIC
                  │                         │
           ┌──────┴──────┐           ┌──────┴──────┐
           │   Peer A    │           │   Peer B    │
           │  layers 0-15│           │ layers 16-31│
           └─────────────┘           └─────────────┘
```

Inference is **pipeline-parallel**: transformer layers are split across peers.  Each peer runs
its slice as an iroh QUIC server, passing activation tensors forward as size-prefixed bincode
frames.  The `SwarmRegistry` tracks EWMA reputation scores for every peer; if a peer drops
mid-generation the `IrohDistributedExecutor` automatically retries with the next-best candidate.

---

## Feature Status

### Infrastructure

| Feature | Status | Notes |
|---------|--------|-------|
| Gossip peer discovery | ✅ Working | iroh-gossip topics, ping/pong RTT probing |
| Capability announcement | ✅ Working | VRAM, bandwidth, GPU score, layer ranges, uptime |
| Layer-range advertisement | ✅ Working | `--layer-start/end/total-layers/model-name` flags |
| Proximity-aware peer scoring | ✅ Working | Weighted: RTT 55%, reliability 20%, bandwidth 15%, GPU 10% |
| SwarmRegistry (soft-state) | ✅ Working | TTL eviction, EWMA reputation α=0.15 |
| Dynamic SwarmPlanner | ✅ Working | Per-range peer selection from live registry |
| Static RoundRobinPlanner | ✅ Working | Even layer distribution across fixed peer list |
| VRAM / layer estimation | ✅ Working | Any Llama-style config (hidden, layers, kv_heads) |
| Cluster constraint solver | ✅ Working | VRAM + bandwidth + node-count constraints |
| State persistence | ✅ Working | JSON peer DB + model registry |
| Model fallback cascade | ✅ Working | Try smaller model if primary doesn't fit |

### Inference Pipeline

| Feature | Status | Notes |
|---------|--------|-------|
| Echo executor (debug) | ✅ Working | Annotated pass-through for routing smoke tests |
| Burn simulate mode | ✅ Working | Arithmetic token simulation, proves pipeline end-to-end |
| Burn MLP mode | ✅ Working | Random-weight matmul + tanh, real NdArray tensor ops |
| Burn weights/forward/embed | ✅ Working | Load real safetensors, compute against loaded weights |
| Autoregressive `generate()` | ✅ Working | Byte tokenizer, prefill + decode loop, EOS detection |
| Tensor frame wire format | ✅ Working | LE f32 bytes + shape + seq_offset, bincode serialized |
| iroh QUIC RPC | ✅ Working | Size-prefixed bincode over QUIC streams |
| Mid-token failover | ✅ Working | Reputation hit → next-best peer → retry up to N times |
| `--executor swarm` | ✅ Working | Gossip discovery → SwarmPlanner → distributed pipeline |
| KV cache (per-session) | ✅ Working | LayerKv chunks, TTL eviction, thread-safe |
| LlamaShard weight loading | ✅ Working | safetensors → Burn Linear/RmsNorm via Record API |
| LlamaShard full forward pass | ⚠️ Partial | Attention/MLP/RoPE compile; numerical correctness vs HF not yet validated |
| WGPU GPU backend | ⚠️ Partial | Compiles; `LlamaShard::load_wgpu()` available; benchmarked |

### API Layer

| Feature | Status | Notes |
|---------|--------|-------|
| `GET /health` | ✅ Working | |
| `GET /v1/models` | ✅ Working | Lists allow-listed models |
| `POST /v1/chat/completions` (non-stream) | ✅ Working | HTTP proxy or local-exec backend |
| `POST /v1/chat/completions` (`stream: true`) | ✅ Working | SSE chunks in OpenAI format; true proxy for HTTP backend, word-stream for local-exec |
| Bearer token + x-api-key auth | ✅ Working | |
| Token-bucket rate limiting | ✅ Working | Configurable RPS + burst |
| API → distributed pipeline | ❌ Not wired | `/v1/chat/completions` calls http/local-exec only; iroh/swarm executor path pending |

### Weight Management

| Feature | Status | Notes |
|---------|--------|-------|
| Shard store (content-addressed) | ✅ Working | blake3 hash, add/get/find/evict/list |
| Shard store used in inference | ❌ Disconnected | Library complete; inference pipeline fetches weights directly from disk paths |
| iroh-blobs P2P weight transfer | 📋 Planned | |

---

## Comparison with Prior Art

| Dimension | Samhati | Reference systems |
|-----------|---------|-------------------|
| **Implementation language** | Rust | Python |
| **P2P transport** | iroh (QUIC, hole-punching) | libp2p / gRPC |
| **Inference backend** | Burn 0.20 (NdArray + WGPU) | PyTorch / CUDA |
| **Peer discovery** | iroh-gossip DHT + ping/pong RTT | DHT |
| **Cluster selection** | RTT + VRAM + reliability scoring | VRAM-only |
| **Reputation system** | EWMA per-peer, updated on every RPC | Partial |
| **Mid-token failover** | ✅ Automatic retry with next-best peer | ❌ |
| **Streaming (SSE)** | ✅ True proxy or simulated word-stream | ✅ OpenAI-compatible |
| **HTTP API auth + rate limiting** | ✅ Built-in | ❌ External |
| **Multi-model registry** | ✅ JSON registry + gossip broadcast | ❌ Single model per network |
| **KV cache** | ✅ Per-session, TTL eviction | ✅ Per-session |
| **Quantization** | 📋 4/5/8-bit planned | ✅ 8-bit (bitsandbytes) |
| **Tensor parallelism** | 📋 Planned | ❌ |
| **MoE expert placement** | 📋 Planned | ❌ |
| **Speculative decoding** | 📋 Planned | ❌ |
| **Memory estimation** | ✅ Weights + KV + runtime overhead | Partial |
| **Production hardening** | 🚧 In progress | ✅ Battle-tested |
| **Latency target** | ≤ 2 s first token (regional) | ~5–10 s observed |

---

## Crates

| Crate | Description |
|-------|-------------|
| `mesh-node` | Main binary — CLI, gossip node, API server, distributed inference driver |
| `inference-coordinator` | Shard planner, coordinator, Burn LLM execution, KV cache, RPC wire format, SwarmPlanner |
| `proximity-router` | Peer scoring (RTT/bandwidth/reliability/GPU) + SwarmRegistry with EWMA reputation |
| `cluster-manager` | Constraint solver: given peers and model, pick smallest feasible cluster |
| `model-config` | Parse any safetensors config.json; estimate VRAM (weights + KV + runtime) |
| `shard-store` | Content-addressed on-disk cache for weight shards (blake3 hashes) |

---

## Quick Start

### Prerequisites

- Rust 1.82+ (`rustup update stable`)
- For GPU support: a WGPU-compatible GPU (Vulkan / Metal / DX12)

```bash
git clone https://github.com/mrunalpendem123/Samhati.git
cd Samhati
cargo build                    # CPU / simulation only
cargo build --features burn    # + Burn NdArray + WGPU GPU backend
```

### Run a Two-Node Gossip Mesh

Terminal 1 — first node:
```bash
cargo run -p mesh-node -- gossip \
  --topic 1717171717171717171717171717171717171717171717171717171717171717
```

It prints a `node_id`. Copy it. Terminal 2 — second node:
```bash
cargo run -p mesh-node -- gossip \
  --topic 1717171717171717171717171717171717171717171717171717171717171717 \
  --bootstrap <node_id_from_terminal_1>
```

Both nodes exchange heartbeats and capability announcements. Add `--rtt-probe` on both to
measure actual network latency.

### Open-Mesh Swarm: Two Inference Nodes + Coordinator

**Node A** — hosts layers 0–16 of `llama` model:
```bash
cargo run -p mesh-node -- gossip \
  --topic <64-hex> \
  --layer-start 0 --layer-end 16 --total-layers 32 --model-name llama \
  --free-vram 24
```

**Node B** — hosts layers 16–32:
```bash
cargo run -p mesh-node -- gossip \
  --topic <64-hex> \
  --bootstrap <node_a_id> \
  --layer-start 16 --layer-end 32 --total-layers 32 --model-name llama \
  --free-vram 24
```

**Coordinator** — discovers both nodes and runs the full model:
```bash
cargo run -p mesh-node -- dist-run \
  --executor swarm \
  --topic <64-hex> \
  --bootstrap <node_a_id> \
  --discover-secs 5 \
  --models models.json --model llama \
  --layers-per-shard 16 \
  --input "Hello, world"
```

### Run Simulated Distributed Inference

```bash
# Echo executor (no tensor ops — routing smoke-test)
cargo run -p mesh-node -- dist-run \
  --peers peer-a,peer-b \
  --config sample-config.json --params-b 7 \
  --input "Hello, world"

# Burn simulate mode (proves the full shard pipeline)
cargo run -p mesh-node --features burn -- dist-run \
  --peers peer-a,peer-b \
  --config sample-config.json --params-b 7 \
  --executor burn --model-mode simulate \
  --input "Hello, world"

# Burn MLP mode (real NdArray tensor ops: matmul + tanh)
cargo run -p mesh-node --features burn -- dist-run \
  --peers peer-a,peer-b \
  --config sample-config.json --params-b 7 \
  --executor burn --model-mode mlp --model-hidden 64 \
  --input "Hello, world"
```

### Start the API Server

```bash
# Proxy to any OpenAI-compatible upstream with SSE streaming
cargo run -p mesh-node -- api \
  --bind 127.0.0.1:8000 \
  --infer-base http://127.0.0.1:8001/v1 \
  --default-model llama-3-8b \
  --api-auth my-secret-key \
  --rate-limit-rps 10

# Test streaming
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3-8b","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

---

## CLI Reference

```
mesh-node <COMMAND> [OPTIONS]
```

### `gossip` — Run a gossip mesh node

Announces capabilities, discovers peers, measures RTT, performs cluster selection.

```bash
cargo run -p mesh-node -- gossip \
  --topic <64-hex>                           # shared topic ID (required)
  [--bootstrap <endpoint_id> ...]            # seed peers
  [--config <path>] [--params-b <f64>]       # model to serve
  [--models <models.json>]                   # multi-model registry
  [--model-name <name>]                      # active model name
  [--fallback-model name:path:params_b]      # fallback chain
  # Layer-range advertisement (for --executor swarm):
  [--layer-start <n>] [--layer-end <n>]      # layer range this node serves
  [--total-layers <n>]                       # full model depth
  # Hardware capabilities:
  [--free-vram <GB>]                         # override VRAM auto-detect
  [--reserve-gb 4] [--safety-factor 0.7]
  [--bandwidth 800] [--reliability 0.95] [--gpu-score 0.8]
  [--rtt-ms 20] [--kv-bits 8] [--context 8192] [--quant 4]
  # Cluster constraints:
  [--min-nodes 2] [--max-nodes 8] [--min-bw 500]
  # Peer health:
  [--rtt-probe] [--rtt-timeout-ms 3000]
  [--peer-ttl-ms 15000] [--peer-fail-threshold 3]
  [--peer-cooldown-ms 30000] [--cluster-stickiness-ms 15000]
  [--load-penalty 0.05] [--load-decay 0.9]
  [--state-dir ./state]
  [--role inference|routing|cache]
  [--interval <secs>]
```

### `serve` — Start an iroh QUIC inference server

Binds an iroh endpoint, optionally loads a real LlamaShard, and serves inference RPCs from
other nodes.  Announces its NodeId on startup — pass that to `dist-run --peers`.

```bash
cargo run -p mesh-node --features burn -- serve \
  --model-path /path/to/model.safetensors \
  --layer-start 0 --layer-end 16 --total-layers 32 \
  --hidden 4096 --intermediate 11008 --vocab 32000 \
  --heads 32 --kv-heads 8 --rope-theta 500000 \
  [--kv-ttl-secs 300]
```

Without `--model-path` the node runs a `MockInferenceServer` (echo with metadata).

### `dist-run` — Execute a distributed inference pass

```bash
cargo run -p mesh-node [--features burn] -- dist-run \
  --peers peer-a,peer-b \
  [--models <path> --model <name>] \
  [--config <path> --params-b <f64>] \
  [--input "prompt text"] \
  [--max-tokens 64] [--temperature 0.7] \
  [--executor echo|burn|iroh|swarm] \
  [--max-retries 2]
```

**Executor modes:**

| `--executor` | Description |
|---|---|
| `echo` | No-op: returns the input annotated with shard info |
| `burn` | Local Burn NdArray simulation (requires `--features burn`) |
| `iroh` | Real distributed pass over iroh QUIC to `--peers` |
| `swarm` | Discovers peers via gossip, uses SwarmPlanner for dynamic routing |

**`burn` executor — `--model-mode` options:**

| Mode | What it does |
|---|---|
| `simulate` | Adds layer index to a scalar — proves pipeline end-to-end |
| `mlp` | Random-weight MLP: matmul + tanh over input bytes |
| `weights` | Loads a named tensor from safetensors, reports shape + checksum |
| `forward` | Matrix-vector multiply with a loaded weight tensor |
| `embed` | Embedding lookup: maps input bytes to token IDs, sums rows |

**`iroh` executor extra flags:**

```bash
--max-retries 2          # failover attempts on peer error
```

**`swarm` executor extra flags:**

```bash
--topic <64-hex>         # gossip topic to join
--bootstrap <id>         # at least one known peer
--discover-secs 5        # how long to collect announcements before planning
--layers-per-shard 16    # target layers per shard slice
--max-retries 2          # failover attempts on peer error
```

### `dist-plan` — Print a layer shard plan

```bash
cargo run -p mesh-node -- dist-plan \
  --peers peer-a,peer-b,peer-c \
  [--models <path> --model <name>] \
  [--config <path> --params-b <f64>] \
  [--min-layers-per-peer 1]
```

### `estimate` — Estimate VRAM requirements

```bash
cargo run -p mesh-node -- estimate \
  --config <path> --params-b <f64> \
  [--quant 4] [--seq 8192] [--batch 1] [--nodes 4] \
  [--kv-bytes 1] [--overhead 1.15] [--runtime-overhead 0.15]
```

### `capacity` — Find minimum nodes needed

```bash
cargo run -p mesh-node -- capacity \
  --free-vram <GB> \
  [--models <path> --model <name>] \
  [--config <path> --params-b <f64>] \
  [--max-nodes 8] [--seq 4096] [--quant 4] [--kv-bytes 1]
```

### `api` — Start the OpenAI-compatible REST API server

```bash
cargo run -p mesh-node -- api \
  --bind 127.0.0.1:8000 \
  [--infer-base http://upstream/v1]   # proxy backend
  [--infer-key sk-...]
  [--default-model <name>]
  [--local-bin /path/to/llama-cli \   # local-exec backend
   --local-args "-m {model} -p {prompt} -n {max_tokens} --temp {temperature}" \
   --local-model /path/to/model.gguf]
  [--api-auth <secret-key>]
  [--allow-model <name>]              # whitelist (repeatable)
  [--rate-limit-rps 5] [--rate-limit-burst 10]
```

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Node health check |
| `GET` | `/v1/models` | List allow-listed models |
| `POST` | `/v1/chat/completions` | Chat inference — supports `"stream": true` (SSE) |

**SSE streaming** — when `"stream": true`:
- **HTTP proxy backend**: content deltas are forwarded token-by-token from upstream as they arrive
- **Local-exec backend**: full response is simulated word-by-word as SSE chunks

### `simulate` — Cluster selection simulation

```bash
cargo run -p mesh-node -- simulate \
  --config <path> --params-b <f64> \
  [--peers peers.json] [--rtt-samples rtt.json] \
  [--min-nodes 2] [--max-nodes 6] [--min-bw 500]
```

### `infer-http` — Query an OpenAI-compatible endpoint directly

```bash
cargo run -p mesh-node -- infer-http \
  --api-base http://127.0.0.1:8000/v1 \
  --model <name> --prompt "Your prompt" \
  [--system "System message"] \
  [--max-tokens 128] [--temperature 0.7] [--api-key sk-...]
```

### `infer-local` — Run a local inference binary

Template variables: `{model}`, `{prompt}`, `{max_tokens}`, `{temperature}`, `{context}`.

```bash
cargo run -p mesh-node -- infer-local \
  --local-bin /path/to/llama-cli \
  --local-args "-m {model} -p {prompt} -n {max_tokens} --temp {temperature} --ctx-size {context}" \
  --local-model /path/to/model.gguf \
  --prompt "Hello"
```

---

## Cargo Features

| Feature | Description |
|---------|-------------|
| *(none)* | Base build: networking, planning, echo executor, API server, SSE streaming |
| `burn` | Enables Burn 0.20 (NdArray + WGPU), `LlamaShard`, `KvCacheStore`, real tensor ops |

```bash
cargo build                    # no ML deps
cargo build --features burn    # full Burn stack
cargo test                     # run all unit tests (11 tests)
cargo test --features burn     # run tests with Burn backend
```

---

## Configuration

### Model config (`config.json`)

Standard HuggingFace `config.json` format — any Llama-style architecture:

```json
{
  "model_type": "llama",
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "hidden_size": 4096,
  "intermediate_size": 11008,
  "vocab_size": 32000,
  "rope_theta": 500000.0,
  "rms_norm_eps": 1e-5
}
```

Pass it with `--config config.json --params-b 7` (7 = billions of parameters).

### Multi-model registry (`models.json`)

```json
[
  {
    "name": "llama-3-8b",
    "config_path": "configs/llama-3-8b.json",
    "params_b": 8.0,
    "quant_bits": 4,
    "kv_bits": 8,
    "max_context": 8192
  },
  {
    "name": "mistral-7b",
    "config_path": "configs/mistral-7b.json",
    "params_b": 7.0,
    "quant_bits": 4
  }
]
```

Use with `--models models.json --model llama-3-8b`.

### Gossip node with layer advertisement

```bash
cargo run -p mesh-node -- gossip \
  --topic <64-hex> \
  --models models.json \
  --model-name llama-3-8b \
  --layer-start 0 --layer-end 16 --total-layers 32 \
  --free-vram 24 \
  --min-nodes 2 --max-nodes 8 \
  --rtt-probe \
  --state-dir ./state
```

---

## Roadmap

### Phase 1 — Foundation ✅

- [x] iroh QUIC transport + gossip peer discovery
- [x] Capability announcement (VRAM, RTT, bandwidth, GPU score)
- [x] Proximity-aware cluster selection
- [x] VRAM estimation for any model config
- [x] Round-robin shard planner
- [x] Content-addressed shard store (blake3)
- [x] OpenAI-compatible HTTP API with auth and rate limiting
- [x] State persistence
- [x] Burn 0.20 integration (NdArray + WGPU, model-agnostic)

### Phase 2 — Open-Mesh Swarm ✅

- [x] SwarmRegistry: soft-state peer store with TTL eviction + EWMA reputation
- [x] Dynamic SwarmPlanner: per-range peer selection from live registry
- [x] Layer-range advertisement in gossip (`--layer-start/end/total-layers/model-name`)
- [x] `--executor swarm` with full gossip discovery → plan → execute flow
- [x] Mid-token failover: reputation hit → retry with next-best peer
- [x] `serve` command: iroh QUIC server with optional LlamaShard weight loading
- [x] Autoregressive `generate()` pipeline (prefill + decode loop)
- [x] SSE streaming (`stream: true`) — true proxy for HTTP backend, simulated for local-exec

### Phase 3 — LlamaShard Validation 🚧

- [ ] End-to-end numerical correctness vs HuggingFace on Llama-3 / Mistral
- [ ] GQA `repeat_kv` in attention forward pass
- [ ] RoPE rotate-half integration test
- [ ] KV cache handoff across shard boundaries over the wire
- [ ] API → distributed tensor pipeline (`/v1/chat/completions` via iroh/swarm executor)
- [ ] ShardStore integration into inference pipeline (P2P weight fetch via blake3 hash)

### Phase 4 — Performance & Scale 📋

- [ ] 4/5/8-bit quantized weight loading (GGUF + safetensors)
- [ ] Tensor parallelism (column/row parallel linear layers)
- [ ] Speculative decoding across shards
- [ ] MoE expert placement (Mixtral, DeepSeek)
- [ ] iroh-blobs weight distribution (P2P checkpoint transfer)
- [ ] Adaptive context reduction under memory pressure
- [ ] Continuous batching

### Phase 5 — Production 📋

- [ ] Reputation persistence across restarts
- [ ] Encrypted inference (homomorphic / TEE investigation)
- [ ] Web dashboard (peer topology, cluster health, reputation heatmap)
- [ ] Docker / container deployment
- [ ] Incentive layer (optional token-gated access)

---

## Project Structure

```
Samhati/
├── crates/
│   ├── mesh-node/                  # Main binary (CLI + server)
│   │   └── src/
│   │       ├── main.rs             # All CLI commands
│   │       ├── server.rs           # iroh QUIC inference server (real + mock)
│   │       ├── protocol.rs         # Gossip message types + CapabilityPayload
│   │       ├── api.rs              # OpenAI REST API (SSE streaming, auth, rate limit)
│   │       ├── inference.rs        # HTTP proxy + local-exec backends + SSE stream helpers
│   │       ├── scheduler.rs        # Cluster candidate scoring
│   │       ├── state.rs            # Peer / session state
│   │       └── persist.rs          # JSON state persistence
│   ├── inference-coordinator/      # Shard planning + tensor execution
│   │   └── src/
│   │       ├── coordinator.rs      # Orchestrates multi-shard pass; generate() loop
│   │       ├── model_runner.rs     # ModelShardRunner (5 Burn simulation modes)
│   │       ├── llm_shard.rs        # LlamaShard<B> (real weight loading + forward)
│   │       ├── kv_cache.rs         # KvCacheStore<B> (per-session KV, TTL eviction)
│   │       ├── tensor_frame.rs     # Wire format for activation tensors
│   │       ├── iroh_executor.rs    # QUIC-based executor with mid-token failover
│   │       ├── swarm_planner.rs    # SwarmPlanner (dynamic peer selection)
│   │       ├── plan.rs             # RoundRobinPlanner + ShardPlan
│   │       └── rpc.rs              # RPC request/response types
│   ├── proximity-router/           # Peer scoring and ranking
│   │   └── src/
│   │       ├── lib.rs              # ProximityRouter, scoring weights, rank_peers()
│   │       └── swarm.rs            # SwarmRegistry, SwarmPeer, LayerRange, reputation
│   ├── cluster-manager/            # VRAM-aware cluster constraint solver
│   ├── model-config/               # Config parsing + VRAM estimation
│   └── shard-store/                # Content-addressed weight shard cache
├── sample-config.json              # Example model config
└── sample-peers.json               # Example peer list
```

---

## Contributing

1. Fork and clone the repo
2. `cargo build` — must compile clean
3. `cargo build --features burn` — must also compile clean
4. `cargo test` — all 11 tests must pass
5. Open a PR with a clear description of the change

Please keep PRs focused — one feature or fix per PR.

---

## License

MIT
