# Samhati — Decentralized Peer-to-Peer LLM Inference Mesh

**Samhati** is a Rust-native, model-agnostic framework for running large language models
collaboratively across a peer-to-peer mesh network.  Each node contributes GPU/CPU memory and
compute; together they host models that no single machine could fit alone.  A proximity-aware
gossip protocol discovers peers, measures real network latency, and dynamically forms the
smallest feasible cluster for each model request.

> **Status:** Active development — core networking, planning, and shard execution are working.
> Real distributed tensor inference and production hardening are in progress.

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
│  │  Gossip /   │   │  Cluster        │   │  Inference             │ │
│  │  Discovery  │──▶│  Manager        │──▶│  Coordinator           │ │
│  │  (iroh)     │   │  (VRAM + RTT)   │   │  (shard planner)       │ │
│  └─────────────┘   └─────────────────┘   └────────┬───────────────┘ │
│                                                    │                 │
│  ┌─────────────┐   ┌─────────────────┐   ┌────────▼───────────────┐ │
│  │  Proximity  │   │  Shard Store    │   │  Burn Backend          │ │
│  │  Router     │   │  (content-addr) │   │  (NdArray / WGPU)      │ │
│  └─────────────┘   └─────────────────┘   └────────────────────────┘ │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  OpenAI-Compatible REST API  (GET /v1/models, POST /v1/chat) │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
                  ▲                         ▲
     iroh QUIC    │   gossip/DHT            │   iroh QUIC
                  │                         │
           ┌──────┴──────┐           ┌──────┴──────┐
           │   Peer A    │           │   Peer B    │
           │  layers 0-9 │           │ layers 10-19│
           └─────────────┘           └─────────────┘
```

Inference is **pipeline-parallel**: the model's transformer layers are split across peers.
Each peer runs its slice as an iroh QUIC server, passing activation tensors forward
(and KV-cache backward) as size-prefixed bincode frames over a single bi-directional QUIC stream.

---

## Feature Status

| Feature | Status | Notes |
|---------|--------|-------|
| **Gossip peer discovery** | ✅ Working | iroh-gossip, ping/pong RTT probing |
| **Capability announcement** | ✅ Working | VRAM, bandwidth, GPU score, quant support |
| **Proximity-aware cluster selection** | ✅ Working | RTT + VRAM + reliability scoring |
| **VRAM / layer estimation** | ✅ Working | Any model config (hidden_size, layers, kv_heads) |
| **Shard plan generation** | ✅ Working | Round-robin layer assignment |
| **Shard store (content-addressed)** | ✅ Working | blake3, iroh-blobs compatible |
| **OpenAI-compatible HTTP API** | ✅ Working | `/health`, `/v1/models`, `/v1/chat/completions` |
| **API auth + rate limiting** | ✅ Working | Bearer token, per-IP RPS cap |
| **Model registry (multi-model)** | ✅ Working | JSON registry, gossip broadcast |
| **Simulated shard execution** | ✅ Working | `--executor burn --model-mode simulate` |
| **Burn NdArray tensor ops** | ✅ Working | MLP, matmul, tanh, softmax |
| **LlamaShard weight loading** | ✅ Working | Any safetensors checkpoint |
| **KV cache (per-session)** | ✅ Working | Auto-eviction by TTL |
| **Model-agnostic config** | ✅ Working | Any Llama-style architecture |
| **State persistence** | ✅ Working | JSON state + peer/model DB |
| **Model fallback cascade** | ✅ Working | Try smaller model if primary doesn't fit |
| **WGPU GPU backend** | 🔧 Integrated | Compiles; thread-safety wrapper in place |
| **Real distributed tensor pass** | 🚧 In Progress | QUIC streaming + activation hand-off |
| **KV cache over wire** | 🚧 In Progress | Encode/decode across shard boundary |
| **Grouped-query attention (GQA)** | 🚧 In Progress | GQA repeat_kv scaffolded |
| **Speculative decoding** | 📋 Planned | Draft-verify across shards |
| **Tensor parallelism (intra-layer)** | 📋 Planned | Column/row parallel linear |
| **MoE expert placement** | 📋 Planned | Expert-parallel for Mixtral-style |
| **8/4-bit weight quantization** | 📋 Planned | GGUF / bitsandbytes parity |
| **Adaptive context reduction** | 📋 Planned | Degrade gracefully under pressure |
| **Fault-tolerant rerouting** | 📋 Planned | Re-plan on peer drop mid-generation |
| **iroh-blobs shard transfer** | 📋 Planned | P2P weight distribution |
| **Streaming token output (SSE)** | 📋 Planned | HTTP chunked + server-sent events |

---

## Comparison with Prior Art

The table below compares Samhati against the reference class of decentralized collaborative
inference systems.

| Dimension | Samhati (this project) | Reference systems |
|-----------|------------------------|-------------------|
| **Implementation language** | Rust | Python |
| **P2P transport** | iroh (QUIC, hole-punching) | libp2p / gRPC |
| **Inference backend** | Burn 0.20 (NdArray + WGPU) | PyTorch / CUDA |
| **Parallelism strategy** | Pipeline-parallel (per-layer shards) | Pipeline-parallel |
| **Tensor parallelism** | 📋 Planned | ❌ Not supported |
| **MoE / expert parallel** | 📋 Planned | ❌ Not supported |
| **Model support** | Any safetensors (Llama, Mistral, Falcon…) | Specific model families |
| **Multi-model registry** | ✅ JSON registry + gossip broadcast | ❌ Single model per network |
| **Cluster selection** | ✅ RTT + VRAM + reliability scoring | Partially (VRAM-only) |
| **KV cache** | ✅ Per-session, TTL eviction | ✅ Per-session |
| **Quantization** | 📋 4/5/8-bit planned | ✅ 8-bit (bitsandbytes) |
| **Speculative decoding** | 📋 Planned | ❌ |
| **Fault tolerance** | 📋 Re-planning on peer drop | Partial |
| **Peer discovery** | ✅ iroh-gossip DHT + ping/pong RTT | ✅ DHT |
| **HTTP API** | ✅ OpenAI-compatible | ✅ OpenAI-compatible |
| **API auth + rate limiting** | ✅ Built-in | ❌ External |
| **Memory estimation** | ✅ Weights + KV + runtime overhead | Partial |
| **State persistence** | ✅ JSON (peer DB, model registry) | ✅ |
| **Latency target** | ≤ 2 s first token (regional) | ~5–10 s observed |
| **Throughput target** | ≥ 20 tok/s regional | ~6 tok/s observed |
| **Production hardening** | 🚧 In progress | ✅ Battle-tested |

**Key differentiators we are building toward:**
- Full Rust stack — no GIL, lower memory overhead, better concurrency
- WGPU backend — GPU inference on macOS/Linux/Windows without CUDA lock-in
- Tensor + pipeline parallelism combined
- MoE expert placement for Mixtral-class models
- Sub-2-second first-token latency via proximity-aware routing

---

## Crates

| Crate | Description |
|-------|-------------|
| `mesh-node` | Main binary — CLI, gossip node, API server, distributed inference driver |
| `inference-coordinator` | Shard planner, coordinator, Burn-based LLM execution, KV cache, RPC wire format |
| `proximity-router` | Peer scoring: RTT, bandwidth, reliability, GPU score → ranked peer list |
| `cluster-manager` | Constraint solver: given peers and model, pick smallest feasible cluster |
| `model-config` | Parse any safetensors config.json; estimate VRAM (weights + KV + runtime) |
| `shard-store` | Content-addressed on-disk cache for weight shards (blake3 hashes) |
| `api` | OpenAI-compatible HTTP layer (axum) |
| `bench` | Benchmarking utilities |

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
measure actual network latency between them.

### Run Simulated Distributed Inference

```bash
# Echo executor (no tensor ops — just routing smoke-test)
cargo run -p mesh-node -- dist-run \
  --peers peer-a,peer-b \
  --config sample-config.json --params-b 7 \
  --input "Hello, world"

# Burn simulate mode (adds layer indices as f32 — proves the shard pipeline)
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

### Load Real Weights and Inspect a Tensor

```bash
cargo run -p mesh-node --features burn -- dist-run \
  --peers peer-a \
  --config sample-config.json --params-b 7 \
  --executor burn --model-mode weights \
  --model-path /path/to/model.safetensors \
  --model-tensor "model.embed_tokens.weight"
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
  --topic <64-hex>                        # shared topic ID
  [--bootstrap <endpoint_id> ...]         # seed peers
  [--config <path>] [--params-b <f64>]   # model to serve
  [--models <models.json>]               # multi-model registry
  [--fallback-model name:path:params_b]  # fallback chain
  [--free-vram <GB>]                     # override VRAM auto-detect
  [--reserve-gb 4] [--safety-factor 0.7]
  [--bandwidth 800] [--reliability 0.95] [--gpu-score 0.8]
  [--rtt-ms 20] [--kv-bits 8] [--context 8192] [--quant 4]
  [--min-nodes 2] [--max-nodes 8] [--min-bw 500]
  [--rtt-probe] [--rtt-timeout-ms 3000]
  [--peer-ttl-ms 15000] [--peer-fail-threshold 3]
  [--peer-cooldown-ms 30000] [--cluster-stickiness-ms 15000]
  [--load-penalty 0.05] [--load-decay 0.9]
  [--state-dir ./state]
  [--role inference|routing|cache]
  [--interval <secs>]
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

### `dist-plan` — Generate a layer shard plan

```bash
cargo run -p mesh-node -- dist-plan \
  --peers peer-a,peer-b,peer-c \
  [--models <path> --model <name>] \
  [--config <path> --params-b <f64>] \
  [--min-layers-per-peer 1]
```

### `dist-run` — Execute a distributed inference pass

```bash
cargo run -p mesh-node [--features burn] -- dist-run \
  --peers peer-a,peer-b \
  [--models <path> --model <name>] \
  [--config <path> --params-b <f64>] \
  [--input "prompt text"] \
  [--max-tokens 64] [--temperature 0.7] \
  [--executor echo|burn|iroh] \
  [--model-path /path/to/weights.safetensors] \
  [--model-device ndarray|wgpu] \
  [--model-mode simulate|mlp|weights|forward|embed] \
  [--model-hidden 64] \
  [--model-tensor "model.embed_tokens.weight"] \
  [--model-sample-bytes 256]
```

**Executor modes:**

| `--executor` | Description |
|---|---|
| `echo` | No-op: returns the input annotated with shard info |
| `burn` | Local Burn NdArray simulation (requires `--features burn`) |
| `iroh` | Real distributed pass over iroh QUIC |

**`--model-mode` options (burn executor):**

| Mode | What it does |
|---|---|
| `simulate` | Adds layer index to a scalar — proves the shard pipeline end-to-end |
| `mlp` | Random-weight MLP: matmul + tanh over the input bytes |
| `weights` | Loads a named tensor from safetensors, reports shape + checksum |
| `forward` | Matrix-vector multiply with a loaded weight tensor |
| `embed` | Embedding lookup: maps input bytes to token IDs, sums embedding rows |

### `simulate` — Cluster selection simulation

```bash
cargo run -p mesh-node -- simulate \
  --config <path> --params-b <f64> \
  [--peers peers.json] [--rtt-samples rtt.json] \
  [--min-nodes 2] [--max-nodes 6] [--min-bw 500]
```

### `infer-http` — Query an OpenAI-compatible server

```bash
cargo run -p mesh-node -- infer-http \
  --api-base http://127.0.0.1:8000/v1 \
  --model <model-name> \
  --prompt "Your prompt here" \
  [--system "System message"] \
  [--max-tokens 128] [--temperature 0.7] [--api-key sk-...]
```

### `infer-local` — Run a local inference engine binary

Supports any CLI tool that accepts a prompt via arguments.  Template variables:
`{model}`, `{prompt}`, `{max_tokens}`, `{temperature}`, `{context}`.

```bash
cargo run -p mesh-node -- infer-local \
  --local-bin /path/to/llama-cli \
  --local-args "-m {model} -p {prompt} -n {max_tokens} --temp {temperature} --ctx-size {context}" \
  --local-model /path/to/model.gguf \
  --prompt "Hello"
```

### `api` — Start the OpenAI-compatible REST API server

```bash
cargo run -p mesh-node -- api \
  --bind 127.0.0.1:8000 \
  [--infer-base http://127.0.0.1:8001/v1] \
  [--default-model <name>] \
  [--local-bin /path/to/llama-cli \
   --local-args "..." \
   --local-model /path/to/model.gguf] \
  [--api-auth <secret-key>] \
  [--allow-model <name>] \
  [--rate-limit-rps 5] [--rate-limit-burst 10]
```

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Node health check |
| `GET` | `/v1/models` | List available models |
| `POST` | `/v1/chat/completions` | Chat inference (proxied or local) |

---

## Cargo Features

| Feature | Description |
|---------|-------------|
| *(none)* | Base build: networking, planning, echo executor, API server |
| `burn` | Enables Burn 0.20 (NdArray + WGPU backends), `LlamaShard`, `KvCacheStore`, real tensor ops |

```bash
cargo build                    # no ML deps
cargo build --features burn    # full Burn stack (safetensors weight loading, NdArray, WGPU)
cargo test                     # run all tests
cargo test --features burn     # run tests with Burn backend
```

---

## Configuration

### Model config (`config.json`)

Any Llama-style architecture.  Standard HuggingFace `config.json` fields are supported:

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

### Gossip with cluster selection + fallback

```bash
cargo run -p mesh-node -- gossip \
  --topic <64-hex> \
  --models models.json \
  --model-name llama-3-8b \
  --fallback-model mistral-7b:configs/mistral-7b.json:7 \
  --free-vram 24 \
  --min-nodes 2 --max-nodes 8 \
  --rtt-probe \
  --state-dir ./state
```

---

## Roadmap

### Phase 1 — Foundation ✅ (complete)
- [x] iroh QUIC transport + gossip peer discovery
- [x] Capability announcement (VRAM, RTT, bandwidth, GPU score)
- [x] Proximity-aware cluster selection
- [x] VRAM estimation for any model config
- [x] Round-robin shard planner
- [x] Content-addressed shard store
- [x] OpenAI-compatible HTTP API with auth and rate limiting
- [x] State persistence
- [x] Burn 0.20 integration (NdArray + WGPU, model-agnostic)

### Phase 2 — Real Distributed Inference 🚧 (in progress)
- [ ] Full pipeline: activation tensors over QUIC between peers
- [ ] KV cache synchronization across shard boundaries
- [ ] LlamaShard end-to-end with real checkpoints (Llama-3, Mistral)
- [ ] Streaming token output (SSE)
- [ ] Fault-tolerant re-planning on peer drop

### Phase 3 — Performance & Scale 📋 (planned)
- [ ] 4/5/8-bit quantized weight loading (GGUF + safetensors)
- [ ] Tensor parallelism (column/row parallel linear layers)
- [ ] Speculative decoding across shards
- [ ] MoE expert placement (Mixtral, DeepSeek)
- [ ] iroh-blobs weight distribution (P2P checkpoint transfer)
- [ ] Adaptive context reduction under memory pressure
- [ ] Continuous batching

### Phase 4 — Production 📋 (planned)
- [ ] Reputation system (peer reliability scoring over time)
- [ ] Encrypted inference (homomorphic / TEE investigation)
- [ ] Web dashboard (peer topology, cluster health)
- [ ] Docker / container deployment
- [ ] Incentive layer (optional token-gated access)

---

## Project Structure

```
Samhati/
├── crates/
│   ├── mesh-node/              # Main binary (CLI + server)
│   │   └── src/
│   │       ├── main.rs         # All CLI commands
│   │       ├── server.rs       # iroh QUIC inference server
│   │       ├── protocol.rs     # Gossip message types
│   │       ├── scheduler.rs    # Cluster candidate scoring
│   │       └── state.rs        # Peer / session state
│   ├── inference-coordinator/  # Shard planning + tensor execution
│   │   └── src/
│   │       ├── coordinator.rs  # Orchestrates multi-shard pass
│   │       ├── model_runner.rs # ModelShardRunner (Burn NdArray modes)
│   │       ├── llm_shard.rs    # LlamaShard<B> (real weight loading)
│   │       ├── kv_cache.rs     # KvCacheStore<B> (per-session KV)
│   │       ├── tensor_frame.rs # Wire format for activation tensors
│   │       ├── iroh_executor.rs# QUIC-based distributed executor
│   │       ├── plan.rs         # RoundRobinPlanner + ShardPlan
│   │       └── rpc.rs          # RPC request/response types
│   ├── proximity-router/       # Peer scoring and ranking
│   ├── cluster-manager/        # VRAM-aware cluster constraint solver
│   ├── model-config/           # Config parsing + VRAM estimation
│   ├── shard-store/            # Content-addressed weight shard cache
│   ├── api/                    # OpenAI-compatible HTTP layer
│   └── bench/                  # Benchmarks
├── sample-config.json          # Example model config
├── sample-peers.json           # Example peer list
└── MVP_SPEC.md                 # Architecture spec
```

---

## Contributing

1. Fork and clone the repo
2. `cargo build` — must compile clean with zero errors
3. `cargo build --features burn` — must also compile clean
4. `cargo test` — all tests must pass
5. Open a PR with a clear description of the change

Please keep PRs focused.  One feature or fix per PR.

---

## License

MIT
