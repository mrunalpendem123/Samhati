# Decentralized Proximity-Aware LLM Mesh (gpt-oss-120b target)

This repo will hold the core mesh node, proximity routing, cluster formation, shard storage, and inference coordination for a decentralized, latency-aware LLM mesh.

## Defaults (MVP)
- Model target: gpt-oss-120b
- Context: 8k
- KV cache: 8-bit (with fp16 fallback)
- Latency targets: first token <= 2.0s (regional), <= 4.0s (WAN); throughput >= 20 tok/s (regional)

See `CONFIG_DEFAULTS.md` and `PLAN.md` for implementation details.

## Quick Run (Gossip MVP)
Open two terminals and run:

```bash
cargo run -p mesh-node -- gossip --topic 1717171717171717171717171717171717171717171717171717171717171717
```

It will print `node_id`. Copy the first node's `node_id` and start a second node with:

```bash
cargo run -p mesh-node -- gossip --topic 1717171717171717171717171717171717171717171717171717171717171717 --bootstrap <node_id>
```

Both nodes should receive periodic heartbeat messages on the shared topic.

## Live Cluster Selection (Gossip + Capabilities)
If you pass a model config, the gossip node will continuously rank peers and select a cluster.

```bash
cargo run -p mesh-node -- gossip \
  --topic 1111111111111111111111111111111111111111111111111111111111111111 \
  --config /Users/mrunalpendem/Desktop/new2/sample-config.json \
  --params-b 117 \
  --min-nodes 2 --max-nodes 6 \
  --free-vram 80 --bandwidth 1200 --reliability 0.98 --gpu-score 0.9 --rtt-ms 10 --kv-bits 8 --context 8192 --quant 4
```

Notes:
- If `--free-vram` is omitted, the node auto-detects available memory and applies a safety margin.
- Peer/model state is persisted to `./state/state.json` by default. Override with `--state-dir`.

### Model Fallback (Optional)
You can provide additional model configs to try if the primary model doesn't fit:

```bash
--model-name gpt120b \
--fallback-model gpt40b:/path/to/40b/config.json:40 \
--fallback-model gpt7b:/path/to/7b/config.json:7
```

### Model Registry
You can supply a JSON registry so the node can choose among multiple models:

```bash
--models /Users/mrunalpendem/Desktop/new2/models.json
```

The registry is also broadcast over gossip so peers can discover what models each node supports.

### RTT Probing (Gossip Ping/Pong)
Enable RTT probing to replace self-reported RTT:

```bash
--rtt-probe --rtt-timeout-ms 3000
```

### Scheduling Controls (Stability + Load Balancing)
```bash
--peer-ttl-ms 15000 --peer-fail-threshold 3 --peer-cooldown-ms 30000 \
--cluster-stickiness-ms 15000 --load-penalty 0.05 --load-decay 0.9
```

### Capacity / Peers Needed
Compute how many peers are needed given free VRAM per node:

```bash
cargo run -p mesh-node -- capacity \
  --free-vram 8 --max-nodes 8 --seq 4096 --quant 4 \
  --models /Users/mrunalpendem/Desktop/new2/models.json \
  --model stablelm-3b-4e1t
```

### Distributed Inference (Scaffold)
Generate a shard plan (layer ranges per peer):

```bash
cargo run -p mesh-node -- dist-plan \
  --peers peer-a,peer-b,peer-c \
  --models /Users/mrunalpendem/Desktop/new2/models.json \
  --model stablelm-3b-4e1t
```

Run a stubbed distributed pass (echo executor):
```bash
cargo run -p mesh-node -- dist-run \
  --peers peer-a,peer-b,peer-c \
  --models /Users/mrunalpendem/Desktop/new2/models.json \
  --model stablelm-3b-4e1t \
  --input "Hello"
```

Run a Candle shard-runner stub (enable the candle feature):
```bash
cargo run -p mesh-node --features candle -- dist-run \
  --peers peer-a,peer-b \
  --models /Users/mrunalpendem/Desktop/new2/models.json \
  --model stablelm-3b-4e1t \
  --executor candle \
  --candle-device cpu \
  --candle-mode simulate
```

Run a Candle MLP test (real tensor ops, no weights yet):
```bash
cargo run -p mesh-node --features candle -- dist-run \
  --peers peer-a,peer-b \
  --models /Users/mrunalpendem/Desktop/new2/models.json \
  --model stablelm-3b-4e1t \
  --executor candle \
  --candle-device cpu \
  --candle-mode mlp \
  --candle-hidden 8
```

Run a Candle weights test (loads real safetensors and computes a checksum):
```bash
cargo run -p mesh-node --features candle -- dist-run \
  --peers peer-a,peer-b \
  --models /Users/mrunalpendem/Desktop/new2/models.json \
  --model stablelm-3b-4e1t \
  --executor candle \
  --candle-mode weights \
  --model-path /path/to/model.safetensors \
  --candle-tensor "model.embed_tokens.weight" \
  --candle-sample-bytes 512
```

### Inference Stub (OpenAI-Compatible HTTP)
Use the `infer-http` command to call any OpenAI-compatible server (e.g., a local model server).

```bash
cargo run -p mesh-node -- infer-http \
  --api-base http://127.0.0.1:8000/v1 \
  --model gpt-oss-20b \
  --prompt "Hello"
```

### Local Inference (External Engine Exec)
You can run a local engine binary (e.g., `llama.cpp` CLI) by passing a command template. The template supports:
`{model}`, `{prompt}`, `{max_tokens}`, `{temperature}`, `{context}`.

```bash
cargo run -p mesh-node -- infer-local \
  --local-bin /path/to/llama-cli \
  --local-args "-m {model} -p {prompt} -n {max_tokens} --temp {temperature} --ctx-size {context}" \
  --local-model /path/to/model.gguf \
  --prompt "Hello"
```

### API Skeleton (OpenAI-Compatible)
Start a minimal API server (no model wired yet):

```bash
cargo run -p mesh-node -- api --bind 127.0.0.1:8000
```

Endpoints:
- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions` (proxy to backend if `--infer-base` is set)

Example proxy:
```bash
cargo run -p mesh-node -- api \
  --bind 127.0.0.1:8000 \
  --infer-base http://127.0.0.1:8001/v1 \
  --default-model gpt-oss-20b
```

Example local engine:
```bash
cargo run -p mesh-node -- api \
  --bind 127.0.0.1:8000 \
  --local-bin /path/to/llama-cli \
  --local-args "-m {model} -p {prompt} -n {max_tokens} --temp {temperature} --ctx-size {context}" \
  --local-model /path/to/model.gguf
```

### API Security (Auth + Allowlist + Rate Limit)
```bash
cargo run -p mesh-node -- api \
  --bind 127.0.0.1:8000 \
  --local-bin /path/to/llama-cli \
  --local-args "-m {model} -p {prompt} -n {max_tokens} --temp {temperature} --ctx-size {context}" \
  --local-model /path/to/model.gguf \
  --api-auth my-secret-key \
  --allow-model stablelm-3b-4e1t \
  --rate-limit-rps 5 --rate-limit-burst 10
```

Clients must set either:
- `Authorization: Bearer my-secret-key`
- or `x-api-key: my-secret-key`
