# Config Defaults (MVP)

These are the defaults we will implement for the first working system. They can be overridden at runtime.

## Model
- Target: gpt-oss-120b
- Source of truth: model config file distributed with the weights (e.g., `config.json`).
- TODO: Extract exact values for:
  - `n_layers`
  - `n_kv_heads`
  - `head_dim`
  - `hidden_size`
  - `vocab_size`

## Inference
- Default context length: 8k
- KV cache precision: 8-bit by default, fp16 fallback
- Quantization candidates: 4-bit, 5-bit, 8-bit

## Latency Targets
- Regional clusters: first token <= 2.0s, throughput >= 20 tok/s
- WAN mesh: first token <= 4.0s

## Cluster Rules
- Max RTT threshold (regional): 20-40 ms (tuned in Phase 2)
- Max nodes per pipeline: adaptive (2..10), chosen by VRAM constraints + latency score
- Routing preference: closest-nodes-first, strict proximity

## Fallbacks
- If no feasible cluster: reduce context length, then reduce quantization bits, then fallback to smaller model
