# Distributed Mesh Spec (Model-Agnostic)

## Goals
- Run any supported LLM architecture (Llama, BLOOM, Falcon, Mixtral) across heterogeneous nodes.
- Capable of scaling to massively large models (e.g. Llama-3.1 405B+).
- Keep latency minimal using proximity-based routing.
- Handle unknown GPU availability via adaptive cluster sizing.

## Defaults
- Context: 8k
- KV cache: 8-bit
- Quantization: 4/5/8-bit candidates
- Latency: first token <= 2s regional, <= 4s WAN; throughput >= 20 tok/s regional

## Coordinator Logic
1. Gather node capabilities (free VRAM, RTT, bandwidth, reliability).
2. Try quantization levels (4/5/8) and node counts (1..10).
3. For each candidate:
   - Estimate VRAM per node
   - Filter nodes by capacity
   - Enforce RTT threshold
4. Choose smallest feasible cluster with best score.
5. If no feasible cluster:
   - Reduce context length
   - Increase cluster size
   - Fall back to smaller model

## Estimation Function (pseudocode)
```
required_per_node = (weights_total / N)
                  + (kv_total / N)
                  + runtime_overhead
```

## Open Decisions
- Exact model config values (from `config.json`)
- MoE expert placement strategy in Phase 2+
