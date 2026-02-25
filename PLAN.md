# Execution Plan

## Phase 0 — Project Skeleton (1 week)
Deliverables
- Rust workspace structure
- Crate layout: `mesh-node`, `proximity-router`, `cluster-manager`, `shard-store`, `inference-coordinator`, `api`, `bench`
- Basic config + CLI

Acceptance criteria
- Builds locally and runs a dummy node process

## Phase 1 — Core Mesh + Proximity Routing (2–3 weeks)
Deliverables
- iroh transport integration
- iroh-gossip membership + capability broadcasting
- RTT measurement + peer scoring
- Tiered bucket selection

Acceptance criteria
- Nodes can discover and rank peers by RTT
- Router selects top-N peers correctly under churn

## Phase 2 — Distributed Inference (MVP) (3–5 weeks)
Deliverables
- candle integration for model execution
- Shard storage with iroh-blobs
- Pipeline execution across 2–4 nodes
- Adaptive cluster sizing (quantization + nodes)

Acceptance criteria
- gpt-oss-120b (4-bit, 8k) runs across a small cluster
- Latency targets achieved in regional setup

## Phase 3 — Regional Optimization (2–4 weeks)
Deliverables
- KV cache affinity + session pinning
- Adaptive context length + fallback logic
- Regional cluster enforcement

Acceptance criteria
- Stable throughput under region-local clusters
- Failover without catastrophic latency spikes

## Phase 4 — Incentive Layer (optional) (2–4 weeks)
Deliverables
- Identity + staking hooks (no on-chain yet)
- Reputation scoring

Acceptance criteria
- Stubs integrated with routing + scheduling

## Phase 5 — Scale + Hardening (ongoing)
Deliverables
- WAN-scale testing
- Load testing and chaos testing
- Observability dashboards

Acceptance criteria
- Sustained multi-region load with minimal SLA violations
