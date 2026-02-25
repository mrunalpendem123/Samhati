# Architecture Overview

## Core Components
- `mesh-node` (binary)
  - Node bootstrap and identity
  - Capability detection and periodic reporting
  - Runs transport, gossip, shard store, and inference worker

- `proximity-router`
  - RTT measurement
  - Peer scoring and ranking
  - Tiered bucket management

- `cluster-manager`
  - Forms and maintains inference clusters
  - Enforces RTT and capability thresholds
  - Handles cluster failover and healing

- `shard-store`
  - Model shard storage and caching (iroh-blobs)
  - Content-addressed model layers

- `inference-coordinator`
  - Builds pipelines
  - Enforces closest-nodes-first routing
  - Handles session KV affinity

## Data Flow
1. Node starts -> publishes capability profile via gossip.
2. Proximity router measures RTT and builds peer ranking.
3. Cluster manager selects a feasible cluster for the requested model/quant/context.
4. Shard store ensures required layers are available (fetch + cache).
5. Inference coordinator dispatches pipeline across cluster.

## Routing Policy
- Primary: minimize RTT (latency score dominates)
- Secondary: bandwidth + reliability + GPU capacity
- Enforce strict locality for MoE expert dispatch if expert-sharded

## MoE Strategy (gpt-oss-120b)
- MVP: keep experts local within a cluster; avoid cross-cluster expert routing
- Phase 2+: explore expert-parallel if VRAM constraints demand it

## Failure Handling
- If cluster health degrades: re-route within region, then cross-region failover
- If required shards are missing: fetch from closest peers with verified integrity
