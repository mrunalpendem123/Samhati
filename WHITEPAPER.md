# Samhati: A Peer-to-Peer Open Intelligence Network

**Mrunal Pendem**
**samhati@proton.me**
**March 2026**

---

## Abstract

A purely peer-to-peer intelligence network would allow any device to serve AI inference without relying on centralized providers. The problem with current AI infrastructure is that inference is metered — every query costs money because it runs on hardware owned by a corporation. We propose a system where every participant simultaneously consumes and contributes intelligence, eliminating the need for a central provider. Queries fan out to multiple independent nodes, each running a small language model. Nodes peer-rank each other's answers using pairwise comparison, and BradleyTerry maximum likelihood estimation selects the statistically best response. A cryptographic Proof of Inference, built from output logprobs and Ed25519 signatures, prevents replay attacks and binds proofs to specific node identities. ELO ratings track node quality over time, and results settle on Solana. The result is an open intelligence network with no gatekeepers, no staking requirements, and no cost to participate.

---

## 1. Introduction

AI inference today follows the client-server model. Users send queries to centralized providers — OpenAI, Anthropic, Google — who run large models on their hardware and charge per token. This creates three problems: cost scales linearly with usage, a single point of failure controls access, and users have no way to verify what model actually served their request.

What is needed is a system that distributes both the cost and the trust across all participants. If every device that uses AI also provides AI, the aggregate cost approaches zero — not because inference is free, but because it is shared. This is the same principle that made BitTorrent work for bandwidth: downloaders are simultaneously uploaders, so nobody pays for distribution.

We propose Samhati, a decentralized network where nodes run small language models (0.5B–14B parameters) on consumer hardware and earn token rewards for producing high-quality answers. Quality is enforced not by verifying internal computation, but by having multiple nodes answer the same query and statistically selecting the best response.

---

## 2. Swarm Inference

A query enters the network and is classified by complexity: Easy, Medium, or Hard. Easy and Medium queries are dispatched to 3 nodes. Hard queries are dispatched to 5 nodes and include a debate round.

Each selected node runs inference independently using llama.cpp and returns an answer alongside a Proof of Inference. The fan-out is parallel — all nodes begin inference simultaneously. A 120-second timeout drops unresponsive nodes without blocking the round.

```
Query → Classify → Select N nodes → Fan-out (parallel)
     → Collect answers + proofs → Rank → Aggregate → Winner
```

The network does not require all nodes to run the same model. Heterogeneity is encouraged: a round might include Qwen-7B, Llama-8B, and DeepSeek-Coder-7B answering the same question. This diversity is a feature — it prevents single-model failure modes and produces a broader range of reasoning approaches.

---

## 3. Peer Ranking

After all answers are collected, every answer is evaluated via pairwise comparison. For each pair of answers (A, B), a randomly assigned judge node is asked: "Which answer is better? Judge only on factual correctness, logical reasoning, and completeness. Ignore confidence, style, and length."

The judge returns a preference score (0–1) and a short reasoning chain (50–100 tokens). The judge cannot evaluate its own answer — this prevents self-promotion bias.

Judge assignment is randomized per round using `BLAKE3(timestamp + prompt + node_count)`. This prevents collusion: even if three nodes coordinate, they cannot predict or control which pairs they will judge.

For N nodes, there are C(N,2) pairs, each judged by exactly one randomly assigned node. This produces fewer total ranking calls than exhaustive evaluation (N × C(N,2)) while maintaining sufficient signal for aggregation.

---

## 4. BradleyTerry Aggregation

The pairwise preferences are aggregated into a global ranking using the BradleyTerry model [1], estimated via the MM (Majorization-Minimization) algorithm [2].

Each node *i* has a latent strength parameter β_i. The probability that node *i* is preferred over node *j* in a pairwise comparison is modeled as:

```
P(i > j) = exp(β_i) / (exp(β_i) + exp(β_j))
```

The MM algorithm iterates:

```
For each node i:
  numerator   = Σ (weight × wins for i)
  denominator = Σ (weight / (exp(β_i) + exp(β_j)))  for all comparisons
  β_i_new     = ln(numerator / denominator)

Normalize: subtract mean(β) from all values.
Converge when max|β_new - β_old| < 10⁻⁶.
```

Rankings from higher-ELO judges carry more weight. The algorithm typically converges in 10–25 iterations. The winner is the node with the highest estimated strength, and the confidence is the softmax probability of the winner.

---

## 5. ELO Reputation

Each node maintains a reputation score R ∈ [0, 1] computed from two independent tracks:

```
R = 0.4 × R_ranking + 0.6 × R_generation
```

**R_generation** measures answer quality — did your answer win? Updated via exponential moving average:

```
R_gen(t+1) = 0.85 × R_gen(t) + 0.15 × (1 if won, 0 if lost)
```

**R_ranking** measures judgment quality — did your rankings agree with BradleyTerry consensus? A node that gives good answers but ranks dishonestly will be caught: its R_ranking drops even if R_generation stays high. Single-track systems like ELO cannot detect this.

**Uncertainty (σ):** Each node starts with σ = 0.4 (uncertain) which decays 5% per round to a minimum of 0.05 (established). Inspired by TrueSkill's Bayesian skill tracking.

**Adaptive validation rate:** New nodes (high σ) are audited ~50% of rounds. Trusted nodes (high R, low σ) are audited ~2%. This reduces verification cost by 25× without sacrificing security.

```
validation_rate = 50% - (50% - 2%) × R × (1 - σ)
```

Domain-specific R_generation is tracked independently for Code, Math, Reasoning, and General. Nodes that fail proof verification are slashed to 0 and must recalibrate. Inactive nodes decay 2% per round.

---

## 6. Proof of Inference

Each node produces a cryptographic proof alongside its answer. The proof is built from the output logprobs — the top-K token probability distributions that the model produces at each generation step.

```
For each generated token t:
  Record token_id and top-8 (token_id, logprob) pairs

Group tokens into chunks of 32.
For each chunk:
  Serialize deterministically (big-endian, sorted by token_id)
  Hash with BLAKE3 → chunk_hash

Construct signable message:
  model_hash || token_count || chunk_hashes || timestamp || node_pubkey

Sign with Ed25519(node_secret_key) → proof
```

The verifier checks: (1) model hash matches the claimed model, (2) token count matches output length, (3) chunk count equals ⌈token_count / 32⌉, (4) chunk indices are sequential, (5) timestamp is within 5 minutes of current time, (6) node_pubkey is registered, (7) Ed25519 signature is valid.

This proof does not verify which model ran internally — it verifies that *a* model produced specific output probability distributions, signed by a specific node, within a recent time window. It prevents replay attacks, proof theft between nodes, and response tampering. It does not prevent a node from running a different model than claimed — that is handled by the ranking mechanism, which naturally filters low-quality outputs regardless of their source.

---

## 7. Network

Nodes discover each other through two mechanisms: an on-chain registry on Solana, and a gossip protocol over iroh [4] QUIC connections.

On startup, a node:

1. Loads its identity — a single Ed25519 keypair that derives its Solana address (base58), iroh NodeId (hex), and proof signing key.
2. Registers on Solana if not already registered. This creates a NodeAccount PDA storing the node's ELO, model name, round count, and accumulated rewards.
3. Reads all registered NodeAccount PDAs from Solana (a free RPC call) to discover existing nodes.
4. Connects to discovered nodes via iroh QUIC with relay-assisted NAT traversal.
5. Joins the gossip topic and begins broadcasting its model name and inference endpoint every 10 seconds.

When a new peer is discovered via gossip, the node verifies it is registered on Solana before accepting it. This prevents unregistered nodes from participating in the swarm.

---

## 8. Incentive

Nodes are rewarded with SMTI tokens for participating in rounds. The reward distribution follows a winner-takes-most model:

```
Base emission per round: 1000 SMTI

Winner pool:      60% × base = 600 SMTI
Participant pool: 40% × base = 400 SMTI

Winner reward  = 600 + (400 × winner_elo / Σ elo_all)
Other rewards  = 400 × node_elo / Σ elo_all
```

Domain specialists — nodes running models tagged for Code, Math, or Reasoning — earn a 1.5× multiplier on rounds matching their domain. This creates a market signal: the network's domain demand counters (tracked on-chain) show operators what the network needs, and specialists are rewarded for filling that demand.

There is no staking requirement. Any device can join the network and begin earning immediately. Economic defense against Sybil attacks comes not from financial stakes but from compute cost: producing a valid Proof of Inference requires running real inference, and producing a winning answer requires a model good enough to beat peers in pairwise ranking.

---

## 9. Domain Routing

A lightweight classifier tags each incoming query with a domain: Code, Math, Reasoning, or General. The classifier runs locally on the orchestrating node in under 1 millisecond using keyword heuristics and structural analysis (presence of code blocks, mathematical notation, multi-step reasoning patterns).

Domain tags serve three purposes: (1) node selection is biased toward specialists in the query's domain, (2) domain-specific ELO is used for ranking weight, and (3) on-chain demand counters are incremented, providing real-time market signals to operators.

---

## 10. Settlement

After each round, the result is settled on Solana via the `submit_round` instruction on the Samhati Anchor program. The on-chain record includes:

- Round ID and timestamp
- Participant public keys
- BLAKE3 hashes of each node's Proof of Inference
- ELO deltas (signed integers) for each participant
- Winner public key
- Domain tag and SMTI emission amount

The Anchor program validates that all array lengths match, the winner is among the participants, and the coordinator's signature is valid. Each participant's NodeAccount is updated atomically: ELO adjusted, round count incremented, rewards credited.

Domain demand counters on the ProtocolConfig PDA are incremented per round, providing a permanent on-chain record of what the network is being asked.

```
Program ID: AB7cSMLv1J7J28DKLMbzo2tyNp1kZSmE67a6Heoa5Mkr (devnet)

PDAs:
  ProtocolConfig  [b"config"]              — authority, emission, demand
  NodeAccount     [b"node", pubkey]        — ELO, model, rounds, rewards
  RoundAccount    [b"round", round_id]     — participants, proofs, deltas
```

---

## 11. Security Analysis

**Sybil attacks.** An attacker spinning up many fake nodes must run real inference for each one (Proof of Inference requires valid logprobs). If answers are low quality, fake nodes lose ELO and stop being selected. The compute cost of maintaining many nodes exceeds the reward from gaming the ranking.

**Collusion.** Coordinating nodes cannot control which pairs they judge — assignments are derived from `BLAKE3(timestamp + prompt + node_count)`, which changes every round and is unpredictable. Even if K out of N nodes collude, the probability that all judges for a colluder's pairs are also colluders drops exponentially with honest participation.

**Replay attacks.** Proofs include a timestamp and are rejected if older than 5 minutes. Each proof is bound to a specific node_pubkey via Ed25519 signature — a proof generated by node A cannot be submitted by node B.

**Self-promotion.** The ranking prompt explicitly instructs judges to evaluate factual correctness only and ignore confidence, persuasive language, and style. Nodes cannot judge their own answers.

**Adversarial inputs.** The network supports 21 models across 4 architectures (Qwen, Llama, DeepSeek, Gemma, Phi, Mistral). An adversarial input that exploits one model is unlikely to affect a different architecture. The debate round (for hard queries) further dilutes single-model failure modes by having nodes revise their answers after seeing alternatives.

**Model freeloading.** A node running a cheap 0.5B model instead of a claimed 7B model will produce lower-quality answers. BradleyTerry ranking will consistently rank it below honest 7B nodes. Its ELO will decay, reducing its selection probability and rewards. The system self-corrects without requiring model verification — quality, not identity, is what matters.

---

## 12. Conclusion

We have proposed a peer-to-peer intelligence network where every participant simultaneously uses and provides AI inference. Queries fan out to multiple independent nodes, peer ranking selects the best answer via BradleyTerry aggregation, and ELO tracks node quality over time. Cryptographic Proofs of Inference prevent replay and tampering without requiring internal model verification. Random judge assignment prevents collusion. Results settle on Solana with on-chain proof hashes, ELO deltas, and domain demand tracking. The system requires no staking, no special hardware, and no permission to join — open intelligence for everyone, forever.

---

## References

[1] R. A. Bradley and M. E. Terry, "Rank analysis of incomplete block designs: I. The method of paired comparisons," *Biometrika*, vol. 39, no. 3/4, pp. 324–345, 1952.

[2] K. Lange, D. R. Hunter, and I. Yang, "Optimization transfer using surrogate objective functions," *Journal of Computational and Graphical Statistics*, vol. 9, no. 1, pp. 1–20, 2000.

[3] A. E. Elo, *The Rating of Chessplayers, Past and Present*, Arco Publishing, 1978.

[4] n0 Computer, "iroh: A toolkit for building distributed systems," https://iroh.computer, 2024.

[5] A. Nakamoto, "Bitcoin: A peer-to-peer electronic cash system," https://bitcoin.org/bitcoin.pdf, 2008.

[6] Y. Du, S. Li, A. Torralba, J. B. Tenenbaum, and I. Mordatch, "Improving factuality and reasoning in language models through multiagent debate," *arXiv preprint arXiv:2305.14325*, 2023.

[7] Fortytwo Network, "Swarm inference with peer-ranked consensus," *arXiv preprint arXiv:2510.24801*, 2025.

[8] Solana Foundation, "Solana: A new architecture for a high performance blockchain," https://solana.com/solana-whitepaper.pdf, 2020.
