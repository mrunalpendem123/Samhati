# Petals Implementation Plan for Decentralized Proximity-Aware LLM Mesh

Based on an analysis of the [Petals repository](https://github.com/bigscience-workshop/petals) and your current `new2` codebase, here is a structured plan to replace your stubs (`EchoExecutor` and `simulate` mode) with a true, Petals-style distributed inference engine in Rust.

---

## 🏗️ Architectural Mapping: Petals ➡️ Your Rust Mesh

Your project already handles the **DHT & Discovery** phase beautifully using `iroh-gossip`. Petals relies heavily on `hivemind` for this. The next phase is bridging the gap for **Distributed Execution**.

| Concept | Petals Implementation | Your Current Rust Stub | Planned Rust Target |
| :--- | :--- | :--- | :--- |
| **P2P Node Networking** | `hivemind` DHT, gRPC | `iroh-gossip`, Capabilities | `iroh` QUIC Streams / ALPN endpoints |
| **Orchestrator** | `InferenceSession` (Client driven) | `Coordinator::run` (Local sequential loop) | `DistributedCoordinator` w/ network streams |
| **Server Request Handler** | `TransformerConnectionHandler` | None / Single CLI Execution | `MeshRpcServer` listening on `iroh` stream |
| **Tensor Execution** | `backend.py` (PyTorch) | `CandleShardRunner` (Simulate mock) | `CandleShardRunner` (Real tensor ops) |
| **State Persistence** | `MemoryCache` (KV Cache) | No state between calls | `KvCacheManager` matching Session IDs |
| **Activations (Tensors)** | Protobuf `ExpertRequest/Response`| Uses `String` text prompts | Custom typed `TensorBytes` (Safetensors) |

---

## 🛠️ Step-by-Step Porting Plan

### Step 1: Upgrade the Communication Protocol
To send actual activation tensors between nodes, you need a streamlined RPC mechanism over your existing `iroh` network.

1. **Create an RPC Definition (`protocol.rs` or new `rpc` crate):**
   * Define protobuf (via `prost`) or bincode/msgpack structures for:
     * `ForwardRequest` (Session ID, Tensors bytes, Step, Prompts)
     * `ForwardResponse` (Tensors bytes)
2. **Setup iroh ALPN Endpoints:**
   * Petals utilizes `hivemind`'s P2P streams. In Rust, you can use `iroh` Endpoint streams (`Endpoint::accept(ALPN_INFERENCE)`).
   * Nodes must listen asynchronously for incoming QUIC streams on an `inference` ALPN.

### Step 2: Implement the `MeshRpcServer` (The "Handler")
This is the equivalent of Petals' `server/handler.py`.

1. **RPC Listener Loop (`mesh-node/src/rpc_server.rs`):**
   * Listen for `ForwardRequest`.
   * Unpack the bytes into Candle `Tensor`.
2. **Context (KV Cache) Management:**
   * Build a `KvCacheManager`. Petals isolates attention cache by request `session_id`.
   * When an RPC call provides a `session_id`, the handler looks up existing KV cache tensors in VRAM and feeds them into the `CandleShardRunner`.
3. **Execution Delivery:**
   * The handler sends the reconstructed Tensors and KV cache to `CandleShardRunner::run_shard_real()`.
   * Serializes the output `Tensor` back to bytes, and responds over the QUIC stream (`ForwardResponse`).

### Step 3: Transform `Coordinator` to operate like `InferenceSession`
Currently, `Coordinator<E>` expects `run_shard` to return a `String`. We need it to operate on distributed streams.

1. **The Client-Driven Loop:**
   * Similar to Petals, your `Coordinator` should hold a session.
   * `let mut current_tensor = inputs;`
   * `for shard in plan.shards {`
     * Determine `peer_id` for that shard.
     * Open an `iroh` connection to that `peer_id`.
     * Send `current_tensor` inside a `ForwardRequest`.
     * Await `ForwardResponse`.
     * `current_tensor = response.tensors;`
   * `}`
2. **Fault Tolerance (Bonus):**
   * If an `iroh` QUIC stream times out or drops (Petals uses `spending_policy.py`), trigger a re-route. Ask `iroh-gossip` for the next best peer supporting those layers.

### Step 4: Upgrade `CandleShardRunner`
Your current `run_simulated` does mock text appending. It needs to utilize the `candle_mode` hooks you scaffolded (`mlp`, `weights`).

1. **Actual Weights Loading:** 
   * Pre-load specific block keys (`model.layers.N.self_attn...`) per standard LLM architecture based on the shard specs.
2. **Run Real Inference:**
   * Construct `candle::Tensor` instances from the incoming ALPN bytes (bincode deserialized).
   * Perform `layer(...)` forward pass using the KV Cache pulled from your new `KvCacheManager`.
3. **Quantization Hooks:**
   * Petals implements dynamic 8-bit cache. Since you already pass `--kv-bits`, make sure `candle` initializes these tensors as quantized structs.

### Step 5: Iterative Testing Path
1. **Network ping-pong:** Connect one node passing raw identical `[0.0, 0.0]` tensors to another node, and log the arrival.
2. **Local Echo:** Node A sends `Tensor` to Node B. Node B multiplies by 2 and returns it. Verify Node C receives `Tensor * 4`.
3. **Single Layer MLP test:** Node B runs a real MLP forward pass on received tensors using random weights.
4. **Integration:** Hook the Pipeline into the end-user API endpoints via `Coordinator`.
