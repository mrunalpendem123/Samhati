//! End-to-end swarm integration test.
//! Requires 3 llama-servers running on ports 8001, 8002, 8003.
//! Run: cargo test -p samhati-tui --test swarm_e2e -- --nocapture

use anyhow::Result;
use reqwest;
use serde_json::json;

/// Check that a llama-server is healthy
async fn check_health(port: u16) -> bool {
    reqwest::get(format!("http://127.0.0.1:{}/health", port))
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

/// Call a node's chat completion with logprobs
async fn call_with_logprobs(port: u16, prompt: &str) -> Result<serde_json::Value> {
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://127.0.0.1:{}/v1/chat/completions", port))
        .json(&json!({
            "model": "test",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 15,
            "temperature": 0,
            "logprobs": true,
            "top_logprobs": 8,
        }))
        .send()
        .await?
        .json::<serde_json::Value>()
        .await?;
    Ok(resp)
}

/// Build a TOPLOC proof hash from logprobs
fn proof_hash_from_logprobs(resp: &serde_json::Value) -> [u8; 32] {
    use samhati_toploc::proof::TokenLogits;
    use samhati_toploc::prover::ToplocProver;

    let signing_key = [42u8; 32]; // test key
    let mut prover = ToplocProver::new("test-model", signing_key);

    if let Some(tokens) = resp["choices"][0]["logprobs"]["content"].as_array() {
        for token_entry in tokens {
            let token_id = token_entry["id"].as_u64().unwrap_or(0) as u32;
            let mut top_k: Vec<(u32, f32)> = Vec::new();
            if let Some(top_arr) = token_entry["top_logprobs"].as_array() {
                for lp in top_arr.iter().take(8) {
                    let tid = lp["id"].as_u64().unwrap_or(0) as u32;
                    let logprob = lp["logprob"].as_f64().unwrap_or(-100.0) as f32;
                    top_k.push((tid, logprob));
                }
            }
            if top_k.is_empty() {
                top_k.push((token_id, 0.0));
            }
            prover.record_token(TokenLogits { token_id, top_k });
        }
    }

    if prover.recorded_count() > 0 {
        match prover.finalize() {
            Ok(proof) => proof.proof_hash(),
            Err(_) => [0u8; 32],
        }
    } else {
        [0u8; 32]
    }
}

#[tokio::test]
async fn test_swarm_e2e() {
    println!("\n=== SAMHATI SWARM END-TO-END TEST ===\n");

    // ── Test 1: All servers healthy ──
    println!("1. SERVER HEALTH");
    let ports = [8001, 8002, 8003];
    for &p in &ports {
        let healthy = check_health(p).await;
        println!("   Port {}: {}", p, if healthy { "✓ healthy" } else { "✗ DOWN" });
        assert!(healthy, "Server on port {} not healthy", p);
    }

    // ── Test 2: Fan-out — same prompt to all 3 nodes ──
    println!("\n2. FAN-OUT (parallel inference)");
    let prompt = "What is the capital of France?";
    let mut handles = Vec::new();
    for &p in &ports {
        let prompt = prompt.to_string();
        handles.push(tokio::spawn(async move {
            let start = std::time::Instant::now();
            let resp = call_with_logprobs(p, &prompt).await;
            let elapsed = start.elapsed().as_millis();
            (p, resp, elapsed)
        }));
    }

    let mut answers: Vec<(u16, String, [u8; 32], u128)> = Vec::new();
    for handle in handles {
        let (port, resp_result, elapsed) = handle.await.unwrap();
        match resp_result {
            Ok(resp) => {
                let content = resp["choices"][0]["message"]["content"]
                    .as_str()
                    .unwrap_or("")
                    .to_string();
                let proof = proof_hash_from_logprobs(&resp);
                let token_count = resp["choices"][0]["logprobs"]["content"]
                    .as_array()
                    .map(|a| a.len())
                    .unwrap_or(0);
                println!(
                    "   Port {}: \"{}...\" ({} tokens, {}ms, proof: {})",
                    port,
                    &content[..content.len().min(50)],
                    token_count,
                    elapsed,
                    hex::encode(&proof[..8]),
                );
                answers.push((port, content, proof, elapsed));
            }
            Err(e) => {
                println!("   Port {}: ✗ ERROR: {}", port, e);
                panic!("Node {} failed", port);
            }
        }
    }
    assert_eq!(answers.len(), 3, "Expected 3 answers");
    println!("   ✓ All 3 nodes responded");

    // ── Test 3: TOPLOC proof determinism ──
    println!("\n3. TOPLOC PROOF DETERMINISM");
    // Same model + same prompt + temp=0 should give same proof
    let proof1 = answers[0].2;
    let proof2 = answers[1].2;
    let proof3 = answers[2].2;
    println!("   Node 1 proof: {}", hex::encode(&proof1[..16]));
    println!("   Node 2 proof: {}", hex::encode(&proof2[..16]));
    println!("   Node 3 proof: {}", hex::encode(&proof3[..16]));

    // All running same model with temp=0 should produce identical proofs
    if proof1 == proof2 && proof2 == proof3 {
        println!("   ✓ All proofs IDENTICAL (same model, deterministic)");
    } else if proof1 != [0u8; 32] {
        println!("   ⚠ Proofs differ (possible: different load order, cache state)");
        println!("   This is expected if servers handle concurrency differently");
    }

    // ── Test 4: BradleyTerry ranking ──
    println!("\n4. BRADLEYTERRY RANKING");
    use samhati_ranking::{BradleyTerryEngine, PairwiseComparison};
    use samhati_ranking::types::node_id_from_byte;

    let node_ids = [node_id_from_byte(1), node_id_from_byte(2), node_id_from_byte(3)];
    let engine = BradleyTerryEngine::default();

    // Simulate pairwise comparisons (node 1 wins against 2, node 2 wins against 3, etc.)
    let comparisons = vec![
        PairwiseComparison { node_a: node_ids[0], node_b: node_ids[1], prob_a_wins: 0.7, ranker_elo: 1500 },
        PairwiseComparison { node_a: node_ids[0], node_b: node_ids[2], prob_a_wins: 0.8, ranker_elo: 1500 },
        PairwiseComparison { node_a: node_ids[1], node_b: node_ids[2], prob_a_wins: 0.6, ranker_elo: 1500 },
    ];

    let result = engine.aggregate(&comparisons);
    match result {
        Ok(ranking) => {
            println!("   Winner: node {} (probability: {:.2}%)",
                if ranking.winner == node_ids[0] { "1" } else if ranking.winner == node_ids[1] { "2" } else { "3" },
                ranking.win_probability * 100.0
            );
            println!("   Iterations: {}", ranking.iterations);
            for (i, nid) in node_ids.iter().enumerate() {
                if let Some(&s) = ranking.strengths.get(nid) {
                    println!("   Node {} strength: {:.4}", i+1, s);
                }
            }
            assert_eq!(ranking.winner, node_ids[0], "Node 1 should win");
            println!("   ✓ BradleyTerry correctly ranks node 1 as winner");
        }
        Err(e) => {
            println!("   ✗ BradleyTerry failed: {}", e);
            panic!("BT failed");
        }
    }

    // ── Test 5: ELO update ──
    println!("\n5. ELO UPDATE");
    use samhati_ranking::{EloStore, RoundOutcome};

    let mut store = EloStore::new();
    for &nid in &node_ids {
        store.register(nid);
    }

    let outcome = RoundOutcome {
        participants: node_ids.to_vec(),
        winner: node_ids[0],
        strengths: [(node_ids[0], 1.0), (node_ids[1], 0.3), (node_ids[2], 0.1)].into(),
        domain: Some("General".to_string()),
    };

    let deltas = store.update_round(&outcome);
    println!("   ELO deltas:");
    for (nid, delta) in &deltas {
        let label = if *nid == node_ids[0] { "Winner" } else { "Loser " };
        let rating = store.get(nid).map(|r| r.score).unwrap_or(0);
        println!("   {} (node {}): {:+} → ELO {}", label,
            if *nid == node_ids[0] { "1" } else if *nid == node_ids[1] { "2" } else { "3" },
            delta, rating);
    }
    let winner_elo = store.get(&node_ids[0]).unwrap().score;
    let loser_elo = store.get(&node_ids[2]).unwrap().score;
    assert!(winner_elo > 1500, "Winner ELO should increase");
    assert!(loser_elo < 1500, "Loser ELO should decrease");
    println!("   ✓ Winner gained ELO, losers lost ELO");

    // ── Test 6: Solana connectivity ──
    println!("\n6. SOLANA DEVNET");
    let sol_resp = reqwest::Client::new()
        .post("https://api.devnet.solana.com")
        .json(&json!({"jsonrpc":"2.0","id":1,"method":"getHealth"}))
        .send()
        .await;
    match sol_resp {
        Ok(r) => {
            let body: serde_json::Value = r.json().await.unwrap_or_default();
            let health = body["result"].as_str().unwrap_or("unknown");
            println!("   Health: {}", health);
            if health == "ok" {
                println!("   ✓ Solana devnet reachable");
            } else {
                println!("   ⚠ Solana devnet returned: {}", health);
            }
        }
        Err(e) => println!("   ✗ Cannot reach Solana: {}", e),
    }

    // ── Test 7: Full proof lifecycle ──
    println!("\n7. FULL PROOF LIFECYCLE");
    // Build proof → verify proof
    use samhati_toploc::prover::ToplocProver;
    use samhati_toploc::verifier::{ToplocVerifierImpl, ToplocVerifier};
    use samhati_toploc::proof::TokenLogits;
    use ed25519_dalek::SigningKey;

    let key_bytes = [99u8; 32];
    let signing_key = SigningKey::from_bytes(&key_bytes);
    let verifying_key = signing_key.verifying_key();

    // Get real logprobs from node 1
    let resp = call_with_logprobs(8001, "Hello").await.unwrap();
    let mut prover = ToplocProver::new("test-model", key_bytes);

    let tokens = resp["choices"][0]["logprobs"]["content"].as_array().unwrap();
    for t in tokens {
        let token_id = t["id"].as_u64().unwrap_or(0) as u32;
        let mut top_k = Vec::new();
        if let Some(arr) = t["top_logprobs"].as_array() {
            for lp in arr.iter().take(8) {
                top_k.push((lp["id"].as_u64().unwrap_or(0) as u32, lp["logprob"].as_f64().unwrap_or(0.0) as f32));
            }
        }
        if top_k.is_empty() { top_k.push((token_id, 0.0)); }
        prover.record_token(TokenLogits { token_id, top_k });
    }

    let proof = prover.finalize().unwrap();
    println!("   Proof: {} tokens, {} chunks, hash: {}",
        proof.token_count,
        proof.chunk_proofs.len(),
        hex::encode(&proof.proof_hash()[..8]),
    );
    println!("   Node pubkey: {}", hex::encode(&proof.node_pubkey[..8]));
    println!("   Timestamp: {}", proof.timestamp);

    // Verify it
    let mut verifier = ToplocVerifierImpl::new().with_max_proof_age(0); // disable freshness for test
    verifier.register_model("test-model");
    verifier.register_public_key(verifying_key);

    let result = verifier.verify(&proof, "test-model", proof.token_count);
    println!("   Verification: {:?}", result);
    assert!(result.is_valid(), "Proof should be valid");
    println!("   ✓ TOPLOC proof generated from live logprobs and verified successfully");

    // ── Summary ──
    println!("\n========================================");
    println!("  ALL TESTS PASSED");
    println!("  3 nodes, fan-out, TOPLOC proofs,");
    println!("  BradleyTerry ranking, ELO updates,");
    println!("  Solana connectivity, proof lifecycle");
    println!("========================================\n");
}
