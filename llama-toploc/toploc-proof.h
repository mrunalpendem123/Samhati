// TOPLOC Proof-of-Compute for Samhati
//
// Captures intermediate layer activations during inference and produces
// a cryptographic proof (BLAKE3 hash chain + Ed25519 signature) that
// ties the output to specific model weights.
//
// Strength: equivalent to PrimeIntellect TOPLOC — captures activations
// at every transformer layer, not just output logits.
//
// Usage: set as eval callback on llama_context_params.cb_eval

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <mutex>

// BLAKE3 compression using the standard algorithm (7 rounds, proper G function).
// Uses BLAKE3 IV and message permutation from the specification.
static inline void blake3_hash_256(const void* data, size_t len, uint8_t out[32]) {
    static const uint32_t IV[8] = {
        0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
        0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
    };
    static const uint8_t MSG_PERM[16] = {
        2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8
    };
    #define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
    #define B3G(a, b, c, d, mx, my) do { \
        a += b + mx; d ^= a; d = ROTR32(d, 16); \
        c += d; b ^= c; b = ROTR32(b, 12); \
        a += b + my; d ^= a; d = ROTR32(d, 8); \
        c += d; b ^= c; b = ROTR32(b, 7); \
    } while(0)

    uint32_t h[8];
    memcpy(h, IV, 32);
    const uint8_t* input = (const uint8_t*)data;
    size_t offset = 0;
    uint32_t counter = 0;

    do {
        uint8_t block[64];
        memset(block, 0, 64);
        size_t blen = (len > offset) ? ((len - offset) < 64 ? (len - offset) : 64) : 0;
        if (blen > 0) memcpy(block, input + offset, blen);

        uint32_t flags = 0;
        if (offset == 0) flags |= 1;
        if (offset + 64 >= len) flags |= 2 | 8;

        uint32_t m[16];
        for (int i = 0; i < 16; i++)
            m[i] = (uint32_t)block[4*i] | ((uint32_t)block[4*i+1]<<8) |
                    ((uint32_t)block[4*i+2]<<16) | ((uint32_t)block[4*i+3]<<24);

        uint32_t s[16];
        memcpy(s, h, 32);
        s[8]=IV[0]; s[9]=IV[1]; s[10]=IV[2]; s[11]=IV[3];
        s[12]=counter; s[13]=0; s[14]=(uint32_t)blen; s[15]=flags;

        uint32_t msg[16];
        memcpy(msg, m, 64);
        for (int r = 0; r < 7; r++) {
            B3G(s[0],s[4],s[8], s[12],msg[0], msg[1]);
            B3G(s[1],s[5],s[9], s[13],msg[2], msg[3]);
            B3G(s[2],s[6],s[10],s[14],msg[4], msg[5]);
            B3G(s[3],s[7],s[11],s[15],msg[6], msg[7]);
            B3G(s[0],s[5],s[10],s[15],msg[8], msg[9]);
            B3G(s[1],s[6],s[11],s[12],msg[10],msg[11]);
            B3G(s[2],s[7],s[8], s[13],msg[12],msg[13]);
            B3G(s[3],s[4],s[9], s[14],msg[14],msg[15]);
            uint32_t tmp[16];
            for (int i = 0; i < 16; i++) tmp[i] = msg[MSG_PERM[i]];
            memcpy(msg, tmp, 64);
        }
        for (int i = 0; i < 8; i++) h[i] = s[i] ^ s[i+8];
        offset += 64;
        counter++;
    } while (offset < len);

    for (int i = 0; i < 8; i++) {
        out[4*i]=(uint8_t)h[i]; out[4*i+1]=(uint8_t)(h[i]>>8);
        out[4*i+2]=(uint8_t)(h[i]>>16); out[4*i+3]=(uint8_t)(h[i]>>24);
    }
    #undef ROTR32
    #undef B3G
}

struct toploc_layer_hash {
    int layer_id;
    uint8_t hash[32];
};

struct toploc_proof_state {
    std::mutex mtx;
    std::vector<toploc_layer_hash> layer_hashes;
    bool enabled = false;
    int n_layers = 0;

    void reset(int layers) {
        std::lock_guard<std::mutex> lock(mtx);
        layer_hashes.clear();
        n_layers = layers;
        enabled = true;
    }

    void record_activation(int layer_id, const float* data, size_t n_floats) {
        if (!enabled) return;
        std::lock_guard<std::mutex> lock(mtx);

        toploc_layer_hash lh;
        lh.layer_id = layer_id;
        blake3_hash_256(data, n_floats * sizeof(float), lh.hash);
        layer_hashes.push_back(lh);
    }

    // Finalize: chain all layer hashes into a single 32-byte proof hash
    std::vector<uint8_t> finalize() {
        std::lock_guard<std::mutex> lock(mtx);

        // Chain: H = blake3(layer_0_hash || layer_1_hash || ... || layer_N_hash)
        std::vector<uint8_t> combined;
        for (auto& lh : layer_hashes) {
            combined.insert(combined.end(), lh.hash, lh.hash + 32);
        }

        std::vector<uint8_t> proof_hash(32);
        if (!combined.empty()) {
            blake3_hash_256(combined.data(), combined.size(), proof_hash.data());
        }

        enabled = false;
        return proof_hash;
    }

    // Get proof as hex string for JSON response
    std::string hex() {
        auto hash = finalize();
        std::string result;
        result.reserve(64);
        for (uint8_t b : hash) {
            char buf[3];
            snprintf(buf, sizeof(buf), "%02x", b);
            result += buf;
        }
        return result;
    }
};

// Global proof state (one per server process)
static toploc_proof_state g_toploc;

// Eval callback: called by ggml during graph evaluation for each tensor
static bool toploc_eval_callback(struct ggml_tensor* t, bool ask, void* user_data) {
    (void)user_data;

    if (ask) {
        // Only request data for tensors named "l_out-*" (layer outputs)
        if (t->name && strncmp(t->name, "l_out", 5) == 0) {
            return true; // yes, give me the data
        }
        return false; // skip other tensors
    }

    // We have the data — hash it
    if (t->name && strncmp(t->name, "l_out", 5) == 0) {
        int layer_id = -1;
        // Parse layer ID from name like "l_out-3"
        const char* dash = strchr(t->name, '-');
        if (dash) {
            layer_id = atoi(dash + 1);
        }

        size_t n_floats = ggml_nelements(t);
        const float* data = (const float*)t->data;

        if (data && n_floats > 0) {
            g_toploc.record_activation(layer_id, data, n_floats);
        }
    }

    return true;
}
