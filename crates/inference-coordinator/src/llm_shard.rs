//! Real Llama-style distributed transformer shard using Burn.
//!
//! A `LlamaShard` loads the transformer layers it owns from a HuggingFace
//! safetensors file (standard LlamaForCausalLM weight naming) and exposes a
//! single `forward()` call that processes activation tensors through those
//! layers, updating the session KV cache in place.
//!
//! # Weight-file naming convention
//!
//! ```text
//! — First shard (layer_start == 0):
//!     model.embed_tokens.weight            [vocab_size, hidden_size]
//!
//! — Every layer L in [layer_start, layer_end):
//!     model.layers.{L}.input_layernorm.weight           [hidden_size]
//!     model.layers.{L}.self_attn.q_proj.weight          [n_heads*head_dim, hidden_size]
//!     model.layers.{L}.self_attn.k_proj.weight          [n_kv_heads*head_dim, hidden_size]
//!     model.layers.{L}.self_attn.v_proj.weight          [n_kv_heads*head_dim, hidden_size]
//!     model.layers.{L}.self_attn.o_proj.weight          [hidden_size, n_heads*head_dim]
//!     model.layers.{L}.post_attention_layernorm.weight  [hidden_size]
//!     model.layers.{L}.mlp.gate_proj.weight             [intermediate_size, hidden_size]
//!     model.layers.{L}.mlp.up_proj.weight               [intermediate_size, hidden_size]
//!     model.layers.{L}.mlp.down_proj.weight             [hidden_size, intermediate_size]
//!
//! — Last shard (layer_end == total_layers):
//!     model.norm.weight                                 [hidden_size]
//!     lm_head.weight                                    [vocab_size, hidden_size]
//! ```

use anyhow::{anyhow, Result};
use burn::module::{ConstantRecord, Param};
use burn::nn::{
    Embedding, EmbeddingConfig, EmbeddingRecord,
    Linear, LinearConfig, LinearRecord,
    RmsNorm, RmsNormConfig, RmsNormRecord,
};
use burn::prelude::Module;
use burn::tensor::{activation, backend::Backend, Int, Tensor, TensorData};
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::kv_cache::{LayerKv, SessionKv};

// ── Config ────────────────────────────────────────────────────────────────────

/// Architecture parameters for one shard of a Llama-style model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaShardConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    /// For GQA (grouped-query attention).  Set equal to `num_attention_heads`
    /// for standard MHA (Llama-1, Llama-2 7B).
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_seq_len: usize,
    /// First transformer layer index owned by this shard (0-indexed).
    pub layer_start: usize,
    /// One-past-the-last layer index owned by this shard.
    pub layer_end: usize,
    /// Total layers in the full model (needed to determine final-shard status).
    pub total_layers: usize,
}

impl LlamaShardConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
    pub fn n_local_layers(&self) -> usize {
        self.layer_end - self.layer_start
    }
    pub fn is_first(&self) -> bool {
        self.layer_start == 0
    }
    pub fn is_last(&self) -> bool {
        self.layer_end >= self.total_layers
    }
}

// ── Weight loading helpers ────────────────────────────────────────────────────

/// Read f32 data from a safetensors view.
fn load_f32(view: &safetensors::tensor::TensorView<'_>) -> Vec<f32> {
    view.data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn load_linear_no_bias<B: Backend>(
    st: &safetensors::SafeTensors<'_>,
    name: &str,
    in_dim: usize,
    out_dim: usize,
    device: &B::Device,
) -> Result<Linear<B>> {
    let view = st
        .tensor(name)
        .map_err(|e| anyhow!("missing tensor {name}: {e:?}"))?;
    let f32_data = load_f32(&view);
    let weight = Tensor::<B, 2>::from_data(
        TensorData::new(f32_data, [out_dim, in_dim]),
        device,
    );
    let record = LinearRecord {
        weight: Param::from_tensor(weight),
        bias: None,
    };
    Ok(LinearConfig::new(in_dim, out_dim)
        .with_bias(false)
        .init::<B>(device)
        .load_record(record))
}

fn load_rms_norm<B: Backend>(
    st: &safetensors::SafeTensors<'_>,
    name: &str,
    size: usize,
    eps: f64,
    device: &B::Device,
) -> Result<RmsNorm<B>> {
    let view = st
        .tensor(name)
        .map_err(|e| anyhow!("missing tensor {name}: {e:?}"))?;
    let f32_data = load_f32(&view);
    let weight = Tensor::<B, 1>::from_data(
        TensorData::new(f32_data, [size]),
        device,
    );
    let record = RmsNormRecord {
        gamma: Param::from_tensor(weight),
        epsilon: ConstantRecord::new(),
    };
    Ok(RmsNormConfig::new(size)
        .with_epsilon(eps)
        .init::<B>(device)
        .load_record(record))
}

// ── RoPE helpers ──────────────────────────────────────────────────────────────

fn rotate_half<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, h, s, d] = x.dims();
    let half = d / 2;
    let x1 = x.clone().slice([0..b, 0..h, 0..s, 0..half]);
    let x2 = x.slice([0..b, 0..h, 0..s, half..d]);
    Tensor::cat(vec![x2.neg(), x1], 3)
}

fn apply_rope<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    seq_offset: usize,
    head_dim: usize,
    theta: f64,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [_b, _h, seq_len, _hd] = q.dims();
    let half = head_dim / 2;

    let freqs: Vec<f32> = (0..half)
        .map(|i| 1.0f32 / (theta as f32).powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    let freqs = Tensor::<B, 4>::from_data(
        TensorData::new(freqs, [1usize, 1, 1, half]),
        device,
    );

    let pos: Vec<f32> = (seq_offset..seq_offset + seq_len)
        .map(|p| p as f32)
        .collect();
    let pos = Tensor::<B, 4>::from_data(
        TensorData::new(pos, [1usize, 1, seq_len, 1]),
        device,
    );

    let angles = pos.mul(freqs); // [1, 1, seq_len, half]
    let cos_half = angles.clone().cos();
    let sin_half = angles.sin();

    // Expand to full head_dim
    let cos = Tensor::cat(vec![cos_half.clone(), cos_half], 3); // [1, 1, seq_len, head_dim]
    let sin = Tensor::cat(vec![sin_half.clone(), sin_half], 3);

    let q_rot = q.clone().mul(cos.clone()).add(rotate_half(q).mul(sin.clone()));
    let k_rot = k.clone().mul(cos).add(rotate_half(k).mul(sin));
    (q_rot, k_rot)
}

// ── GQA helpers ───────────────────────────────────────────────────────────────

/// Repeat KV heads to match the query head count for GQA.
///
/// Equivalent to `torch.repeat_interleave(x, n_rep, dim=1)`:
/// each KV head is repeated `n_rep` times contiguously so that query head
/// groups align with their corresponding KV head.
///
/// `x`: `[batch, n_kv_heads, seq, head_dim]` → `[batch, n_heads, seq, head_dim]`
///
/// Example (n_kv_heads=2, n_rep=2):
///   input heads : [h0, h1]
///   output heads: [h0, h0, h1, h1]   ← each head repeated, then next head
fn repeat_kv<B: Backend>(x: Tensor<B, 4>, n_rep: usize) -> Tensor<B, 4> {
    if n_rep == 1 {
        return x;
    }
    let [b, n_kv_heads, s, d] = x.dims();
    let mut chunks: Vec<Tensor<B, 4>> = Vec::with_capacity(n_kv_heads * n_rep);
    for h in 0..n_kv_heads {
        let head = x.clone().slice([0..b, h..h + 1, 0..s, 0..d]);
        for _ in 0..n_rep {
            chunks.push(head.clone());
        }
    }
    Tensor::cat(chunks, 1)
}

// ── Attention layer ───────────────────────────────────────────────────────────

struct LlamaAttention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_theta: f64,
}

impl<B: Backend> LlamaAttention<B> {
    fn load(
        st: &safetensors::SafeTensors<'_>,
        cfg: &LlamaShardConfig,
        layer_idx: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let hd = cfg.head_dim();
        let prefix = format!("model.layers.{layer_idx}.self_attn");
        Ok(Self {
            q_proj: load_linear_no_bias::<B>(
                st,
                &format!("{prefix}.q_proj.weight"),
                cfg.hidden_size,
                cfg.num_attention_heads * hd,
                device,
            )?,
            k_proj: load_linear_no_bias::<B>(
                st,
                &format!("{prefix}.k_proj.weight"),
                cfg.hidden_size,
                cfg.num_key_value_heads * hd,
                device,
            )?,
            v_proj: load_linear_no_bias::<B>(
                st,
                &format!("{prefix}.v_proj.weight"),
                cfg.hidden_size,
                cfg.num_key_value_heads * hd,
                device,
            )?,
            o_proj: load_linear_no_bias::<B>(
                st,
                &format!("{prefix}.o_proj.weight"),
                cfg.num_attention_heads * hd,
                cfg.hidden_size,
                device,
            )?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: hd,
            rope_theta: cfg.rope_theta,
        })
    }

    fn forward(
        &self,
        x: Tensor<B, 3>,
        seq_offset: usize,
        kv: &mut LayerKv<B>,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let [b, seq, _h] = x.dims();

        // Project
        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        // Reshape → [batch, heads, seq, head_dim]
        let q: Tensor<B, 4> = q
            .reshape([b, seq, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k: Tensor<B, 4> = k
            .reshape([b, seq, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v: Tensor<B, 4> = v
            .reshape([b, seq, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        // Apply RoPE
        let (q, k) = apply_rope(q, k, seq_offset, self.head_dim, self.rope_theta, device);

        // Update KV cache
        let (k_full, v_full) = kv.append(k, v);

        // GQA expansion
        let n_rep = self.num_heads / self.num_kv_heads;
        let k_exp = repeat_kv(k_full, n_rep);
        let v_exp = repeat_kv(v_full, n_rep);

        // Scaled dot-product attention: [b, heads, seq, total_seq]
        let scale = (self.head_dim as f32).sqrt();
        let attn_w = q.matmul(k_exp.swap_dims(2, 3)).div_scalar(scale);

        // Causal mask (only during prefill when seq > 1)
        let total_seq = attn_w.dims()[3];
        let attn_w = if seq > 1 {
            let past = total_seq - seq;
            let mut mask_data = vec![0.0f32; seq * total_seq];
            for i in 0..seq {
                for j in (past + i + 1)..total_seq {
                    mask_data[i * total_seq + j] = f32::NEG_INFINITY;
                }
            }
            let mask = Tensor::<B, 4>::from_data(
                TensorData::new(mask_data, [1usize, 1, seq, total_seq]),
                device,
            );
            attn_w.add(mask)
        } else {
            attn_w
        };

        let attn_p = activation::softmax(attn_w, 3);
        let out = attn_p.matmul(v_exp); // [b, heads, seq, head_dim]

        // Reshape back → [b, seq, hidden]
        let out: Tensor<B, 3> = out.swap_dims(1, 2).reshape([b, seq, self.num_heads * self.head_dim]);
        self.o_proj.forward(out)
    }
}

// ── MLP ───────────────────────────────────────────────────────────────────────

struct LlamaMlp<B: Backend> {
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
}

impl<B: Backend> LlamaMlp<B> {
    fn load(
        st: &safetensors::SafeTensors<'_>,
        cfg: &LlamaShardConfig,
        layer_idx: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let prefix = format!("model.layers.{layer_idx}.mlp");
        Ok(Self {
            gate_proj: load_linear_no_bias::<B>(
                st,
                &format!("{prefix}.gate_proj.weight"),
                cfg.hidden_size,
                cfg.intermediate_size,
                device,
            )?,
            up_proj: load_linear_no_bias::<B>(
                st,
                &format!("{prefix}.up_proj.weight"),
                cfg.hidden_size,
                cfg.intermediate_size,
                device,
            )?,
            down_proj: load_linear_no_bias::<B>(
                st,
                &format!("{prefix}.down_proj.weight"),
                cfg.intermediate_size,
                cfg.hidden_size,
                device,
            )?,
        })
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // SwiGLU: down(silu(gate(x)) * up(x))
        let gate = activation::silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        let activated = gate.mul(up);
        self.down_proj.forward(activated)
    }
}

// ── Decoder layer ─────────────────────────────────────────────────────────────

struct DecoderLayer<B: Backend> {
    input_norm: RmsNorm<B>,
    attn: LlamaAttention<B>,
    post_attn_norm: RmsNorm<B>,
    mlp: LlamaMlp<B>,
}

impl<B: Backend> DecoderLayer<B> {
    fn load(
        st: &safetensors::SafeTensors<'_>,
        cfg: &LlamaShardConfig,
        layer_idx: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let prefix = format!("model.layers.{layer_idx}");
        Ok(Self {
            input_norm: load_rms_norm::<B>(
                st,
                &format!("{prefix}.input_layernorm.weight"),
                cfg.hidden_size,
                cfg.rms_norm_eps,
                device,
            )?,
            attn: LlamaAttention::load(st, cfg, layer_idx, device)?,
            post_attn_norm: load_rms_norm::<B>(
                st,
                &format!("{prefix}.post_attention_layernorm.weight"),
                cfg.hidden_size,
                cfg.rms_norm_eps,
                device,
            )?,
            mlp: LlamaMlp::load(st, cfg, layer_idx, device)?,
        })
    }

    fn forward(
        &self,
        x: Tensor<B, 3>,
        seq_offset: usize,
        kv: &mut LayerKv<B>,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        // Pre-norm → attention → residual
        let h = self.input_norm.forward(x.clone());
        let h = self.attn.forward(h, seq_offset, kv, device);
        let x = x.add(h);

        // Pre-norm → MLP → residual
        let h = self.post_attn_norm.forward(x.clone());
        let h = self.mlp.forward(h);
        x.add(h)
    }
}

// ── Shard ─────────────────────────────────────────────────────────────────────

/// A contiguous slice of transformer layers loaded from disk.
pub struct LlamaShard<B: Backend> {
    cfg: LlamaShardConfig,
    embed: Option<Embedding<B>>,    // only on first shard
    layers: Vec<DecoderLayer<B>>,
    final_norm: Option<RmsNorm<B>>, // only on last shard
    lm_head: Option<Linear<B>>,    // only on last shard
    device: B::Device,
}

impl<B: Backend> LlamaShard<B> {
    /// Load a shard from one or more safetensors files.
    ///
    /// `weight_files` should list all `.safetensors` shards that together
    /// contain the weights for `[cfg.layer_start, cfg.layer_end)`.
    pub fn load(
        weight_files: &[impl AsRef<Path>],
        cfg: LlamaShardConfig,
        device: B::Device,
    ) -> Result<Self> {
        // Merge all safetensors files into one in-memory byte buffer, then
        // open the first file for tensor access.  For simplicity we load
        // tensors from each file in sequence, delegating to the first file
        // that contains the requested tensor name.
        let bytes_vec: Vec<Vec<u8>> = weight_files
            .iter()
            .map(|p| std::fs::read(p.as_ref()))
            .collect::<std::io::Result<_>>()?;

        // We use a helper that tries each file until the tensor is found.
        let load_from_files = |tensor_name: &str| -> Result<Vec<f32>> {
            for bytes in &bytes_vec {
                let st = safetensors::SafeTensors::deserialize(bytes)
                    .map_err(|e| anyhow!("safetensors deserialize: {e:?}"))?;
                if let Ok(view) = st.tensor(tensor_name) {
                    return Ok(load_f32(&view));
                }
            }
            Err(anyhow!("tensor not found in any weight file: {tensor_name}"))
        };

        let load_shape = |tensor_name: &str| -> Result<Vec<usize>> {
            for bytes in &bytes_vec {
                let st = safetensors::SafeTensors::deserialize(bytes)
                    .map_err(|e| anyhow!("safetensors deserialize: {e:?}"))?;
                if let Ok(view) = st.tensor(tensor_name) {
                    return Ok(view.shape().to_vec());
                }
            }
            Err(anyhow!("tensor not found in any weight file: {tensor_name}"))
        };

        // Helper: build a Linear (no bias) from raw f32 data.
        let mk_linear = |name: &str, in_d: usize, out_d: usize| -> Result<Linear<B>> {
            let data = load_from_files(name)?;
            let weight = Tensor::<B, 2>::from_data(
                TensorData::new(data, [out_d, in_d]),
                &device,
            );
            let record = LinearRecord {
                weight: Param::from_tensor(weight),
                bias: None,
            };
            Ok(LinearConfig::new(in_d, out_d)
                .with_bias(false)
                .init::<B>(&device)
                .load_record(record))
        };

        // Helper: build an RmsNorm from raw f32 data.
        let mk_rms = |name: &str, size: usize| -> Result<RmsNorm<B>> {
            let data = load_from_files(name)?;
            let weight =
                Tensor::<B, 1>::from_data(TensorData::new(data, [size]), &device);
            let record = RmsNormRecord {
                gamma: Param::from_tensor(weight),
                epsilon: ConstantRecord::new(),
            };
            Ok(RmsNormConfig::new(size)
                .with_epsilon(cfg.rms_norm_eps)
                .init::<B>(&device)
                .load_record(record))
        };

        // Embedding (first shard only)
        let embed = if cfg.is_first() {
            let data = load_from_files("model.embed_tokens.weight")?;
            let shape = load_shape("model.embed_tokens.weight")?;
            let (vs, hs) = (shape[0], shape[1]);
            let weight = Tensor::<B, 2>::from_data(TensorData::new(data, [vs, hs]), &device);
            let record = EmbeddingRecord {
                weight: Param::from_tensor(weight),
            };
            Some(
                EmbeddingConfig::new(vs, hs)
                    .init::<B>(&device)
                    .load_record(record),
            )
        } else {
            None
        };

        // Decoder layers
        let mut layers = Vec::with_capacity(cfg.n_local_layers());
        for l in cfg.layer_start..cfg.layer_end {
            let p = format!("model.layers.{l}");
            let hd = cfg.head_dim();
            let input_norm = mk_rms(&format!("{p}.input_layernorm.weight"), cfg.hidden_size)?;
            let q_proj = mk_linear(
                &format!("{p}.self_attn.q_proj.weight"),
                cfg.hidden_size,
                cfg.num_attention_heads * hd,
            )?;
            let k_proj = mk_linear(
                &format!("{p}.self_attn.k_proj.weight"),
                cfg.hidden_size,
                cfg.num_key_value_heads * hd,
            )?;
            let v_proj = mk_linear(
                &format!("{p}.self_attn.v_proj.weight"),
                cfg.hidden_size,
                cfg.num_key_value_heads * hd,
            )?;
            let o_proj = mk_linear(
                &format!("{p}.self_attn.o_proj.weight"),
                cfg.num_attention_heads * hd,
                cfg.hidden_size,
            )?;
            let post_attn_norm =
                mk_rms(&format!("{p}.post_attention_layernorm.weight"), cfg.hidden_size)?;
            let gate_proj = mk_linear(
                &format!("{p}.mlp.gate_proj.weight"),
                cfg.hidden_size,
                cfg.intermediate_size,
            )?;
            let up_proj = mk_linear(
                &format!("{p}.mlp.up_proj.weight"),
                cfg.hidden_size,
                cfg.intermediate_size,
            )?;
            let down_proj = mk_linear(
                &format!("{p}.mlp.down_proj.weight"),
                cfg.intermediate_size,
                cfg.hidden_size,
            )?;
            layers.push(DecoderLayer {
                input_norm,
                attn: LlamaAttention {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    num_heads: cfg.num_attention_heads,
                    num_kv_heads: cfg.num_key_value_heads,
                    head_dim: hd,
                    rope_theta: cfg.rope_theta,
                },
                post_attn_norm,
                mlp: LlamaMlp {
                    gate_proj,
                    up_proj,
                    down_proj,
                },
            });
        }

        // Final norm + lm_head (last shard only)
        let (final_norm, lm_head) = if cfg.is_last() {
            let norm = mk_rms("model.norm.weight", cfg.hidden_size)?;
            let head = mk_linear("lm_head.weight", cfg.hidden_size, cfg.vocab_size)?;
            (Some(norm), Some(head))
        } else {
            (None, None)
        };

        Ok(Self { cfg, embed, layers, final_norm, lm_head, device })
    }

    pub fn config(&self) -> &LlamaShardConfig {
        &self.cfg
    }

    pub fn device(&self) -> &B::Device {
        &self.device
    }

    /// Forward pass through all transformer layers owned by this shard.
    ///
    /// # Arguments
    /// * `input_frame`  — If this is the first shard, contains token IDs as
    ///   `f32` of shape `[1, seq_len]`.  Otherwise, hidden states of shape
    ///   `[1, seq_len, hidden_size]`.
    /// * `session`      — Mutable reference to the session's KV cache.
    ///
    /// # Returns
    /// * If NOT the last shard: hidden states `[1, seq_len, hidden_size]`.
    /// * If the last shard:     sampled next-token ID as `[1, 1]` (single f32).
    pub fn forward(
        &self,
        input_frame: &crate::tensor_frame::TensorFrame,
        session: &mut SessionKv<B>,
    ) -> Result<crate::tensor_frame::TensorFrame> {
        let seq_offset = input_frame.seq_offset;

        // 1. Convert input to hidden states [1, seq_len, hidden_size]
        let mut hidden: Tensor<B, 3> = if let Some(embed) = &self.embed {
            // First shard: input is token IDs [1, seq_len] stored as f32
            let token_ids_f32 = input_frame.to_f32_vec()?;
            let seq_len = input_frame.shape.last().copied().unwrap_or(1);
            let ids: Vec<i32> = token_ids_f32.iter().map(|&f| f as i32).collect();
            let id_tensor =
                Tensor::<B, 2, Int>::from_data(TensorData::new(ids, [1usize, seq_len]), &self.device);
            embed.forward(id_tensor) // [1, seq_len, hidden_size]
        } else {
            // Middle / last shard: input is already hidden states
            input_frame.to_burn_3d(&self.device)?
        };

        // 2. Run decoder layers
        for (local_idx, layer) in self.layers.iter().enumerate() {
            let kv = &mut session.layers[local_idx];
            hidden = layer.forward(hidden, seq_offset, kv, &self.device);
        }

        // Update session seq position
        let new_tokens = input_frame.shape.last().copied().unwrap_or(1);
        session.seq_pos += new_tokens;

        // 3. Produce output
        if let (Some(norm), Some(head)) = (&self.final_norm, &self.lm_head) {
            // Last shard: hidden → logits → argmax → single token ID
            let normed = norm.forward(hidden);
            let [_b, seq, hidden_sz] = normed.dims();

            // Take only the last position for next-token prediction → [1, hidden_size]
            let last_hidden: Tensor<B, 2> =
                normed.slice([0..1usize, seq - 1..seq, 0..hidden_sz]).reshape([1, hidden_sz]);

            let logits = head.forward(last_hidden); // [1, vocab_size]

            // Greedy argmax via CPU extraction (avoids Int tensor type complexity)
            let logits_data = logits.into_data();
            let logit_vals: Vec<f32> = logits_data
                .to_vec()
                .map_err(|e| anyhow!("logits to_vec: {e:?}"))?;
            let next_id = logit_vals
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as f32)
                .unwrap_or(0.0);

            Ok(crate::tensor_frame::TensorFrame::from_f32(
                &[next_id],
                vec![1, 1],
                session.seq_pos,
            ))
        } else {
            // Pass hidden states to next shard
            crate::tensor_frame::TensorFrame::from_burn(hidden, session.seq_pos)
        }
    }
}

// ── Convenience constructors for the default WGPU device ─────────────────────

impl LlamaShard<crate::InferenceBackend> {
    /// Load from one or more safetensors files using the best available WGPU
    /// device.  This is the primary entry point for production use.
    pub fn load_wgpu(
        weight_files: &[impl AsRef<Path>],
        cfg: LlamaShardConfig,
    ) -> Result<Self> {
        use burn::backend::wgpu::WgpuDevice;
        Self::load(weight_files, cfg, WgpuDevice::BestAvailable)
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────
//
// Run with: cargo test -p inference-coordinator --features burn
//
// All tests use the CPU NdArray backend so they require no GPU and run fast.
// They validate `rotate_half`, `apply_rope`, and `repeat_kv` against
// hand-computed reference values derived from the standard Llama RoPE / GQA
// formulae.

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::NdArray;
    use burn::tensor::{Tensor, TensorData};

    type TB = NdArray;

    fn cpu() -> NdArrayDevice {
        NdArrayDevice::Cpu
    }

    /// Extract flat f32 values from a tensor (row-major / C-order).
    fn to_f32(t: Tensor<TB, 4>) -> Vec<f32> {
        t.into_data().to_vec().unwrap()
    }

    fn assert_close(got: &[f32], expected: &[f32], eps: f32, label: &str) {
        assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < eps,
                "{label}[{i}]: got {g:.7}, expected {e:.7} (eps={eps})"
            );
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // rotate_half
    //
    // Definition: rotate_half([x1 | x2]) = [-x2 | x1]
    // where x1 is the first half of the last dimension and x2 the second half.
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_rotate_half_basic() {
        // Shape [1, 1, 1, 4]:  x1=[1,2]  x2=[3,4]
        // Expected:            [-3, -4, 1, 2]
        let d = cpu();
        let x = Tensor::<TB, 4>::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [1, 1, 1, 4]),
            &d,
        );
        let got = to_f32(rotate_half(x));
        assert_close(&got, &[-3.0, -4.0, 1.0, 2.0], 1e-6, "rotate_half basic");
    }

    #[test]
    fn test_rotate_half_two_heads() {
        // Shape [1, 2, 1, 4]:
        //   head 0: [1, 2, 3, 4] → [-3, -4, 1, 2]
        //   head 1: [5, 6, 7, 8] → [-7, -8, 5, 6]
        let d = cpu();
        let x = Tensor::<TB, 4>::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [1, 2, 1, 4]),
            &d,
        );
        let got = to_f32(rotate_half(x));
        assert_close(
            &got,
            &[-3.0, -4.0, 1.0, 2.0, -7.0, -8.0, 5.0, 6.0],
            1e-6,
            "rotate_half two heads",
        );
    }

    #[test]
    fn test_rotate_half_applied_twice_negates() {
        // rotate_half(rotate_half(x)) == -x for any x.
        // Proof: rotate_half([-x2|x1]) = [-x1|-x2] = -(x1|x2) = -x
        let d = cpu();
        let data = vec![1.0f32, -2.0, 3.0, -4.0];
        let x = Tensor::<TB, 4>::from_data(
            TensorData::new(data.clone(), [1, 1, 1, 4]),
            &d,
        );
        let got = to_f32(rotate_half(rotate_half(x)));
        let expected: Vec<f32> = data.iter().map(|v| -v).collect();
        assert_close(&got, &expected, 1e-6, "rotate_half twice → negation");
    }

    // ────────────────────────────────────────────────────────────────────────
    // apply_rope
    //
    // Standard Llama RoPE with rotate-half style:
    //   freqs[i]  = 1 / theta^(2i / head_dim)   for i in [0, head_dim/2)
    //   angle[p,i] = position_p * freqs[i]
    //   cos_full = [cos(angle[p,0]), …, cos(angle[p,half-1]),
    //               cos(angle[p,0]), …, cos(angle[p,half-1])]   (duplicated)
    //   sin_full  = (same pattern with sin)
    //   x_rot     = x * cos_full + rotate_half(x) * sin_full
    //
    // At position 0 every angle is 0 → cos=1, sin=0 → identity.
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_apply_rope_position_zero_is_identity() {
        // At position 0 all angles are 0, so the rotation is the identity.
        let d = cpu();
        let q_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let k_data = vec![5.0f32, 6.0, 7.0, 8.0];
        let q = Tensor::<TB, 4>::from_data(
            TensorData::new(q_data.clone(), [1, 1, 1, 4]),
            &d,
        );
        let k = Tensor::<TB, 4>::from_data(
            TensorData::new(k_data.clone(), [1, 1, 1, 4]),
            &d,
        );
        let (q_rot, k_rot) = apply_rope(q, k, /*seq_offset=*/0, /*head_dim=*/4, /*theta=*/10000.0, &d);
        assert_close(&to_f32(q_rot), &q_data, 1e-6, "rope pos=0 q");
        assert_close(&to_f32(k_rot), &k_data, 1e-6, "rope pos=0 k");
    }

    #[test]
    fn test_apply_rope_single_token_known_values() {
        // head_dim=4, half=2, theta=10000, token at position 1 (seq_offset=1).
        //
        // freqs = [1/(10000^(0/4)), 1/(10000^(2/4))] = [1.0, 0.01]
        // angles at pos 1 = [1.0, 0.01]
        // cos = [cos(1), cos(0.01), cos(1), cos(0.01)]
        // sin = [sin(1), sin(0.01), sin(1), sin(0.01)]
        //
        // q = [1, 0, 0, 0]
        //   rotate_half(q) = [0, 0, 1, 0]
        //   q_rot = [cos(1), 0, sin(1), 0]
        //
        // k = [0, 1, 0, 0]
        //   rotate_half(k) = [0, 0, 0, 1]
        //   k_rot = [0, cos(0.01), 0, sin(0.01)]
        let d = cpu();
        let q = Tensor::<TB, 4>::from_data(
            TensorData::new(vec![1.0f32, 0.0, 0.0, 0.0], [1, 1, 1, 4]),
            &d,
        );
        let k = Tensor::<TB, 4>::from_data(
            TensorData::new(vec![0.0f32, 1.0, 0.0, 0.0], [1, 1, 1, 4]),
            &d,
        );
        let (q_rot, k_rot) = apply_rope(q, k, /*seq_offset=*/1, 4, 10000.0, &d);

        let cos1: f32 = 1.0_f32.cos();
        let sin1: f32 = 1.0_f32.sin();
        let cos001: f32 = 0.01_f32.cos();
        let sin001: f32 = 0.01_f32.sin();

        assert_close(&to_f32(q_rot), &[cos1, 0.0, sin1, 0.0],       1e-5, "rope single q");
        assert_close(&to_f32(k_rot), &[0.0, cos001, 0.0, sin001],    1e-5, "rope single k");
    }

    #[test]
    fn test_apply_rope_multi_token_sequence() {
        // Two tokens, seq_offset=0: positions are [0, 1].
        // Token 0 (pos 0) → identity.
        // Token 1 (pos 1) → same rotation as single-token test above.
        //
        // q shape: [1, 1, 2, 4], both tokens = [1, 0, 0, 0]
        // Expected q_rot: [[1,0,0,0], [cos(1),0,sin(1),0]]
        let d = cpu();
        let q_data = vec![1.0f32, 0.0, 0.0, 0.0,  // token 0
                          1.0,    0.0, 0.0, 0.0];  // token 1
        let k_data = vec![0.0f32, 1.0, 0.0, 0.0,  // token 0
                          0.0,    1.0, 0.0, 0.0];  // token 1
        let q = Tensor::<TB, 4>::from_data(TensorData::new(q_data, [1, 1, 2, 4]), &d);
        let k = Tensor::<TB, 4>::from_data(TensorData::new(k_data, [1, 1, 2, 4]), &d);

        let (q_rot, k_rot) = apply_rope(q, k, /*seq_offset=*/0, 4, 10000.0, &d);

        let cos1:   f32 = 1.0_f32.cos();
        let sin1:   f32 = 1.0_f32.sin();
        let cos001: f32 = 0.01_f32.cos();
        let sin001: f32 = 0.01_f32.sin();

        // token 0: identity; token 1: rotated
        let q_expected = vec![
            1.0, 0.0, 0.0, 0.0,           // pos 0 – identity
            cos1, 0.0, sin1, 0.0,          // pos 1
        ];
        let k_expected = vec![
            0.0, 1.0, 0.0, 0.0,           // pos 0 – identity
            0.0, cos001, 0.0, sin001,      // pos 1
        ];
        assert_close(&to_f32(q_rot), &q_expected, 1e-5, "rope multi q");
        assert_close(&to_f32(k_rot), &k_expected, 1e-5, "rope multi k");
    }

    // ────────────────────────────────────────────────────────────────────────
    // repeat_kv  (GQA head expansion)
    //
    // Correct semantics: each KV head is repeated n_rep times contiguously,
    // matching torch.repeat_interleave(x, n_rep, dim=1).
    //
    //   n_kv_heads=2, n_rep=2  →  [h0, h0, h1, h1]
    //
    // The previous (buggy) implementation concatenated n_rep full copies of x:
    //   →  [h0, h1, h0, h1]   — wrong; query head groups would attend the
    //      wrong KV heads in GQA models (Llama-3+, Mistral, etc.).
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_repeat_kv_n_rep_1_is_passthrough() {
        let d = cpu();
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let x = Tensor::<TB, 4>::from_data(TensorData::new(data.clone(), [1, 2, 1, 2]), &d);
        assert_close(&to_f32(repeat_kv(x, 1)), &data, 1e-6, "repeat_kv n_rep=1");
    }

    #[test]
    fn test_repeat_kv_gqa_two_heads_n_rep_2() {
        // n_kv_heads=2, n_rep=2 → n_heads=4
        // head 0 = [1, 2]  (query heads 0 and 1 attend this KV head)
        // head 1 = [3, 4]  (query heads 2 and 3 attend this KV head)
        //
        // Correct output (repeat_interleave): [h0, h0, h1, h1]
        //   flat: [1, 2, 1, 2, 3, 4, 3, 4]
        //
        // Previous buggy output (repeat tile): [h0, h1, h0, h1]
        //   flat: [1, 2, 3, 4, 1, 2, 3, 4]   ← wrong KV alignment
        let d = cpu();
        let x = Tensor::<TB, 4>::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [1, 2, 1, 2]),
            &d,
        );
        let out = repeat_kv(x, 2);
        assert_eq!(out.dims(), [1, 4, 1, 2], "repeat_kv shape");
        assert_close(
            &to_f32(out),
            &[1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0],
            1e-6,
            "repeat_kv GQA 2×2",
        );
    }

    #[test]
    fn test_repeat_kv_three_kv_heads_n_rep_4() {
        // n_kv_heads=3, n_rep=4 → n_heads=12
        // Verify that each KV head appears exactly n_rep times consecutively.
        let d = cpu();
        // head_dim=1 for simplicity: head 0=10, head 1=20, head 2=30
        let x = Tensor::<TB, 4>::from_data(
            TensorData::new(vec![10.0f32, 20.0, 30.0], [1, 3, 1, 1]),
            &d,
        );
        let out = repeat_kv(x, 4);
        assert_eq!(out.dims(), [1, 12, 1, 1]);
        let got = to_f32(out);
        // Expected: [10,10,10,10, 20,20,20,20, 30,30,30,30]
        let expected: Vec<f32> = [10.0f32; 4]
            .iter()
            .chain([20.0f32; 4].iter())
            .chain([30.0f32; 4].iter())
            .copied()
            .collect();
        assert_close(&got, &expected, 1e-6, "repeat_kv 3 heads × 4");
    }

    #[test]
    fn test_repeat_kv_output_shape() {
        // Generic shape check: [batch=2, n_kv=4, seq=7, head_dim=8], n_rep=3
        // → [2, 12, 7, 8]
        let d = cpu();
        let x = Tensor::<TB, 4>::from_data(
            TensorData::new(vec![0.0f32; 2 * 4 * 7 * 8], [2, 4, 7, 8]),
            &d,
        );
        assert_eq!(repeat_kv(x, 3).dims(), [2, 12, 7, 8]);
    }
}
