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
/// `x`: `[batch, n_kv_heads, seq, head_dim]` → `[batch, n_heads, seq, head_dim]`
fn repeat_kv<B: Backend>(x: Tensor<B, 4>, n_rep: usize) -> Tensor<B, 4> {
    if n_rep == 1 {
        return x;
    }
    let copies: Vec<Tensor<B, 4>> = (0..n_rep).map(|_| x.clone()).collect();
    Tensor::cat(copies, 1)
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
