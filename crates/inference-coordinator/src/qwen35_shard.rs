//! Qwen3.5 hybrid attention shard using Burn.
//!
//! Qwen3.5 alternates between **Gated DeltaNet** (linear attention) and
//! **standard attention** layers in a 3:1 pattern (24 linear + 8 full for 9B).
//!
//! Key differences from Llama/Qwen2.5:
//! - Weight prefix: `model.language_model.layers.X` (not `model.layers.X`)
//! - RMSNorm: applies `(1.0 + weight)` instead of `weight`
//! - Full attention: q_proj outputs 2× (includes output gate), QK-norm, partial RoPE
//! - Linear attention: Gated DeltaNet with conv1d, delta rule recurrence
//! - Tied embeddings: no separate `lm_head.weight`
//!
//! # Weight-file naming convention
//!
//! ```text
//! — First shard (layer_start == 0):
//!     model.language_model.embed_tokens.weight      [vocab_size, hidden_size]
//!
//! — Full attention layers (every 4th, using `self_attn`):
//!     model.language_model.layers.{L}.input_layernorm.weight
//!     model.language_model.layers.{L}.self_attn.q_proj.weight  [num_heads*head_dim*2, hidden]
//!     model.language_model.layers.{L}.self_attn.q_norm.weight  [head_dim]
//!     model.language_model.layers.{L}.self_attn.k_proj.weight  [kv_heads*head_dim, hidden]
//!     model.language_model.layers.{L}.self_attn.k_norm.weight  [head_dim]
//!     model.language_model.layers.{L}.self_attn.v_proj.weight  [kv_heads*head_dim, hidden]
//!     model.language_model.layers.{L}.self_attn.o_proj.weight  [hidden, num_heads*head_dim]
//!     model.language_model.layers.{L}.post_attention_layernorm.weight
//!     model.language_model.layers.{L}.mlp.{gate,up,down}_proj.weight
//!
//! — Linear attention layers (others, using `linear_attn`):
//!     model.language_model.layers.{L}.input_layernorm.weight
//!     model.language_model.layers.{L}.linear_attn.in_proj_qkv.weight  [qkv_dim, hidden]
//!     model.language_model.layers.{L}.linear_attn.in_proj_z.weight    [v_total, hidden]
//!     model.language_model.layers.{L}.linear_attn.in_proj_a.weight    [n_key_heads, hidden]
//!     model.language_model.layers.{L}.linear_attn.in_proj_b.weight    [n_key_heads, hidden]
//!     model.language_model.layers.{L}.linear_attn.conv1d.weight       [qkv_dim, 1, kernel]
//!     model.language_model.layers.{L}.linear_attn.norm.weight         [v_total]
//!     model.language_model.layers.{L}.linear_attn.out_proj.weight     [hidden, v_total]
//!     model.language_model.layers.{L}.linear_attn.A_log               [n_key_heads]
//!     model.language_model.layers.{L}.linear_attn.dt_bias             [n_key_heads]
//!     model.language_model.layers.{L}.post_attention_layernorm.weight
//!     model.language_model.layers.{L}.mlp.{gate,up,down}_proj.weight
//!
//! — Last shard (layer_end == total_layers):
//!     model.language_model.norm.weight
//! ```

use anyhow::{anyhow, Result};
use burn::module::Param;
use burn::nn::{
    Embedding, EmbeddingConfig, EmbeddingRecord,
    Linear, LinearConfig, LinearLayout, LinearRecord,
};
use burn::prelude::Module;
use burn::tensor::{activation, backend::Backend, Int, Tensor, TensorData};
use serde::{Deserialize, Serialize};

use crate::kv_cache::LayerKv;
use crate::qwen35_cache::{LayerState, Qwen35Session, SsmState};

// ── Weight loading helpers (duplicated from llm_shard — they're private) ────

fn f16_to_f32(bits: u16) -> f32 {
    let sign     = (bits >> 15) as u32;
    let exp_h    = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x3ff) as u32;
    let f32_bits = if exp_h == 0 {
        if mantissa == 0 {
            sign << 31
        } else {
            let mut m = mantissa;
            let mut e = 127u32 - 14;
            while (m & (1 << 10)) == 0 { m <<= 1; e -= 1; }
            m &= !(1 << 10);
            (sign << 31) | (e << 23) | (m << 13)
        }
    } else if exp_h == 0x1f {
        (sign << 31) | (0xffu32 << 23) | (mantissa << 13)
    } else {
        (sign << 31) | ((exp_h + 112) << 23) | (mantissa << 13)
    };
    f32::from_bits(f32_bits)
}

fn load_f32(view: &safetensors::tensor::TensorView<'_>) -> Vec<f32> {
    use safetensors::Dtype;
    let data = view.data();
    let shape_elems: usize = view.shape().iter().product::<usize>().max(1);
    let bpe = if data.len() % shape_elems == 0 {
        data.len() / shape_elems
    } else {
        match view.dtype() {
            Dtype::F32 => 4,
            _ => 2,
        }
    };
    match bpe {
        4 => data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        2 => match view.dtype() {
            Dtype::F16 => data
                .chunks_exact(2)
                .map(|b| f16_to_f32(u16::from_le_bytes(b.try_into().unwrap())))
                .collect(),
            _ => data
                .chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes(b.try_into().unwrap());
                    f32::from_bits((bits as u32) << 16)
                })
                .collect(),
        },
        bpe => panic!(
            "tensor dtype={:?} shape={:?} has unsupported bytes/element ({bpe}); data_len={}",
            view.dtype(), view.shape(), data.len()
        ),
    }
}

// ── Config ──────────────────────────────────────────────────────────────────

/// Architecture parameters for one shard of a Qwen3.5 model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen35ShardConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    /// Full-attention heads.
    pub num_attention_heads: usize,
    /// Full-attention KV heads (GQA).
    pub num_key_value_heads: usize,
    /// Head dim for full-attention layers.
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_seq_len: usize,
    /// Fraction of head_dim that gets rotary embeddings (e.g. 0.25 → 64 of 256).
    pub partial_rotary_factor: f64,
    /// How often a full-attention layer appears (e.g. 4 means every 4th layer).
    /// Layer L is full attention if `(L + 1) % full_attn_interval == 0`.
    pub full_attn_interval: usize,
    /// Conv1d kernel size for DeltaNet layers.
    pub conv_kernel: usize,

    // ── Linear attention (DeltaNet) specific dims ───────────────────────
    /// Number of key/query heads in DeltaNet layers.
    pub linear_num_key_heads: usize,
    /// Key/query head dimension in DeltaNet layers.
    pub linear_key_head_dim: usize,
    /// Number of value heads in DeltaNet layers.
    pub linear_num_value_heads: usize,
    /// Value head dimension in DeltaNet layers.
    pub linear_value_head_dim: usize,

    /// Total layers in the full model.
    pub total_layers: usize,
    pub layer_start: usize,
    pub layer_end: usize,
}

impl Qwen35ShardConfig {
    pub fn n_local_layers(&self) -> usize {
        self.layer_end - self.layer_start
    }
    pub fn is_first(&self) -> bool {
        self.layer_start == 0
    }
    pub fn is_last(&self) -> bool {
        self.layer_end >= self.total_layers
    }
    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f64 * self.partial_rotary_factor) as usize
    }
    pub fn is_full_attention(&self, l: usize) -> bool {
        (l + 1) % self.full_attn_interval == 0
    }

    /// Total Q+K+V dimension for the combined in_proj_qkv in DeltaNet.
    pub fn linear_qkv_dim(&self) -> usize {
        let q = self.linear_num_key_heads * self.linear_key_head_dim;
        let k = self.linear_num_key_heads * self.linear_key_head_dim;
        let v = self.linear_num_value_heads * self.linear_value_head_dim;
        q + k + v
    }

    /// Total value output dim for DeltaNet.
    pub fn linear_v_total(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    /// Produce a list of bools for local layers: true = full attention.
    pub fn local_layer_types(&self) -> Vec<bool> {
        (self.layer_start..self.layer_end)
            .map(|l| self.is_full_attention(l))
            .collect()
    }
}

// ── Qwen3.5 RMSNorm ────────────────────────────────────────────────────────

/// RMSNorm variant: `norm(x) * (1.0 + weight)`.
struct Qwen35RmsNorm<B: Backend> {
    weight: Tensor<B, 1>,
    eps: f64,
}

impl<B: Backend> Qwen35RmsNorm<B> {
    fn load(data: Vec<f32>, size: usize, eps: f64, device: &B::Device) -> Self {
        let weight = Tensor::<B, 1>::from_data(TensorData::new(data, [size]), device);
        Self { weight, eps }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_b, _s, h] = x.dims();
        let x_sq = x.clone().powf_scalar(2.0);
        let mean_sq = x_sq.sum_dim(2).div_scalar(h as f32);
        let rms = mean_sq.add_scalar(self.eps as f32).sqrt();
        let normed = x.div(rms);
        let scale = self.weight.clone().add_scalar(1.0f32).reshape([1, 1, h]);
        normed.mul(scale)
    }
}

/// Per-head RMSNorm for QK normalization in full-attention layers.
struct HeadRmsNorm<B: Backend> {
    weight: Tensor<B, 1>,
    eps: f64,
}

impl<B: Backend> HeadRmsNorm<B> {
    fn load(data: Vec<f32>, dim: usize, eps: f64, device: &B::Device) -> Self {
        let weight = Tensor::<B, 1>::from_data(TensorData::new(data, [dim]), device);
        Self { weight, eps }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_b, _h, _s, d] = x.dims();
        let x_sq = x.clone().powf_scalar(2.0);
        let mean_sq = x_sq.sum_dim(3).div_scalar(d as f32);
        let rms = mean_sq.add_scalar(self.eps as f32).sqrt();
        let normed = x.div(rms);
        let scale = self.weight.clone().add_scalar(1.0f32).reshape([1, 1, 1, d]);
        normed.mul(scale)
    }
}

/// Gated RMSNorm: `rms_norm(x) * silu(gate)`.
struct GatedRmsNorm<B: Backend> {
    weight: Tensor<B, 1>,
    eps: f64,
    dim: usize,
}

impl<B: Backend> GatedRmsNorm<B> {
    fn load(data: Vec<f32>, dim: usize, eps: f64, device: &B::Device) -> Self {
        let weight = Tensor::<B, 1>::from_data(TensorData::new(data, [dim]), device);
        Self { weight, eps, dim }
    }

    fn forward(&self, x: Tensor<B, 3>, gate: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_b, _s, d] = x.dims();
        let x_sq = x.clone().powf_scalar(2.0);
        let mean_sq = x_sq.sum_dim(2).div_scalar(d as f32);
        let rms = mean_sq.add_scalar(self.eps as f32).sqrt();
        let normed = x.div(rms);
        let scale = self.weight.clone().add_scalar(1.0f32).reshape([1, 1, self.dim]);
        normed.mul(scale).mul(activation::silu(gate))
    }
}

// ── RoPE helpers ────────────────────────────────────────────────────────────

fn rotate_half<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, h, s, d] = x.dims();
    let half = d / 2;
    let x1 = x.clone().slice([0..b, 0..h, 0..s, 0..half]);
    let x2 = x.slice([0..b, 0..h, 0..s, half..d]);
    Tensor::cat(vec![x2.neg(), x1], 3)
}

/// Partial RoPE: only the first `rotary_dim` dimensions are rotated.
fn apply_partial_rope<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    seq_offset: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta: f64,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [b, n_heads, seq_len, _hd] = q.dims();
    let half_rot = rotary_dim / 2;

    let freqs: Vec<f32> = (0..half_rot)
        .map(|i| 1.0f32 / (theta as f32).powf(2.0 * i as f32 / rotary_dim as f32))
        .collect();
    let freqs = Tensor::<B, 4>::from_data(
        TensorData::new(freqs, [1usize, 1, 1, half_rot]),
        device,
    );

    let pos: Vec<f32> = (seq_offset..seq_offset + seq_len)
        .map(|p| p as f32)
        .collect();
    let pos = Tensor::<B, 4>::from_data(
        TensorData::new(pos, [1usize, 1, seq_len, 1]),
        device,
    );

    let angles = pos.mul(freqs);
    let cos_half = angles.clone().cos();
    let sin_half = angles.sin();
    let cos = Tensor::cat(vec![cos_half.clone(), cos_half], 3);
    let sin = Tensor::cat(vec![sin_half.clone(), sin_half], 3);

    let q_rot_part = q.clone().slice([0..b, 0..n_heads, 0..seq_len, 0..rotary_dim]);
    let q_pass = q.slice([0..b, 0..n_heads, 0..seq_len, rotary_dim..head_dim]);
    let q_rot = q_rot_part.clone().mul(cos.clone())
        .add(rotate_half(q_rot_part).mul(sin.clone()));

    let [_, n_kv, _, _] = k.dims();
    let k_rot_part = k.clone().slice([0..b, 0..n_kv, 0..seq_len, 0..rotary_dim]);
    let k_pass = k.slice([0..b, 0..n_kv, 0..seq_len, rotary_dim..head_dim]);
    let k_rot = k_rot_part.clone().mul(cos)
        .add(rotate_half(k_rot_part).mul(sin));

    (Tensor::cat(vec![q_rot, q_pass], 3), Tensor::cat(vec![k_rot, k_pass], 3))
}

// ── L2 norm ─────────────────────────────────────────────────────────────────

fn l2_norm_4d<B: Backend>(x: Tensor<B, 4>, eps: f32) -> Tensor<B, 4> {
    let x_sq = x.clone().powf_scalar(2.0);
    let sum_sq = x_sq.sum_dim(3);
    let norm = sum_sq.add_scalar(eps).sqrt();
    x.div(norm)
}

// ── GQA repeat ──────────────────────────────────────────────────────────────

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

// ── Full Attention Layer ────────────────────────────────────────────────────

struct Qwen35FullAttention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    q_norm: HeadRmsNorm<B>,
    k_norm: HeadRmsNorm<B>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f64,
    /// Whether q_proj includes an output gate (2× output dim).
    has_output_gate: bool,
}

impl<B: Backend> Qwen35FullAttention<B> {
    fn forward(
        &self,
        x: Tensor<B, 3>,
        seq_offset: usize,
        kv: &mut LayerKv<B>,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let [b, seq, _h] = x.dims();

        let q_raw = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        // Handle output gate if q_proj is double-sized
        let (q, gate) = if self.has_output_gate {
            let q_gate_dim = self.num_heads * self.head_dim;
            let query_flat = q_raw.clone().slice([0..b, 0..seq, 0..q_gate_dim]);
            let gate_flat = q_raw.slice([0..b, 0..seq, q_gate_dim..q_gate_dim * 2]);
            let gate: Tensor<B, 4> = gate_flat
                .reshape([b, seq, self.num_heads, self.head_dim])
                .swap_dims(1, 2);
            (query_flat, Some(gate))
        } else {
            (q_raw, None)
        };

        let q: Tensor<B, 4> = q
            .reshape([b, seq, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k: Tensor<B, 4> = k
            .reshape([b, seq, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v: Tensor<B, 4> = v
            .reshape([b, seq, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        // QK-norm
        let q = self.q_norm.forward(q);
        let k = self.k_norm.forward(k);

        // Partial RoPE
        let (q, k) = apply_partial_rope(
            q, k, seq_offset,
            self.head_dim, self.rotary_dim, self.rope_theta, device,
        );

        // KV cache
        let (k_full, v_full) = kv.append(k, v);

        // GQA expansion
        let n_rep = self.num_heads / self.num_kv_heads;
        let k_exp = repeat_kv(k_full, n_rep);
        let v_exp = repeat_kv(v_full, n_rep);

        // Scaled dot-product attention
        let scale = (self.head_dim as f32).sqrt();
        let attn_w = q.matmul(k_exp.swap_dims(2, 3)).div_scalar(scale);

        // Causal mask
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
        let mut out = attn_p.matmul(v_exp);

        // Apply output gate if present
        if let Some(gate) = gate {
            out = out.mul(activation::sigmoid(gate));
        }

        let out: Tensor<B, 3> = out
            .swap_dims(1, 2)
            .reshape([b, seq, self.num_heads * self.head_dim]);
        self.o_proj.forward(out)
    }
}

// ── Gated DeltaNet (Linear Attention) ───────────────────────────────────────

struct Qwen35GatedDeltaNet<B: Backend> {
    /// Combined Q+K+V projection.
    in_proj_qkv: Linear<B>,
    /// Gate projection for output gating.
    in_proj_z: Linear<B>,
    /// Decay projection (one per key head).
    in_proj_a: Linear<B>,
    /// Beta (learning rate) projection (one per key head).
    in_proj_b: Linear<B>,
    /// Output projection.
    out_proj: Linear<B>,
    /// Decay parameter (log space).
    a_log: Tensor<B, 1>,
    /// Time-step bias.
    dt_bias: Tensor<B, 1>,
    /// Depthwise conv1d weights (no bias).
    conv_weight: Tensor<B, 3>,  // [qkv_dim, 1, kernel]
    /// Gated RMSNorm on output.
    g_norm: GatedRmsNorm<B>,

    num_key_heads: usize,
    key_dim: usize,
    num_value_heads: usize,
    value_dim: usize,
    conv_kernel: usize,
    q_total: usize,  // num_key_heads * key_dim
    k_total: usize,  // num_key_heads * key_dim
    v_total: usize,  // num_value_heads * value_dim
}

impl<B: Backend> Qwen35GatedDeltaNet<B> {
    fn forward(
        &self,
        x: Tensor<B, 3>,
        state: &mut SsmState<B>,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let [b, seq, _h] = x.dims();

        // Combined QKV projection
        let qkv = self.in_proj_qkv.forward(x.clone()); // [b, seq, q+k+v]
        let qkv_dim = self.q_total + self.k_total + self.v_total;

        // Causal depthwise conv1d on the full QKV
        let qkv = self.causal_conv1d(qkv, qkv_dim, state, device);

        // SiLU activation after conv
        let qkv = activation::silu(qkv);

        // Split into Q, K, V
        let q_end = self.q_total;
        let k_end = q_end + self.k_total;
        let q = qkv.clone().slice([0..b, 0..seq, 0..q_end]);
        let k = qkv.clone().slice([0..b, 0..seq, q_end..k_end]);
        let v = qkv.slice([0..b, 0..seq, k_end..qkv_dim]);

        // Gate + decay/beta projections
        let z = self.in_proj_z.forward(x.clone());  // [b, seq, v_total]
        let a = self.in_proj_a.forward(x.clone());  // [b, seq, num_key_heads]
        let beta = self.in_proj_b.forward(x);        // [b, seq, num_key_heads]

        // Reshape to per-head: [b, heads, seq, dim]
        let q: Tensor<B, 4> = q
            .reshape([b, seq, self.num_key_heads, self.key_dim])
            .swap_dims(1, 2);
        let k: Tensor<B, 4> = k
            .reshape([b, seq, self.num_key_heads, self.key_dim])
            .swap_dims(1, 2);
        let v: Tensor<B, 4> = v
            .reshape([b, seq, self.num_value_heads, self.value_dim])
            .swap_dims(1, 2);

        // L2-normalize Q and K
        let q = l2_norm_4d(q, 1e-6);
        let k = l2_norm_4d(k, 1e-6);

        // Compute decay: g = exp(-exp(A_log) * softplus(a + dt_bias))
        let a: Tensor<B, 4> = a
            .reshape([b, seq, self.num_key_heads, 1])
            .swap_dims(1, 2);
        let dt_bias = self.dt_bias.clone().reshape([1, self.num_key_heads, 1, 1]);
        let a_plus_bias = a.add(dt_bias);
        let softplus_val = a_plus_bias.exp().add_scalar(1.0f32).log();
        let a_log = self.a_log.clone().exp().reshape([1, self.num_key_heads, 1, 1]);
        let g = a_log.mul(softplus_val).neg().exp();

        // Beta: sigmoid
        let beta: Tensor<B, 4> = beta
            .reshape([b, seq, self.num_key_heads, 1])
            .swap_dims(1, 2);
        let beta = activation::sigmoid(beta);

        // Expand V heads to match K heads if needed (for the delta rule)
        // Delta rule uses key_heads for state, but value can have more heads.
        // We need to handle the asymmetric head counts.
        let y = self.delta_rule_recurrence(q, k, v, g, beta, state, device);

        // y: [b, num_value_heads, seq, value_dim] → [b, seq, v_total]
        let y: Tensor<B, 3> = y
            .swap_dims(1, 2)
            .reshape([b, seq, self.v_total]);

        // Gated RMSNorm + output projection
        let y = self.g_norm.forward(y, z);
        self.out_proj.forward(y)
    }

    /// Causal depthwise conv1d (no bias).
    fn causal_conv1d(
        &self,
        x: Tensor<B, 3>,
        channels: usize,
        state: &mut SsmState<B>,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let [b, seq, _ch] = x.dims();
        let k = self.conv_kernel;

        if seq == 1 {
            // Decode: shift state, compute single output
            let x_col = x.clone().reshape([b, channels, 1]);
            let conv_st = if let Some(ref cs) = state.conv_state {
                let old = cs.clone().slice([0..b, 0..channels, 1..k - 1]);
                Tensor::cat(vec![old, x_col], 2)
            } else {
                let zeros = Tensor::<B, 3>::zeros([b, channels, k - 2], device);
                Tensor::cat(vec![zeros, x_col], 2)
            };
            state.conv_state = Some(conv_st.clone());

            let conv_input = Tensor::cat(
                vec![conv_st, x.reshape([b, channels, 1])],
                2,
            );
            let weight = self.conv_weight.clone().reshape([1, channels, k]);
            let out = conv_input.mul(weight).sum_dim(2); // [b, channels, 1]
            out.reshape([b, 1, channels])
        } else {
            // Prefill: left-pad and full conv
            let x_t = x.swap_dims(1, 2); // [b, channels, seq]
            let pad = Tensor::<B, 3>::zeros([b, channels, k - 1], device);
            let padded = Tensor::cat(vec![pad, x_t], 2);

            let weight = self.conv_weight.clone().reshape([1, channels, k]);
            let mut out_slices: Vec<Tensor<B, 3>> = Vec::with_capacity(seq);
            for t in 0..seq {
                let window = padded.clone().slice([0..b, 0..channels, t..t + k]);
                let val = window.mul(weight.clone()).sum_dim(2);
                out_slices.push(val);
            }
            let out = Tensor::cat(out_slices, 2); // [b, channels, seq]

            // Save conv state
            let total_len = seq + k - 1;
            let last_state = if seq >= k - 1 {
                padded.slice([0..b, 0..channels, seq..seq + k - 1])
            } else {
                padded.slice([0..b, 0..channels, total_len - (k - 1)..total_len])
            };
            state.conv_state = Some(last_state);

            out.swap_dims(1, 2) // [b, seq, channels]
        }
    }

    /// Gated delta rule recurrence with asymmetric key/value heads.
    ///
    /// When num_value_heads > num_key_heads, each key head manages
    /// `num_value_heads / num_key_heads` value heads' state matrices.
    fn delta_rule_recurrence(
        &self,
        q: Tensor<B, 4>,     // [b, num_key_heads, seq, key_dim]
        k: Tensor<B, 4>,     // [b, num_key_heads, seq, key_dim]
        v: Tensor<B, 4>,     // [b, num_value_heads, seq, value_dim]
        g: Tensor<B, 4>,     // [b, num_key_heads, seq, 1]
        beta: Tensor<B, 4>,  // [b, num_key_heads, seq, 1]
        state: &mut SsmState<B>,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        let [b, _n_key_heads, seq, _kd] = q.dims();
        let n_key_heads = self.num_key_heads;
        let n_val_heads = self.num_value_heads;
        let key_dim = self.key_dim;
        let value_dim = self.value_dim;
        // Each key head covers this many value heads
        let val_heads_per_key = n_val_heads / n_key_heads;
        // State dimension per key head: val_heads_per_key * value_dim
        let state_v_dim = val_heads_per_key * value_dim;

        // h: [b, n_key_heads, key_dim, state_v_dim]
        let mut h = state.h.take().unwrap_or_else(|| {
            Tensor::<B, 4>::zeros([b, n_key_heads, key_dim, state_v_dim], device)
        });

        let mut y_steps: Vec<Tensor<B, 4>> = Vec::with_capacity(seq);

        for t in 0..seq {
            let q_t = q.clone().slice([0..b, 0..n_key_heads, t..t + 1, 0..key_dim]);
            let k_t = k.clone().slice([0..b, 0..n_key_heads, t..t + 1, 0..key_dim]);
            let g_t = g.clone().slice([0..b, 0..n_key_heads, t..t + 1, 0..1]);
            let beta_t = beta.clone().slice([0..b, 0..n_key_heads, t..t + 1, 0..1]);

            // Gather value heads grouped by key head: [b, n_key_heads, 1, state_v_dim]
            let mut v_groups: Vec<Tensor<B, 4>> = Vec::with_capacity(n_key_heads);
            for kh in 0..n_key_heads {
                let vh_start = kh * val_heads_per_key;
                let vh_end = vh_start + val_heads_per_key;
                // [b, val_heads_per_key, 1, value_dim]
                let v_slice = v.clone().slice([0..b, vh_start..vh_end, t..t + 1, 0..value_dim]);
                // Reshape to [b, 1, 1, state_v_dim]
                let v_flat = v_slice.reshape([b, 1, 1, state_v_dim]);
                v_groups.push(v_flat);
            }
            let v_t = Tensor::cat(v_groups, 1); // [b, n_key_heads, 1, state_v_dim]

            // k_t^T: [b, n_key_heads, key_dim, 1]
            #[allow(non_snake_case)]
            let k_t_T = k_t.clone().swap_dims(2, 3);

            // retrieved = k_t @ h → [b, n_key_heads, 1, state_v_dim]
            let retrieved = k_t.matmul(h.clone());
            let error = v_t.sub(retrieved);

            // outer product: k_t^T @ error → [b, n_key_heads, key_dim, state_v_dim]
            let update = k_t_T.matmul(error);

            h = h.mul(g_t).add(update.mul(beta_t));

            // y_t = q_t @ h → [b, n_key_heads, 1, state_v_dim]
            let y_t = q_t.matmul(h.clone());
            y_steps.push(y_t);
        }

        state.h = Some(h);

        // Concatenate time steps: [b, n_key_heads, seq, state_v_dim]
        let y = Tensor::cat(y_steps, 2);

        // Reshape back to [b, n_value_heads, seq, value_dim]
        // y is [b, n_key_heads, seq, val_heads_per_key * value_dim]
        // We need to un-group the value heads
        let mut value_head_list: Vec<Tensor<B, 4>> = Vec::with_capacity(n_val_heads);
        for kh in 0..n_key_heads {
            let group = y.clone().slice([0..b, kh..kh + 1, 0..seq, 0..state_v_dim]);
            // [b, 1, seq, val_heads_per_key * value_dim] → [b, val_heads_per_key, seq, value_dim]
            let ungrouped = group.reshape([b, val_heads_per_key, seq, value_dim]);
            value_head_list.push(ungrouped);
        }
        Tensor::cat(value_head_list, 1) // [b, n_value_heads, seq, value_dim]
    }
}

// ── MLP ─────────────────────────────────────────────────────────────────────

struct Qwen35Mlp<B: Backend> {
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
}

impl<B: Backend> Qwen35Mlp<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = activation::silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate.mul(up))
    }
}

// ── Decoder Layer ───────────────────────────────────────────────────────────

enum Qwen35Attn<B: Backend> {
    Full(Qwen35FullAttention<B>),
    Linear(Qwen35GatedDeltaNet<B>),
}

struct Qwen35DecoderLayer<B: Backend> {
    input_norm: Qwen35RmsNorm<B>,
    attn: Qwen35Attn<B>,
    post_attn_norm: Qwen35RmsNorm<B>,
    mlp: Qwen35Mlp<B>,
}

impl<B: Backend> Qwen35DecoderLayer<B> {
    fn forward(
        &self,
        x: Tensor<B, 3>,
        seq_offset: usize,
        layer_state: &mut LayerState<B>,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let h = self.input_norm.forward(x.clone());
        let h = match (&self.attn, layer_state) {
            (Qwen35Attn::Full(attn), LayerState::FullAttention(kv)) => {
                attn.forward(h, seq_offset, kv, device)
            }
            (Qwen35Attn::Linear(delta), LayerState::LinearAttention(ssm)) => {
                delta.forward(h, ssm, device)
            }
            _ => panic!("layer type mismatch between model and session state"),
        };
        let x = x.add(h);
        let h = self.post_attn_norm.forward(x.clone());
        let h = self.mlp.forward(h);
        x.add(h)
    }
}

// ── Shard ───────────────────────────────────────────────────────────────────

pub struct Qwen35Shard<B: Backend> {
    cfg: Qwen35ShardConfig,
    embed: Option<Embedding<B>>,
    layers: Vec<Qwen35DecoderLayer<B>>,
    final_norm: Option<Qwen35RmsNorm<B>>,
    lm_head: Option<Linear<B>>,
    device: B::Device,
}

impl<B: Backend> Qwen35Shard<B> {
    fn load_from_byte_slices(
        buffers: &[&[u8]],
        cfg: Qwen35ShardConfig,
        device: B::Device,
    ) -> Result<Self> {
        // Deserialize all safetensors files ONCE upfront (cached).
        let st_objects: Vec<safetensors::SafeTensors<'_>> = buffers
            .iter()
            .map(|b| safetensors::SafeTensors::deserialize(b))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| anyhow!("safetensors deserialize: {e:?}"))?;

        let load_from_files = |tensor_name: &str| -> Result<Vec<f32>> {
            for st in &st_objects {
                if let Ok(view) = st.tensor(tensor_name) {
                    let elems = view.shape().iter().product::<usize>();
                    eprintln!(
                        "  [load] {tensor_name}: shape={:?} dtype={:?} data_bytes={}  → {} elems",
                        view.shape(), view.dtype(), view.data().len(), elems
                    );
                    return Ok(load_f32(&view));
                }
            }
            Err(anyhow!("tensor not found in any weight file: {tensor_name}"))
        };

        let load_shape = |tensor_name: &str| -> Result<Vec<usize>> {
            for st in &st_objects {
                if let Ok(view) = st.tensor(tensor_name) {
                    return Ok(view.shape().to_vec());
                }
            }
            Err(anyhow!("tensor not found in any weight file: {tensor_name}"))
        };

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
                .with_layout(LinearLayout::Col)
                .init::<B>(&device)
                .load_record(record))
        };

        let mk_rms = |name: &str, size: usize| -> Result<Qwen35RmsNorm<B>> {
            let data = load_from_files(name)?;
            Ok(Qwen35RmsNorm::load(data, size, cfg.rms_norm_eps, &device))
        };

        let mk_head_rms = |name: &str, dim: usize| -> Result<HeadRmsNorm<B>> {
            let data = load_from_files(name)?;
            Ok(HeadRmsNorm::load(data, dim, cfg.rms_norm_eps, &device))
        };

        let mk_gated_rms = |name: &str, dim: usize| -> Result<GatedRmsNorm<B>> {
            let data = load_from_files(name)?;
            Ok(GatedRmsNorm::load(data, dim, cfg.rms_norm_eps, &device))
        };

        let load_tensor_1d = |name: &str, dim: usize| -> Result<Tensor<B, 1>> {
            let data = load_from_files(name)?;
            Ok(Tensor::<B, 1>::from_data(TensorData::new(data, [dim]), &device))
        };

        let load_tensor_3d = |name: &str, d0: usize, d1: usize, d2: usize| -> Result<Tensor<B, 3>> {
            let data = load_from_files(name)?;
            Ok(Tensor::<B, 3>::from_data(TensorData::new(data, [d0, d1, d2]), &device))
        };

        // Embedding (first shard only)
        let embed = if cfg.is_first() {
            let name = "model.language_model.embed_tokens.weight";
            let data = load_from_files(name)?;
            let shape = load_shape(name)?;
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
            let p = format!("model.language_model.layers.{l}");

            let input_norm = mk_rms(&format!("{p}.input_layernorm.weight"), cfg.hidden_size)?;
            let post_attn_norm = mk_rms(&format!("{p}.post_attention_layernorm.weight"), cfg.hidden_size)?;

            let mlp = Qwen35Mlp {
                gate_proj: mk_linear(&format!("{p}.mlp.gate_proj.weight"), cfg.hidden_size, cfg.intermediate_size)?,
                up_proj: mk_linear(&format!("{p}.mlp.up_proj.weight"), cfg.hidden_size, cfg.intermediate_size)?,
                down_proj: mk_linear(&format!("{p}.mlp.down_proj.weight"), cfg.intermediate_size, cfg.hidden_size)?,
            };

            let attn = if cfg.is_full_attention(l) {
                // Full attention with self_attn prefix
                let ap = format!("{p}.self_attn");

                // Detect q_proj output size to determine if there's an output gate
                let q_shape = load_shape(&format!("{ap}.q_proj.weight"))?;
                let q_out_dim = q_shape[0];
                let expected_q_dim = cfg.num_attention_heads * cfg.head_dim;
                let has_output_gate = q_out_dim == expected_q_dim * 2;

                let q_proj_out = if has_output_gate { expected_q_dim * 2 } else { expected_q_dim };

                Qwen35Attn::Full(Qwen35FullAttention {
                    q_proj: mk_linear(&format!("{ap}.q_proj.weight"), cfg.hidden_size, q_proj_out)?,
                    k_proj: mk_linear(&format!("{ap}.k_proj.weight"), cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim)?,
                    v_proj: mk_linear(&format!("{ap}.v_proj.weight"), cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim)?,
                    o_proj: mk_linear(&format!("{ap}.o_proj.weight"), cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size)?,
                    q_norm: mk_head_rms(&format!("{ap}.q_norm.weight"), cfg.head_dim)?,
                    k_norm: mk_head_rms(&format!("{ap}.k_norm.weight"), cfg.head_dim)?,
                    num_heads: cfg.num_attention_heads,
                    num_kv_heads: cfg.num_key_value_heads,
                    head_dim: cfg.head_dim,
                    rotary_dim: cfg.rotary_dim(),
                    rope_theta: cfg.rope_theta,
                    has_output_gate,
                })
            } else {
                // Linear attention with linear_attn prefix
                let ap = format!("{p}.linear_attn");
                let q_total = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
                let k_total = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
                let v_total = cfg.linear_v_total();
                let qkv_dim = cfg.linear_qkv_dim();

                Qwen35Attn::Linear(Qwen35GatedDeltaNet {
                    in_proj_qkv: mk_linear(&format!("{ap}.in_proj_qkv.weight"), cfg.hidden_size, qkv_dim)?,
                    in_proj_z: mk_linear(&format!("{ap}.in_proj_z.weight"), cfg.hidden_size, v_total)?,
                    in_proj_a: mk_linear(&format!("{ap}.in_proj_a.weight"), cfg.hidden_size, cfg.linear_num_key_heads)?,
                    in_proj_b: mk_linear(&format!("{ap}.in_proj_b.weight"), cfg.hidden_size, cfg.linear_num_key_heads)?,
                    out_proj: mk_linear(&format!("{ap}.out_proj.weight"), v_total, cfg.hidden_size)?,
                    a_log: load_tensor_1d(&format!("{ap}.A_log"), cfg.linear_num_key_heads)?,
                    dt_bias: load_tensor_1d(&format!("{ap}.dt_bias"), cfg.linear_num_key_heads)?,
                    conv_weight: load_tensor_3d(&format!("{ap}.conv1d.weight"), qkv_dim, 1, cfg.conv_kernel)?,
                    g_norm: mk_gated_rms(&format!("{ap}.norm.weight"), v_total)?,
                    num_key_heads: cfg.linear_num_key_heads,
                    key_dim: cfg.linear_key_head_dim,
                    num_value_heads: cfg.linear_num_value_heads,
                    value_dim: cfg.linear_value_head_dim,
                    conv_kernel: cfg.conv_kernel,
                    q_total,
                    k_total,
                    v_total,
                })
            };

            layers.push(Qwen35DecoderLayer { input_norm, attn, post_attn_norm, mlp });
        }

        // Final norm (last shard only)
        let final_norm = if cfg.is_last() {
            Some(mk_rms("model.language_model.norm.weight", cfg.hidden_size)?)
        } else {
            None
        };

        // lm_head — Qwen3.5 may use tied embeddings
        let lm_head = if cfg.is_last() {
            let head_data = load_from_files("lm_head.weight")
                .or_else(|_| load_from_files("model.language_model.lm_head.weight"))
                .or_else(|_| {
                    eprintln!("  [lm_head] not found — using weight-tied embed_tokens");
                    load_from_files("model.language_model.embed_tokens.weight")
                })?;
            let weight = Tensor::<B, 2>::from_data(
                TensorData::new(head_data, [cfg.vocab_size, cfg.hidden_size]),
                &device,
            );
            let record = LinearRecord {
                weight: Param::from_tensor(weight),
                bias: None,
            };
            Some(
                LinearConfig::new(cfg.hidden_size, cfg.vocab_size)
                    .with_bias(false)
                    .with_layout(LinearLayout::Col)
                    .init::<B>(&device)
                    .load_record(record),
            )
        } else {
            None
        };

        Ok(Self { cfg, embed, layers, final_norm, lm_head, device })
    }

    pub fn load_from_bytes_multi(
        bytes_vec: Vec<Vec<u8>>,
        cfg: Qwen35ShardConfig,
        device: B::Device,
    ) -> Result<Self> {
        let slices: Vec<&[u8]> = bytes_vec.iter().map(|v| v.as_slice()).collect();
        Self::load_from_byte_slices(&slices, cfg, device)
    }

    pub fn load_from_bytes(bytes: &[u8], cfg: Qwen35ShardConfig, device: B::Device) -> Result<Self> {
        Self::load_from_byte_slices(&[bytes], cfg, device)
    }

    pub fn config(&self) -> &Qwen35ShardConfig { &self.cfg }
    pub fn device(&self) -> &B::Device { &self.device }

    /// Forward pass through all layers owned by this shard.
    pub fn forward(
        &self,
        input_frame: &crate::tensor_frame::TensorFrame,
        session: &mut Qwen35Session<B>,
    ) -> Result<crate::tensor_frame::TensorFrame> {
        let seq_offset = input_frame.seq_offset;

        let mut hidden: Tensor<B, 3> = if let Some(embed) = &self.embed {
            let token_ids_f32 = input_frame.to_f32_vec()?;
            let seq_len = input_frame.shape.last().copied().unwrap_or(1);
            let ids: Vec<i32> = token_ids_f32.iter().map(|&f| f as i32).collect();
            let id_tensor =
                Tensor::<B, 2, Int>::from_data(TensorData::new(ids, [1usize, seq_len]), &self.device);
            embed.forward(id_tensor)
        } else {
            input_frame.to_burn_3d(&self.device)?
        };

        for (local_idx, layer) in self.layers.iter().enumerate() {
            let layer_state = &mut session.layers[local_idx];
            hidden = layer.forward(hidden, seq_offset, layer_state, &self.device);
        }

        let new_tokens = input_frame.shape.last().copied().unwrap_or(1);
        session.seq_pos += new_tokens;

        if let (Some(norm), Some(head)) = (&self.final_norm, &self.lm_head) {
            let normed = norm.forward(hidden);
            let [_b, seq, hidden_sz] = normed.dims();
            let last_hidden: Tensor<B, 2> =
                normed.slice([0..1usize, seq - 1..seq, 0..hidden_sz]).reshape([1, hidden_sz]);
            let logits = head.forward(last_hidden);

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
            crate::tensor_frame::TensorFrame::from_burn(hidden, session.seq_pos)
        }
    }
}

// ── Convenience constructors for WGPU ───────────────────────────────────────

impl Qwen35Shard<crate::InferenceBackend> {
    pub fn load_from_bytes_wgpu(bytes: &[u8], cfg: Qwen35ShardConfig) -> Result<Self> {
        use burn::backend::wgpu::WgpuDevice;
        Self::load_from_bytes(bytes, cfg, WgpuDevice::BestAvailable)
    }

    pub fn load_from_bytes_multi_wgpu(
        bytes_vec: Vec<Vec<u8>>,
        cfg: Qwen35ShardConfig,
    ) -> Result<Self> {
        use burn::backend::wgpu::WgpuDevice;
        Self::load_from_bytes_multi(bytes_vec, cfg, WgpuDevice::BestAvailable)
    }
}

// ── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::NdArray;

    type TB = NdArray;

    #[test]
    fn test_qwen35_rms_norm() {
        let device = NdArrayDevice::Cpu;
        let norm = Qwen35RmsNorm::<TB>::load(vec![0.0; 4], 4, 1e-5, &device);
        let x = Tensor::<TB, 3>::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [1, 1, 4]),
            &device,
        );
        let out = norm.forward(x);
        let vals: Vec<f32> = out.into_data().to_vec().unwrap();
        let rms = (30.0f32 / 4.0).sqrt();
        for (i, &v) in vals.iter().enumerate() {
            let expected = (i as f32 + 1.0) / rms;
            assert!((v - expected).abs() < 1e-4, "mismatch at {i}: {v} vs {expected}");
        }
    }

    #[test]
    fn test_qwen35_rms_norm_with_weight() {
        let device = NdArrayDevice::Cpu;
        let norm = Qwen35RmsNorm::<TB>::load(vec![0.5; 4], 4, 1e-5, &device);
        let x = Tensor::<TB, 3>::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [1, 1, 4]),
            &device,
        );
        let out = norm.forward(x);
        let vals: Vec<f32> = out.into_data().to_vec().unwrap();
        let rms = (30.0f32 / 4.0).sqrt();
        for (i, &v) in vals.iter().enumerate() {
            let expected = (i as f32 + 1.0) / rms * 1.5;
            assert!((v - expected).abs() < 1e-4, "mismatch at {i}: {v} vs {expected}");
        }
    }

    #[test]
    fn test_partial_rope_passthrough() {
        let device = NdArrayDevice::Cpu;
        let head_dim = 8;
        let rotary_dim = 4;

        let q_data: Vec<f32> = (0..head_dim).map(|i| i as f32 + 1.0).collect();
        let k_data: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let q = Tensor::<TB, 4>::from_data(
            TensorData::new(q_data.clone(), [1usize, 1, 1, head_dim]),
            &device,
        );
        let k = Tensor::<TB, 4>::from_data(
            TensorData::new(k_data.clone(), [1usize, 1, 1, head_dim]),
            &device,
        );

        let (q_out, k_out) = apply_partial_rope(q, k, 0, head_dim, rotary_dim, 10000.0, &device);
        let q_vals: Vec<f32> = q_out.into_data().to_vec().unwrap();
        let k_vals: Vec<f32> = k_out.into_data().to_vec().unwrap();

        for i in rotary_dim..head_dim {
            assert!((q_vals[i] - q_data[i]).abs() < 1e-4, "Q passthrough dim {i} changed");
            assert!((k_vals[i] - k_data[i]).abs() < 1e-4, "K passthrough dim {i} changed");
        }
    }

    #[test]
    fn test_l2_norm() {
        let device = NdArrayDevice::Cpu;
        let x = Tensor::<TB, 4>::from_data(
            TensorData::new(vec![3.0f32, 4.0], [1usize, 1, 1, 2]),
            &device,
        );
        let normed = l2_norm_4d(x, 1e-6);
        let vals: Vec<f32> = normed.into_data().to_vec().unwrap();
        assert!((vals[0] - 0.6).abs() < 1e-4);
        assert!((vals[1] - 0.8).abs() < 1e-4);
    }

    #[test]
    fn test_config_layer_types_9b() {
        let cfg = Qwen35ShardConfig {
            hidden_size: 4096,
            intermediate_size: 12288,
            num_attention_heads: 16,
            num_key_value_heads: 4,
            head_dim: 256,
            vocab_size: 248320,
            rms_norm_eps: 1e-6,
            rope_theta: 10000000.0,
            max_seq_len: 4096,
            partial_rotary_factor: 0.25,
            full_attn_interval: 4,
            conv_kernel: 4,
            linear_num_key_heads: 16,
            linear_key_head_dim: 128,
            linear_num_value_heads: 32,
            linear_value_head_dim: 128,
            total_layers: 32,
            layer_start: 0,
            layer_end: 32,
        };
        let types = cfg.local_layer_types();
        assert_eq!(types.len(), 32);
        // Full attention at layers 3, 7, 11, 15, 19, 23, 27, 31
        assert_eq!(types.iter().filter(|&&t| t).count(), 8);
        assert_eq!(types.iter().filter(|&&t| !t).count(), 24);
        assert!(types[3]);
        assert!(types[7]);
        assert!(!types[0]);
        assert!(!types[1]);
        assert!(!types[2]);
    }

    #[test]
    fn test_linear_qkv_dim() {
        let cfg = Qwen35ShardConfig {
            hidden_size: 4096,
            intermediate_size: 12288,
            num_attention_heads: 16,
            num_key_value_heads: 4,
            head_dim: 256,
            vocab_size: 248320,
            rms_norm_eps: 1e-6,
            rope_theta: 10000000.0,
            max_seq_len: 4096,
            partial_rotary_factor: 0.25,
            full_attn_interval: 4,
            conv_kernel: 4,
            linear_num_key_heads: 16,
            linear_key_head_dim: 128,
            linear_num_value_heads: 32,
            linear_value_head_dim: 128,
            total_layers: 32,
            layer_start: 0,
            layer_end: 32,
        };
        // Q=16*128=2048, K=16*128=2048, V=32*128=4096 → total=8192
        assert_eq!(cfg.linear_qkv_dim(), 8192);
        assert_eq!(cfg.linear_v_total(), 4096);
    }
}
