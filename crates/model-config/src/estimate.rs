use crate::ModelConfig;

#[derive(Debug, Clone)]
pub struct EstimateInput {
    pub quant_bits: f64,
    pub seq_len: usize,
    pub batch: usize,
    pub bytes_per_kv: f64,
    pub nodes: usize,
    pub overhead: f64,
    pub runtime_overhead_ratio: f64,
}

impl Default for EstimateInput {
    fn default() -> Self {
        Self {
            quant_bits: 4.0,
            seq_len: 8192,
            batch: 1,
            bytes_per_kv: 1.0,
            nodes: 4,
            overhead: 1.15,
            runtime_overhead_ratio: 0.15,
        }
    }
}

pub fn estimate_weights_gb(cfg: &ModelConfig, quant_bits: f64, overhead: f64) -> f64 {
    let weights_total = cfg.params_b * 1e9 * (quant_bits / 8.0) * overhead;
    weights_total / (1024.0 * 1024.0 * 1024.0)
}

pub fn estimate_required_vram_gb(cfg: &ModelConfig, input: &EstimateInput) -> f64 {
    let weights_total = cfg.params_b * 1e9 * (input.quant_bits / 8.0) * input.overhead;

    let kv_per_token = 2.0
        * cfg.n_layers as f64
        * cfg.n_kv_heads as f64
        * cfg.head_dim as f64
        * input.bytes_per_kv;

    let kv_total = kv_per_token * input.seq_len as f64 * input.batch as f64;

    let weights_per_node = weights_total / input.nodes as f64;
    let kv_per_node = kv_total / input.nodes as f64;

    let runtime_overhead = input.runtime_overhead_ratio * (weights_per_node + kv_per_node);

    (weights_per_node + kv_per_node + runtime_overhead) / (1024.0 * 1024.0 * 1024.0)
}
