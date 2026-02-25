use anyhow::{anyhow, Result};
use async_trait::async_trait;

use crate::plan::ShardSpec;
use crate::{InferenceRequest, ShardExecutor};

#[derive(Debug, Clone)]
pub struct CandleShardRunnerConfig {
    pub model_path: Option<String>,
    pub layer_start: usize,
    pub layer_end: usize,
    pub device: String,
    pub mode: String,
    pub hidden: usize,
    pub tensor_name: Option<String>,
    pub sample_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct CandleShardRunner {
    config: CandleShardRunnerConfig,
}

impl CandleShardRunner {
    pub fn new(config: CandleShardRunnerConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &CandleShardRunnerConfig {
        &self.config
    }

    #[cfg(feature = "candle")]
    pub fn load_model(&self) -> Result<()> {
        let _ = &self.config;
        Ok(())
    }
}

#[async_trait]
impl ShardExecutor for CandleShardRunner {
    async fn run_shard(
        &self,
        shard: &ShardSpec,
        _input: &str,
        _request: &InferenceRequest,
    ) -> Result<String> {
        #[cfg(feature = "candle")]
        {
            return run_simulated(&self.config, shard, _input, _request);
        }

        #[cfg(not(feature = "candle"))]
        Err(anyhow!(
            "candle feature not enabled (peer={}, layers={}..{})",
            shard.peer_id,
            shard.layer_start,
            shard.layer_end
        ))
    }
}

#[cfg(feature = "candle")]
fn run_simulated(
    config: &CandleShardRunnerConfig,
    shard: &ShardSpec,
    input: &str,
    request: &InferenceRequest,
) -> Result<String> {
    use candle_core::safetensors::SliceSafetensors;
    use candle_core::{Device, Tensor};

    let device = match config.device.as_str() {
        "metal" => Device::new_metal(0).unwrap_or(Device::Cpu),
        "cpu" | _ => Device::Cpu,
    };

    match config.mode.as_str() {
        "simulate" => {
            let mut x = Tensor::from_slice(&[input.len() as f32], (1,), &device)?;
            for layer in shard.layer_start..shard.layer_end {
                let add = Tensor::from_slice(&[layer as f32], (1,), &device)?;
                x = x.broadcast_add(&add)?;
            }

            let out = x.to_vec1::<f32>()?.get(0).copied().unwrap_or(0.0);
            Ok(format!(
                "simulated-shard peer={} layers={}..{} input_len={} max_tokens={} temp={:.2} value={:.2}",
                shard.peer_id,
                shard.layer_start,
                shard.layer_end,
                input.len(),
                request.max_tokens,
                request.temperature,
                out
            ))
        }
        "mlp" => {
            let hidden = config.hidden.max(1);
            let mut features = vec![0.0f32; hidden];
            for (idx, byte) in input.as_bytes().iter().enumerate() {
                let slot = idx % hidden;
                features[slot] += *byte as f32 / 255.0;
            }

            let mut x = Tensor::from_slice(&features, (1, hidden), &device)?;
            let mut weights = vec![0.0f32; hidden * hidden];
            for i in 0..hidden {
                for j in 0..hidden {
                    let idx = i * hidden + j;
                    weights[idx] = (idx as f32 + 1.0) / (hidden * hidden) as f32;
                }
            }
            let w = Tensor::from_slice(&weights, (hidden, hidden), &device)?;

            for _ in shard.layer_start..shard.layer_end {
                x = x.matmul(&w)?;
                x = x.tanh()?;
            }

            let sum = x.sum_all()?.to_vec0::<f32>()?;
            Ok(format!(
                "candle-mlp peer={} layers={}..{} hidden={} input_len={} max_tokens={} temp={:.2} sum={:.4}",
                shard.peer_id,
                shard.layer_start,
                shard.layer_end,
                hidden,
                input.len(),
                request.max_tokens,
                request.temperature,
                sum
            ))
        }
        "weights" => {
            let path = config
                .model_path
                .as_ref()
                .ok_or_else(|| anyhow!("model_path is required for weights mode"))?;
            let data = std::fs::read(path)?;
            let safetensors = SliceSafetensors::new(&data)?;
            let (name, shape) = if let Some(name) = &config.tensor_name {
                let view = safetensors.get(name)?;
                (name.clone(), view.shape().to_vec())
            } else {
                let mut best: Option<(String, Vec<usize>, usize)> = None;
                for (name, view) in safetensors.tensors() {
                    let shape = view.shape().to_vec();
                    let elems = shape.iter().product::<usize>().max(1);
                    let replace = match &best {
                        None => true,
                        Some((_, _, best_elems)) => elems < *best_elems,
                    };
                    if replace {
                        best = Some((name, shape, elems));
                    }
                }
                let (name, shape, _) =
                    best.ok_or_else(|| anyhow!("no tensors found in {path}"))?;
                (name, shape)
            };
            let tensor = safetensors.load(&name, &device)?;
            let slice = match tensor.dims() {
                [_d0, d1, ..] => {
                    let len1 = 16usize.min(*d1);
                    tensor.narrow(0, 0, 1)?.narrow(1, 0, len1)?
                }
                [d0] => {
                    let len0 = 16usize.min(*d0);
                    tensor.narrow(0, 0, len0)?
                }
                _ => tensor.clone(),
            };
            let slice = slice.to_dtype(candle_core::DType::F32)?;
            let slice_sum = slice.sum_all()?.to_vec0::<f32>()?;
            let sample = slice_sum;
            Ok(format!(
                "weights peer={} layers={}..{} tensor={} dtype={:?} shape={:?} slice_sum={:.4}",
                shard.peer_id,
                shard.layer_start,
                shard.layer_end,
                name,
                tensor.dtype(),
                shape,
                sample
            ))
        }
        "forward" => {
            let path = config
                .model_path
                .as_ref()
                .ok_or_else(|| anyhow!("model_path is required for forward mode"))?;
            let data = std::fs::read(path)?;
            let safetensors = SliceSafetensors::new(&data)?;
            let (name, shape) = if let Some(name) = &config.tensor_name {
                let view = safetensors.get(name)?;
                (name.clone(), view.shape().to_vec())
            } else {
                let mut best: Option<(String, Vec<usize>, usize)> = None;
                for (name, view) in safetensors.tensors() {
                    let shape = view.shape().to_vec();
                    let elems = shape.iter().product::<usize>().max(1);
                    let replace = match &best {
                        None => true,
                        Some((_, _, best_elems)) => elems < *best_elems,
                    };
                    if replace {
                        best = Some((name, shape, elems));
                    }
                }
                let (name, shape, _) =
                    best.ok_or_else(|| anyhow!("no tensors found in {path}"))?;
                (name, shape)
            };
            let tensor = safetensors.load(&name, &device)?.to_dtype(candle_core::DType::F32)?;

            let input_dim = match tensor.dims() {
                [_, d1, ..] => *d1,
                [d0] => *d0,
                dims => *dims
                    .last()
                    .ok_or_else(|| anyhow!("tensor {name} has no dims"))?,
            };
            let mut input_vec = vec![0.0f32; input_dim.max(1)];
            for (idx, byte) in input.as_bytes().iter().enumerate() {
                let slot = idx % input_vec.len();
                input_vec[slot] += *byte as f32 / 255.0;
            }
            let x = Tensor::from_slice(&input_vec, (1, input_vec.len()), &device)?;

            let out = match tensor.dims() {
                [d0, d1] => {
                    if *d1 == input_vec.len() {
                        let w = tensor.transpose(0, 1)?;
                        x.matmul(&w)?
                    } else if *d0 == input_vec.len() {
                        x.matmul(&tensor)?
                    } else {
                        return Err(anyhow!(
                            "input dim {} does not match tensor {} shape {:?}",
                            input_vec.len(),
                            name,
                            shape
                        ));
                    }
                }
                [d0] => {
                    if *d0 != input_vec.len() {
                        return Err(anyhow!(
                            "input dim {} does not match tensor {} shape {:?}",
                            input_vec.len(),
                            name,
                            shape
                        ));
                    }
                    x.mul(&tensor)?
                }
                _ => {
                    return Err(anyhow!(
                        "tensor {name} has unsupported shape {:?} for forward mode",
                        shape
                    ));
                }
            };
            let sum = out.sum_all()?.to_vec0::<f32>()?;
            Ok(format!(
                "forward peer={} layers={}..{} tensor={} shape={:?} input_dim={} sum={:.4}",
                shard.peer_id,
                shard.layer_start,
                shard.layer_end,
                name,
                shape,
                input_vec.len(),
                sum
            ))
        }
        "embed" => {
            let path = config
                .model_path
                .as_ref()
                .ok_or_else(|| anyhow!("model_path is required for embed mode"))?;
            let name = config
                .tensor_name
                .as_ref()
                .ok_or_else(|| anyhow!("--candle-tensor is required for embed mode"))?
                .clone();
            let data = std::fs::read(path)?;
            let safetensors = SliceSafetensors::new(&data)?;
            let tensor = safetensors.load(&name, &device)?.to_dtype(candle_core::DType::F32)?;

            let dims = tensor.dims();
            if dims.len() != 2 {
                return Err(anyhow!(
                    "embed mode expects 2D tensor, got shape {:?} for {name}",
                    dims
                ));
            }
            let vocab = dims[0].max(1);
            let hidden = dims[1].max(1);

            let bytes = input.as_bytes();
            let seq_len = bytes
                .len()
                .min(request.max_tokens as usize)
                .max(1);
            let mut ids = Vec::with_capacity(seq_len);
            for i in 0..seq_len {
                let b = if bytes.is_empty() { 0 } else { bytes[i % bytes.len()] };
                ids.push((b as usize % vocab) as u32);
            }
            let ids = Tensor::new(ids.as_slice(), &device)?;
            let embeds = tensor.embedding(&ids)?;
            let sum = embeds.sum_all()?.to_vec0::<f32>()?;
            Ok(format!(
                "embed peer={} layers={}..{} tensor={} vocab={} hidden={} seq_len={} sum={:.4}",
                shard.peer_id,
                shard.layer_start,
                shard.layer_end,
                name,
                vocab,
                hidden,
                seq_len,
                sum
            ))
        }
        other => Err(anyhow!(
            "unknown candle mode '{other}' (expected simulate, mlp, weights, forward, or embed)"
        )),
    }
}
