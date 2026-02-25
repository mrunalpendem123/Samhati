use anyhow::{anyhow, Result};
use async_trait::async_trait;

use crate::plan::ShardSpec;
use crate::{InferenceRequest, ShardExecutor};

#[derive(Debug, Clone)]
pub struct ModelShardRunnerConfig {
    pub model_path: Option<String>,
    pub layer_start: usize,
    pub layer_end: usize,
    /// "ndarray" | "wgpu"
    pub backend: String,
    pub mode: String,
    pub hidden: usize,
    pub tensor_name: Option<String>,
    pub sample_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct ModelShardRunner {
    config: ModelShardRunnerConfig,
}

impl ModelShardRunner {
    pub fn new(config: ModelShardRunnerConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &ModelShardRunnerConfig {
        &self.config
    }
}

#[async_trait]
impl ShardExecutor for ModelShardRunner {
    async fn run_shard(
        &self,
        shard: &ShardSpec,
        _input: &str,
        _request: &InferenceRequest,
    ) -> Result<String> {
        #[cfg(feature = "burn")]
        {
            return run_simulated(&self.config, shard, _input, _request);
        }

        #[cfg(not(feature = "burn"))]
        Err(anyhow!(
            "burn feature not enabled (peer={}, layers={}..{})",
            shard.peer_id,
            shard.layer_start,
            shard.layer_end
        ))
    }
}

#[cfg(feature = "burn")]
fn run_simulated(
    config: &ModelShardRunnerConfig,
    shard: &ShardSpec,
    input: &str,
    request: &InferenceRequest,
) -> Result<String> {
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::tensor::{activation, Tensor, TensorData};

    let device = NdArrayDevice::Cpu;

    match config.mode.as_str() {
        "simulate" => {
            let mut x = Tensor::<NdArray, 1>::from_data(
                TensorData::new(vec![input.len() as f32], [1usize]),
                &device,
            );
            for layer in shard.layer_start..shard.layer_end {
                let add = Tensor::<NdArray, 1>::from_data(
                    TensorData::new(vec![layer as f32], [1usize]),
                    &device,
                );
                x = x.add(add);
            }
            let data = x.into_data();
            let vals: Vec<f32> = data.to_vec().unwrap_or_default();
            let val = vals.first().copied().unwrap_or(0.0);
            Ok(format!(
                "simulated-shard peer={} layers={}..{} input_len={} max_tokens={} temp={:.2} value={:.2}",
                shard.peer_id,
                shard.layer_start,
                shard.layer_end,
                input.len(),
                request.max_tokens,
                request.temperature,
                val
            ))
        }
        "mlp" => {
            let hidden = config.hidden.max(1);
            let mut features = vec![0.0f32; hidden];
            for (idx, byte) in input.as_bytes().iter().enumerate() {
                let slot = idx % hidden;
                features[slot] += *byte as f32 / 255.0;
            }

            let mut x = Tensor::<NdArray, 2>::from_data(
                TensorData::new(features, [1usize, hidden]),
                &device,
            );
            let mut weights = vec![0.0f32; hidden * hidden];
            for i in 0..hidden {
                for j in 0..hidden {
                    let idx = i * hidden + j;
                    weights[idx] = (idx as f32 + 1.0) / (hidden * hidden) as f32;
                }
            }
            let w = Tensor::<NdArray, 2>::from_data(
                TensorData::new(weights, [hidden, hidden]),
                &device,
            );

            for _ in shard.layer_start..shard.layer_end {
                x = x.matmul(w.clone());
                x = activation::tanh(x);
            }

            let sum: f32 = x.sum().into_scalar();
            Ok(format!(
                "burn-mlp peer={} layers={}..{} hidden={} input_len={} max_tokens={} temp={:.2} sum={:.4}",
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
            let bytes = std::fs::read(path)?;
            let st = safetensors::SafeTensors::deserialize(&bytes)?;
            let (name, shape) = if let Some(name) = &config.tensor_name {
                let view = st.tensor(name)?;
                (name.clone(), view.shape().to_vec())
            } else {
                let mut best: Option<(String, Vec<usize>, usize)> = None;
                for (name, view) in st.tensors() {
                    let shape = view.shape().to_vec();
                    let elems = shape.iter().product::<usize>().max(1);
                    let replace = match &best {
                        None => true,
                        Some((_, _, best_elems)) => elems < *best_elems,
                    };
                    if replace {
                        best = Some((name.to_string(), shape, elems));
                    }
                }
                let (name, shape, _) =
                    best.ok_or_else(|| anyhow!("no tensors found in {path}"))?;
                (name, shape)
            };
            let view = st.tensor(&name)?;
            let f32_data: Vec<f32> = view
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let len = f32_data.len().min(16);
            let slice_sum: f32 = f32_data[..len].iter().sum();
            Ok(format!(
                "weights peer={} layers={}..{} tensor={} shape={:?} slice_sum={:.4}",
                shard.peer_id,
                shard.layer_start,
                shard.layer_end,
                name,
                shape,
                slice_sum
            ))
        }
        "forward" => {
            let path = config
                .model_path
                .as_ref()
                .ok_or_else(|| anyhow!("model_path is required for forward mode"))?;
            let bytes = std::fs::read(path)?;
            let st = safetensors::SafeTensors::deserialize(&bytes)?;
            let (name, shape) = if let Some(name) = &config.tensor_name {
                let view = st.tensor(name)?;
                (name.clone(), view.shape().to_vec())
            } else {
                let mut best: Option<(String, Vec<usize>, usize)> = None;
                for (name, view) in st.tensors() {
                    let shape = view.shape().to_vec();
                    let elems = shape.iter().product::<usize>().max(1);
                    let replace = match &best {
                        None => true,
                        Some((_, _, best_elems)) => elems < *best_elems,
                    };
                    if replace {
                        best = Some((name.to_string(), shape, elems));
                    }
                }
                let (name, shape, _) =
                    best.ok_or_else(|| anyhow!("no tensors found in {path}"))?;
                (name, shape)
            };
            let view = st.tensor(&name)?;
            let f32_data: Vec<f32> = view
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();

            let input_dim = match shape.as_slice() {
                [_, d1, ..] => *d1,
                [d0] => *d0,
                _ => {
                    *shape
                        .last()
                        .ok_or_else(|| anyhow!("tensor {name} has no dims"))?
                }
            };
            let mut input_vec = vec![0.0f32; input_dim.max(1)];
            for (idx, byte) in input.as_bytes().iter().enumerate() {
                let slot = idx % input_vec.len();
                input_vec[slot] += *byte as f32 / 255.0;
            }

            let x = Tensor::<NdArray, 2>::from_data(
                TensorData::new(input_vec.clone(), [1usize, input_vec.len()]),
                &device,
            );

            let sum: f32 = match shape.as_slice() {
                [d0, d1] => {
                    if *d1 == input_vec.len() {
                        // weight is [out, in] → transpose to [in, out]
                        let w = Tensor::<NdArray, 2>::from_data(
                            TensorData::new(f32_data, [*d0, *d1]),
                            &device,
                        )
                        .transpose();
                        x.matmul(w).sum().into_scalar()
                    } else if *d0 == input_vec.len() {
                        let w = Tensor::<NdArray, 2>::from_data(
                            TensorData::new(f32_data, [*d0, *d1]),
                            &device,
                        );
                        x.matmul(w).sum().into_scalar()
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
                    let w = Tensor::<NdArray, 2>::from_data(
                        TensorData::new(f32_data, [1usize, *d0]),
                        &device,
                    );
                    x.mul(w).sum().into_scalar()
                }
                _ => {
                    return Err(anyhow!(
                        "tensor {name} has unsupported shape {:?} for forward mode",
                        shape
                    ));
                }
            };

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
                .ok_or_else(|| anyhow!("--model-tensor is required for embed mode"))?
                .clone();
            let bytes = std::fs::read(path)?;
            let st = safetensors::SafeTensors::deserialize(&bytes)?;
            let view = st.tensor(&name)?;
            let f32_data: Vec<f32> = view
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let shape = view.shape().to_vec();
            if shape.len() != 2 {
                return Err(anyhow!(
                    "embed mode expects 2D tensor, got shape {:?} for {name}",
                    shape
                ));
            }
            let vocab = shape[0].max(1);
            let hidden = shape[1].max(1);

            let input_bytes = input.as_bytes();
            let seq_len = input_bytes
                .len()
                .min(request.max_tokens as usize)
                .max(1);
            let mut sum = 0.0f32;
            for i in 0..seq_len {
                let b = if input_bytes.is_empty() {
                    0
                } else {
                    input_bytes[i % input_bytes.len()]
                };
                let row_start = (b as usize % vocab) * hidden;
                let row_end = (row_start + hidden).min(f32_data.len());
                sum += f32_data[row_start..row_end].iter().sum::<f32>();
            }
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
            "unknown mode '{other}' (expected simulate, mlp, weights, forward, or embed)"
        )),
    }
}
