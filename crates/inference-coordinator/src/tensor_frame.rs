//! Wire format for activation tensors passed between mesh shard nodes.
//!
//! `TensorFrame` carries shape + element bytes + a sequence-position offset
//! so every downstream shard can correctly align its KV cache and RoPE embeddings.

use anyhow::{anyhow, Result};
use half::f16;
use serde::{Deserialize, Serialize};

/// Supported on-wire element types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WireDType {
    F32,
    F16,
}

/// A typed, serialisable tensor passed between nodes.
///
/// Data is stored as flat little-endian F32 bytes regardless of the
/// original candle dtype (we always cast to F32 before sending).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorFrame {
    /// Logical shape, e.g. `[batch, seq_len, hidden_size]`.
    pub shape: Vec<usize>,
    pub dtype: WireDType,
    /// Flat little-endian f32 bytes.
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
    /// How many tokens have already been processed for this session.
    /// Each shard uses this to compute the correct RoPE positions.
    pub seq_offset: usize,
}

impl TensorFrame {
    /// Build from a flat `f32` slice (stored as F32 on wire).
    pub fn from_f32(data: &[f32], shape: Vec<usize>, seq_offset: usize) -> Self {
        let bytes = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        Self { shape, dtype: WireDType::F32, data: bytes, seq_offset }
    }

    /// Build from a flat `f32` slice but store as F16 on the wire (halves bandwidth).
    ///
    /// Each f32 is down-cast to IEEE 754 half-precision.  For activation tensors
    /// this is lossless in practice — Petals confirmed "dynamic blockwise
    /// quantisation of hidden states halves bandwidth without any noticeable
    /// effect on generation quality".
    pub fn from_f32_as_f16(data: &[f32], shape: Vec<usize>, seq_offset: usize) -> Self {
        let bytes: Vec<u8> = data
            .iter()
            .flat_map(|&v| f16::from_f32(v).to_le_bytes())
            .collect();
        Self { shape, dtype: WireDType::F16, data: bytes, seq_offset }
    }

    /// Build directly from half-precision bytes.
    pub fn from_f16_bytes(data: Vec<u8>, shape: Vec<usize>, seq_offset: usize) -> Self {
        Self { shape, dtype: WireDType::F16, data, seq_offset }
    }

    /// Decode back to `Vec<f32>`, handling both F32 and F16 wire formats.
    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        match self.dtype {
            WireDType::F32 => {
                if self.data.len() % 4 != 0 {
                    return Err(anyhow!(
                        "TensorFrame F32 data length {} is not a multiple of 4",
                        self.data.len()
                    ));
                }
                Ok(self
                    .data
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect())
            }
            WireDType::F16 => {
                if self.data.len() % 2 != 0 {
                    return Err(anyhow!(
                        "TensorFrame F16 data length {} is not a multiple of 2",
                        self.data.len()
                    ));
                }
                Ok(self
                    .data
                    .chunks_exact(2)
                    .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
                    .collect())
            }
        }
    }

    /// Convert this frame to F16 wire format (no-op if already F16).
    pub fn to_f16_wire(&self) -> Result<Self> {
        match self.dtype {
            WireDType::F16 => Ok(self.clone()),
            WireDType::F32 => {
                let f32s = self.to_f32_vec()?;
                Ok(Self::from_f32_as_f16(&f32s, self.shape.clone(), self.seq_offset))
            }
        }
    }

    /// Convert this frame to F32 wire format (no-op if already F32).
    pub fn to_f32_wire(&self) -> Result<Self> {
        match self.dtype {
            WireDType::F32 => Ok(self.clone()),
            WireDType::F16 => {
                let f32s = self.to_f32_vec()?;
                Ok(Self::from_f32(&f32s, self.shape.clone(), self.seq_offset))
            }
        }
    }

    /// Total number of elements (`shape.iter().product()`).
    pub fn numel(&self) -> usize {
        self.shape.iter().product::<usize>().max(1)
    }

    /// Size in bytes of one element for the current dtype.
    pub fn element_size(&self) -> usize {
        match self.dtype {
            WireDType::F32 => 4,
            WireDType::F16 => 2,
        }
    }

    /// Total wire size of the tensor data in bytes.
    pub fn wire_size(&self) -> usize {
        self.data.len()
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        Ok(bincode::serialize(self)?)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Ok(bincode::deserialize(bytes)?)
    }
}

// ── Burn integration ──────────────────────────────────────────────────────────

#[cfg(feature = "burn")]
impl TensorFrame {
    /// Snapshot a Burn tensor (any backend, any dim) into a wire frame.
    pub fn from_burn<B: burn::tensor::backend::Backend, const D: usize>(
        t: burn::tensor::Tensor<B, D>,
        seq_offset: usize,
    ) -> Result<Self> {
        let shape: Vec<usize> = t.dims().iter().copied().collect();
        let data = t.into_data();
        let f32s: Vec<f32> = data
            .to_vec()
            .map_err(|e| anyhow!("tensor data convert: {e:?}"))?;
        Ok(Self::from_f32(&f32s, shape, seq_offset))
    }

    /// Materialise a 3-D Burn tensor from this frame.
    pub fn to_burn_3d<B: burn::tensor::backend::Backend>(
        &self,
        device: &B::Device,
    ) -> Result<burn::tensor::Tensor<B, 3>> {
        if self.shape.len() != 3 {
            return Err(anyhow!(
                "expected 3D tensor frame, got shape {:?}",
                self.shape
            ));
        }
        let f32s = self.to_f32_vec()?;
        Ok(burn::tensor::Tensor::<B, 3>::from_data(
            burn::tensor::TensorData::new(f32s, [self.shape[0], self.shape[1], self.shape[2]]),
            device,
        ))
    }

    /// Materialise a 2-D Burn tensor from this frame.
    pub fn to_burn_2d<B: burn::tensor::backend::Backend>(
        &self,
        device: &B::Device,
    ) -> Result<burn::tensor::Tensor<B, 2>> {
        if self.shape.len() < 2 {
            return Err(anyhow!(
                "expected at least 2D tensor frame, got shape {:?}",
                self.shape
            ));
        }
        let f32s = self.to_f32_vec()?;
        Ok(burn::tensor::Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(f32s, [self.shape[0], self.shape[1]]),
            device,
        ))
    }

    /// Materialise a 1-D Burn tensor from this frame.
    pub fn to_burn_1d<B: burn::tensor::backend::Backend>(
        &self,
        device: &B::Device,
    ) -> Result<burn::tensor::Tensor<B, 1>> {
        let f32s = self.to_f32_vec()?;
        let len = f32s.len();
        Ok(burn::tensor::Tensor::<B, 1>::from_data(
            burn::tensor::TensorData::new(f32s, [len]),
            device,
        ))
    }
}
