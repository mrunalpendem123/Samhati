//! Ring-buffer caches for fault-tolerant inference recovery.
//!
//! Implements the Petals dual-cache protocol (NeurIPS 2023):
//! - **Client-side activation cache**: stores the activations sent to each shard
//!   stage so they can be replayed to a replacement node on failure.
//! - **Server-side output cache**: stores the output activations after each
//!   forward pass so upstream nodes can replay them to a replacement.
//!
//! Memory costs are minimal (512 × ~5 KB = 2.5 MB client, 64 × ~5 KB = 320 KB
//! per server node for a 3B model) and provide sub-second recovery from node
//! dropout instead of full restarts.

use std::collections::VecDeque;

use crate::tensor_frame::TensorFrame;

// ── Client activation cache ─────────────────────────────────────────────────

/// A cached activation sent to a shard stage.
#[derive(Debug, Clone)]
pub struct CachedActivation {
    /// Inference session identifier.
    pub session_id: String,
    /// Index of the shard in the pipeline plan (0-based).
    pub shard_index: usize,
    /// Peer ID the activation was sent to.
    pub peer_id: String,
    /// Sequence offset at the time of sending.
    pub seq_offset: usize,
    /// Layer range this shard covers.
    pub layer_start: usize,
    pub layer_end: usize,
    /// The activation tensor that was sent.
    pub frame: TensorFrame,
}

/// Ring buffer of activations sent to shard stages for fault recovery.
///
/// When a peer fails mid-generation, the cached activations for that shard
/// can be replayed to a replacement node, which processes them sequentially
/// to rebuild its KV cache and resume generation.
#[derive(Debug, Clone)]
pub struct ActivationRingBuffer {
    buffer: VecDeque<CachedActivation>,
    capacity: usize,
}

impl ActivationRingBuffer {
    /// Create a new activation cache with the given capacity (default: 512).
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Push a new cached activation, evicting the oldest if over capacity.
    pub fn push(&mut self, entry: CachedActivation) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(entry);
    }

    /// Get all cached activations for a given session and shard index,
    /// in chronological order (oldest first).
    pub fn get_replay_for_shard(
        &self,
        session_id: &str,
        shard_index: usize,
    ) -> Vec<&CachedActivation> {
        self.buffer
            .iter()
            .filter(|a| a.session_id == session_id && a.shard_index == shard_index)
            .collect()
    }

    /// Get all cached activations for a given session and layer range,
    /// in chronological order (oldest first).
    pub fn get_replay_for_layers(
        &self,
        session_id: &str,
        layer_start: usize,
        layer_end: usize,
    ) -> Vec<&CachedActivation> {
        self.buffer
            .iter()
            .filter(|a| {
                a.session_id == session_id
                    && a.layer_start == layer_start
                    && a.layer_end == layer_end
            })
            .collect()
    }

    /// Remove all cached activations for a session (e.g. after completion).
    pub fn clear_session(&mut self, session_id: &str) {
        self.buffer.retain(|a| a.session_id != session_id);
    }

    /// Number of entries currently in the buffer.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

impl Default for ActivationRingBuffer {
    fn default() -> Self {
        Self::new(512)
    }
}

// ── Server output cache ─────────────────────────────────────────────────────

/// A cached output activation from a forward pass.
#[derive(Debug, Clone)]
pub struct CachedOutput {
    /// Inference session identifier.
    pub session_id: String,
    /// Sequence offset at the time of output.
    pub seq_offset: usize,
    /// The output activation tensor.
    pub frame: TensorFrame,
}

/// Ring buffer of output activations for server-side fault recovery.
///
/// Each shard node stores the outputs of its last N forward passes so that
/// if a downstream node fails, the outputs can be replayed to the replacement.
#[derive(Debug, Clone)]
pub struct OutputRingBuffer {
    buffer: VecDeque<CachedOutput>,
    capacity: usize,
}

impl OutputRingBuffer {
    /// Create a new output cache with the given capacity (default: 64).
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Push a new cached output, evicting the oldest if over capacity.
    pub fn push(&mut self, entry: CachedOutput) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(entry);
    }

    /// Get all cached outputs for a given session, in chronological order.
    pub fn get_replay(&self, session_id: &str) -> Vec<&CachedOutput> {
        self.buffer
            .iter()
            .filter(|o| o.session_id == session_id)
            .collect()
    }

    /// Remove all cached outputs for a session.
    pub fn clear_session(&mut self, session_id: &str) {
        self.buffer.retain(|o| o.session_id != session_id);
    }

    /// Number of entries currently in the buffer.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

impl Default for OutputRingBuffer {
    fn default() -> Self {
        Self::new(64)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_frame(seq_offset: usize) -> TensorFrame {
        TensorFrame::from_f32(&[1.0, 2.0], vec![1, 2], seq_offset)
    }

    #[test]
    fn activation_ring_buffer_push_and_retrieve() {
        let mut buf = ActivationRingBuffer::new(10);

        buf.push(CachedActivation {
            session_id: "s1".into(),
            shard_index: 0,
            peer_id: "p1".into(),
            seq_offset: 0,
            layer_start: 0,
            layer_end: 16,
            frame: dummy_frame(0),
        });
        buf.push(CachedActivation {
            session_id: "s1".into(),
            shard_index: 0,
            peer_id: "p1".into(),
            seq_offset: 5,
            layer_start: 0,
            layer_end: 16,
            frame: dummy_frame(5),
        });
        buf.push(CachedActivation {
            session_id: "s1".into(),
            shard_index: 1,
            peer_id: "p2".into(),
            seq_offset: 0,
            layer_start: 16,
            layer_end: 32,
            frame: dummy_frame(0),
        });

        let replay = buf.get_replay_for_shard("s1", 0);
        assert_eq!(replay.len(), 2);
        assert_eq!(replay[0].seq_offset, 0);
        assert_eq!(replay[1].seq_offset, 5);

        let replay = buf.get_replay_for_shard("s1", 1);
        assert_eq!(replay.len(), 1);
    }

    #[test]
    fn activation_ring_buffer_evicts_oldest() {
        let mut buf = ActivationRingBuffer::new(3);

        for i in 0..5 {
            buf.push(CachedActivation {
                session_id: "s1".into(),
                shard_index: 0,
                peer_id: "p1".into(),
                seq_offset: i,
                layer_start: 0,
                layer_end: 8,
                frame: dummy_frame(i),
            });
        }

        assert_eq!(buf.len(), 3);
        let replay = buf.get_replay_for_shard("s1", 0);
        // Should have offsets 2, 3, 4 (oldest two evicted)
        assert_eq!(replay[0].seq_offset, 2);
        assert_eq!(replay[1].seq_offset, 3);
        assert_eq!(replay[2].seq_offset, 4);
    }

    #[test]
    fn activation_ring_buffer_clear_session() {
        let mut buf = ActivationRingBuffer::new(10);
        buf.push(CachedActivation {
            session_id: "s1".into(),
            shard_index: 0,
            peer_id: "p1".into(),
            seq_offset: 0,
            layer_start: 0,
            layer_end: 8,
            frame: dummy_frame(0),
        });
        buf.push(CachedActivation {
            session_id: "s2".into(),
            shard_index: 0,
            peer_id: "p1".into(),
            seq_offset: 0,
            layer_start: 0,
            layer_end: 8,
            frame: dummy_frame(0),
        });

        buf.clear_session("s1");
        assert_eq!(buf.len(), 1);
        assert!(buf.get_replay_for_shard("s1", 0).is_empty());
        assert_eq!(buf.get_replay_for_shard("s2", 0).len(), 1);
    }

    #[test]
    fn output_ring_buffer_push_and_retrieve() {
        let mut buf = OutputRingBuffer::new(5);

        buf.push(CachedOutput {
            session_id: "s1".into(),
            seq_offset: 0,
            frame: dummy_frame(0),
        });
        buf.push(CachedOutput {
            session_id: "s1".into(),
            seq_offset: 1,
            frame: dummy_frame(1),
        });

        let replay = buf.get_replay("s1");
        assert_eq!(replay.len(), 2);
        assert_eq!(replay[0].seq_offset, 0);
        assert_eq!(replay[1].seq_offset, 1);
    }

    #[test]
    fn output_ring_buffer_evicts_oldest() {
        let mut buf = OutputRingBuffer::new(2);
        for i in 0..4 {
            buf.push(CachedOutput {
                session_id: "s1".into(),
                seq_offset: i,
                frame: dummy_frame(i),
            });
        }
        assert_eq!(buf.len(), 2);
        let replay = buf.get_replay("s1");
        assert_eq!(replay[0].seq_offset, 2);
        assert_eq!(replay[1].seq_offset, 3);
    }
}
