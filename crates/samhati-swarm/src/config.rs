use crate::types::SwarmConfig;
use std::time::Duration;

/// Builder for [`SwarmConfig`] with sensible defaults.
pub struct SwarmConfigBuilder {
    config: SwarmConfig,
}

impl SwarmConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: SwarmConfig::default(),
        }
    }

    /// Maximum wall-clock time to wait for the slowest node during fan-out.
    pub fn max_fan_out_timeout(mut self, d: Duration) -> Self {
        self.config.max_fan_out_timeout = d;
        self
    }

    /// Minimum number of valid (proof-verified) responses needed to proceed.
    pub fn min_responses(mut self, n: usize) -> Self {
        self.config.min_responses = n;
        self
    }

    /// If true, responses without a valid TOPLOC proof are discarded.
    pub fn toploc_required(mut self, required: bool) -> Self {
        self.config.toploc_required = required;
        self
    }

    /// Minimum BradleyTerry winner probability to emit a training example.
    pub fn training_confidence_threshold(mut self, t: f32) -> Self {
        self.config.training_confidence_threshold = t;
        self
    }

    /// Consume the builder and return the config.
    pub fn build(self) -> SwarmConfig {
        self.config
    }
}

impl Default for SwarmConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}
