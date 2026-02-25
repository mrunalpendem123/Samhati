use cluster_manager::{select_cluster, ClusterConstraints, ClusterSelection};
use model_config::{estimate_required_vram_gb, EstimateInput, ModelConfig};
use proximity_router::RankedPeer;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectionKey {
    pub model_name: String,
    pub nodes: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SelectionResult {
    pub key: SelectionKey,
    pub selection: ClusterSelection,
}

#[derive(Debug, Clone)]
pub enum SelectionError {
    NoCandidates,
    NoModelFits,
}

#[derive(Debug, Clone)]
pub struct ModelCandidate<'a> {
    pub name: &'a str,
    pub config: &'a ModelConfig,
    pub ranked: Vec<RankedPeer>,
}

pub fn select_best<'a>(
    candidates: &[ModelCandidate<'a>],
    constraints: &ClusterConstraints,
    estimate_input: &EstimateInput,
) -> Result<SelectionResult, SelectionError> {
    if candidates.is_empty() {
        return Err(SelectionError::NoCandidates);
    }

    for candidate in candidates {
        let result = select_cluster(&candidate.ranked, constraints, |nodes| {
            let mut estimate = estimate_input.clone();
            estimate.nodes = nodes;
            estimate_required_vram_gb(candidate.config, &estimate)
        });

        if let Ok(selection) = result {
            return Ok(SelectionResult {
                key: SelectionKey {
                    model_name: candidate.name.to_string(),
                    nodes: selection.nodes.clone(),
                },
                selection,
            });
        }
    }

    Err(SelectionError::NoModelFits)
}
