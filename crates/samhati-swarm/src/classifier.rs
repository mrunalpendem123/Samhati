use crate::types::Complexity;
use serde::{Deserialize, Serialize};

/// Result of prompt complexity classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityResult {
    /// Recommended swarm size (1, 3, 5, 7, or 9).
    pub n_nodes: usize,
    /// Complexity tier.
    pub complexity: Complexity,
    /// Detected domain tags (e.g. "code", "math", "hindi").
    pub domain_tags: Vec<String>,
    /// Classifier confidence in [0, 1].
    pub confidence: f32,
}

/// Fast, CPU-only heuristic classifier that maps a prompt to a [`Complexity`]
/// tier and a set of domain tags.  Designed to complete in < 1 ms.
pub struct ComplexityClassifier;

impl ComplexityClassifier {
    pub fn new() -> Self {
        Self
    }

    /// Classify a prompt and return the recommended swarm size + metadata.
    pub fn classify(&self, prompt: &str) -> ComplexityResult {
        let lower = prompt.to_lowercase();
        let token_count = prompt.split_whitespace().count();

        // --- base tier from token count ---
        let mut tier: u8 = if token_count < 10 {
            0 // Trivial
        } else if token_count < 50 {
            1 // Conversational
        } else if token_count < 200 {
            2 // Reasoning
        } else if token_count < 500 {
            3 // Hard
        } else {
            4 // Expert
        };

        let mut domain_tags: Vec<String> = Vec::new();

        // --- code indicators ---
        let code_signals = [
            "```", "fn ", "def ", "class ", "debug", "error", "fix",
            "impl ", "struct ", "enum ", "import ", "require(", "#include",
            "async fn", "pub fn",
        ];
        let has_code = code_signals.iter().any(|s| lower.contains(s));
        if has_code {
            tier = tier.saturating_add(1).min(4);
            domain_tags.push("code".into());
        }

        // more specific code domain tags
        let rust_keywords = ["fn ", "impl ", "struct ", "enum ", "cargo", "crate", "borrow checker"];
        if rust_keywords.iter().any(|k| lower.contains(k)) {
            if !domain_tags.contains(&"rust".to_string()) {
                domain_tags.push("rust".into());
            }
        }
        let python_kw = ["def ", "import ", "python", "pip ", "numpy", "torch"];
        if python_kw.iter().any(|k| lower.contains(k)) {
            if !domain_tags.contains(&"python".to_string()) {
                domain_tags.push("python".into());
            }
        }

        // --- math indicators ---
        let math_signals = [
            "prove", "solve", "integral", "derivative", "equation",
            "theorem", "matrix", "eigenvalue", "polynomial", "calculus",
            "summation", "limit", "convergence",
        ];
        let has_math = math_signals.iter().any(|s| lower.contains(s));
        let has_math_symbols = prompt.contains('∫') || prompt.contains('∑')
            || prompt.contains('∂') || prompt.contains('√')
            || prompt.contains('≤') || prompt.contains('≥');
        if has_math || has_math_symbols {
            tier = tier.saturating_add(1).min(4);
            domain_tags.push("math".into());
        }

        // --- multi-step reasoning indicators ---
        let reasoning_signals = [
            "step by step", "explain why", "compare", "contrast",
            "analyze", "evaluate", "critically", "trade-off",
            "pros and cons", "reasoning", "chain of thought",
        ];
        if reasoning_signals.iter().any(|s| lower.contains(s)) {
            tier = tier.saturating_add(1).min(4);
        }

        // --- Hindi / Indic language detection ---
        let hindi_indicators = [
            "कृपया", "बताइए", "समझाइए", "हिंदी", "हिन्दी",
            "क्या", "कैसे", "क्यों", "मुझे", "यह",
        ];
        if hindi_indicators.iter().any(|s| prompt.contains(s)) {
            domain_tags.push("hindi".into());
        }

        // --- question complexity bump ---
        let question_marks = prompt.chars().filter(|&c| c == '?').count();
        if question_marks >= 3 {
            tier = tier.saturating_add(1).min(4);
        }

        let complexity = match tier {
            0 => Complexity::Trivial,
            1 => Complexity::Conversational,
            2 => Complexity::Reasoning,
            3 => Complexity::Hard,
            _ => Complexity::Expert,
        };

        // Confidence: higher when signals agree with token-count heuristic.
        // Simple proxy: how many signal categories fired.
        let signals_fired = has_code as u8
            + has_math as u8
            + has_math_symbols as u8
            + (question_marks >= 3) as u8
            + reasoning_signals.iter().any(|s| lower.contains(s)) as u8;
        let confidence = if signals_fired == 0 {
            0.85 // pure token-length, usually correct for trivial/conversational
        } else {
            (0.70 + 0.06 * signals_fired as f32).min(1.0)
        };

        ComplexityResult {
            n_nodes: complexity.n_nodes(),
            complexity,
            domain_tags,
            confidence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classifier() -> ComplexityClassifier {
        ComplexityClassifier::new()
    }

    #[test]
    fn trivial_prompt() {
        let r = classifier().classify("hi");
        assert_eq!(r.complexity, Complexity::Trivial);
        assert_eq!(r.n_nodes, 1);
    }

    #[test]
    fn conversational_prompt() {
        let r = classifier().classify("What is the capital of France and can you tell me a fun fact about it?");
        assert!(matches!(r.complexity, Complexity::Conversational | Complexity::Reasoning));
        assert!(r.n_nodes >= 3);
    }

    #[test]
    fn code_prompt_bumps_up() {
        let r = classifier().classify("fn main() { debug this error }");
        assert!(r.domain_tags.contains(&"code".to_string()));
        // code signals should bump beyond Trivial even though token count is low
        assert!(r.n_nodes >= 3);
    }

    #[test]
    fn math_prompt() {
        let r = classifier().classify("Prove that the integral of e^(-x^2) from -inf to inf equals sqrt(pi).");
        assert!(r.domain_tags.contains(&"math".to_string()));
        assert!(r.n_nodes >= 5);
    }

    #[test]
    fn rust_domain_detected() {
        let r = classifier().classify("Explain how the borrow checker works in Rust and impl a struct with lifetimes");
        assert!(r.domain_tags.contains(&"rust".to_string()));
        assert!(r.domain_tags.contains(&"code".to_string()));
    }

    #[test]
    fn hindi_detected() {
        let r = classifier().classify("कृपया मुझे बताइए कि भारत की राजधानी क्या है");
        assert!(r.domain_tags.contains(&"hindi".to_string()));
    }

    #[test]
    fn multi_step_reasoning() {
        let r = classifier().classify("Compare and contrast TCP and UDP. Explain why each protocol is preferred in different scenarios step by step.");
        // reasoning signals + multi-sentence → at least Reasoning
        assert!(r.n_nodes >= 5);
    }

    #[test]
    fn expert_long_prompt() {
        let words: String = (0..600).map(|i| format!("word{i}")).collect::<Vec<_>>().join(" ");
        let r = classifier().classify(&words);
        assert_eq!(r.complexity, Complexity::Expert);
        assert_eq!(r.n_nodes, 9);
    }

    #[test]
    fn user_override_n_values() {
        // n_nodes from complexity is deterministic
        assert_eq!(Complexity::Trivial.n_nodes(), 1);
        assert_eq!(Complexity::Conversational.n_nodes(), 3);
        assert_eq!(Complexity::Reasoning.n_nodes(), 5);
        assert_eq!(Complexity::Hard.n_nodes(), 7);
        assert_eq!(Complexity::Expert.n_nodes(), 9);
    }
}
