"""Pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PipelineConfig:
    """Central configuration for the Samhati training pipeline."""

    # Data collection
    data_dir: Path = Path("data/rounds")
    min_confidence: float = 0.70

    # Training
    base_model: str = "Qwen/Qwen2.5-3B"
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 2048
    bf16: bool = True

    # DPO
    dpo_beta: float = 0.1
    dpo_loss_type: str = "sigmoid"
    dpo_learning_rate: float = 5e-5
    dpo_epochs: int = 1
    dpo_batch_size: int = 2
    dpo_gradient_accumulation_steps: int = 16

    # Export
    output_dir: Path = Path("models/")
    gguf_quantization: str = "Q4_K_M"
    hf_repo_prefix: str = "samhati"

    # Scheduler
    training_interval_days: int = 30
    min_examples_to_train: int = 5000

    # Domains
    domains: list[str] = field(default_factory=lambda: [
        "general", "code-rust", "code-python",
        "math", "science", "defi", "legal",
    ])

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)

    @classmethod
    def from_yaml(cls, path: Path) -> PipelineConfig:
        """Load configuration from a YAML file, merging with defaults."""
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


@dataclass
class DomainConfig:
    """Per-domain training configuration loaded from domains.yaml."""

    name: str
    description: str = ""
    base_model: str = "Qwen/Qwen2.5-3B"
    keywords: list[str] = field(default_factory=list)
    min_examples: int = 5000

    @staticmethod
    def load_all(path: Path) -> dict[str, DomainConfig]:
        """Load all domain configs from a YAML file."""
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}

        domains: dict[str, DomainConfig] = {}
        for name, cfg in raw.get("domains", {}).items():
            domains[name] = DomainConfig(
                name=name,
                description=cfg.get("description", ""),
                base_model=cfg.get("base_model", "Qwen/Qwen2.5-3B"),
                keywords=cfg.get("keywords", []),
                min_examples=cfg.get("min_examples", 5000),
            )
        return domains
