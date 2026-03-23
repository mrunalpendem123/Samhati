"""Monthly training scheduler.

Orchestrates the full self-evolving training cycle: collect data, train SFT,
train DPO, merge adapters, export GGUF, push to HuggingFace.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from samhati_pipeline.collector import load_examples, stats
from samhati_pipeline.config import DomainConfig, PipelineConfig
from samhati_pipeline.dataset import build_dpo_dataset, build_sft_dataset
from samhati_pipeline.exporter import (
    export_gguf,
    generate_model_card,
    notify_nodes,
    push_to_hub,
)
from samhati_pipeline.trainer import SamhatiTrainer

logger = logging.getLogger(__name__)

_STATE_FILE = ".scheduler_state.json"


def _load_state(output_dir: Path) -> dict[str, Any]:
    """Load scheduler state (last training timestamps per domain)."""
    state_path = output_dir / _STATE_FILE
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return {}


def _save_state(output_dir: Path, state: dict[str, Any]) -> None:
    """Persist scheduler state."""
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / _STATE_FILE
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


class TrainingScheduler:
    """Manages periodic training runs across domains."""

    def __init__(self, config: PipelineConfig, domain_configs: Optional[dict[str, DomainConfig]] = None) -> None:
        self.config = config
        self.domain_configs = domain_configs or {}
        self.trainer = SamhatiTrainer(config)

    def check_and_run(self) -> dict[str, Any]:
        """Check all domains and trigger training where enough data has accumulated.

        Returns:
            Summary dict with per-domain results.
        """
        cfg = self.config
        state = _load_state(cfg.output_dir)
        data_stats = stats(cfg.data_dir)
        results: dict[str, Any] = {}

        for domain in cfg.domains:
            domain_info = data_stats.get("domains", {}).get(domain, {})
            count = domain_info.get("count", 0)

            # Check minimum examples
            min_needed = cfg.min_examples_to_train
            if domain in self.domain_configs:
                min_needed = self.domain_configs[domain].min_examples
            if count < min_needed:
                logger.info(
                    "Domain %s: %d examples (need %d), skipping.",
                    domain, count, min_needed,
                )
                results[domain] = {"status": "skipped", "reason": "insufficient_data", "count": count}
                continue

            # Check if enough time has passed since last training
            last_trained = state.get(domain, {}).get("last_trained")
            if last_trained:
                last_dt = datetime.fromisoformat(last_trained)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                cutoff = datetime.now(timezone.utc) - timedelta(days=cfg.training_interval_days)
                if last_dt > cutoff:
                    logger.info("Domain %s: trained recently (%s), skipping.", domain, last_trained)
                    results[domain] = {"status": "skipped", "reason": "recently_trained"}
                    continue

            # Run the pipeline
            try:
                result = self.run_domain_pipeline(domain)
                state.setdefault(domain, {})["last_trained"] = datetime.now(timezone.utc).isoformat()
                _save_state(cfg.output_dir, state)
                results[domain] = {"status": "success", **result}
            except Exception as exc:
                logger.exception("Domain %s: training failed", domain)
                results[domain] = {"status": "error", "error": str(exc)}

        return results

    def run_domain_pipeline(
        self,
        domain: str,
        base_model: Optional[str] = None,
    ) -> dict[str, Any]:
        """Execute the full training pipeline for a single domain.

        Steps:
            1. Load examples since last training
            2. Build SFT dataset
            3. Train SFT with QLoRA
            4. Build DPO dataset
            5. Train DPO on top of SFT
            6. Merge adapters
            7. Export GGUF
            8. Push to HuggingFace
            9. Notify nodes

        Args:
            domain: Domain name.
            base_model: Optional base model override.

        Returns:
            Dict with paths, stats, and timing information.
        """
        cfg = self.config
        result: dict[str, Any] = {"domain": domain}
        start_time = time.monotonic()

        # Resolve base model
        if base_model is None and domain in self.domain_configs:
            base_model = self.domain_configs[domain].base_model
        effective_model = base_model or cfg.base_model
        result["base_model"] = effective_model

        # 1. Load examples
        state = _load_state(cfg.output_dir)
        last_trained = state.get(domain, {}).get("last_trained")
        min_date = datetime.fromisoformat(last_trained) if last_trained else None
        examples = load_examples(cfg.data_dir, domain, min_date=min_date)
        logger.info("Domain %s: loaded %d examples", domain, len(examples))
        result["total_examples"] = len(examples)

        if not examples:
            raise ValueError(f"No examples found for domain {domain}")

        # 2. Build SFT dataset
        sft_dataset = build_sft_dataset(examples, model_name=effective_model)
        result["sft_examples"] = len(sft_dataset)
        logger.info("Domain %s: SFT dataset has %d rows", domain, len(sft_dataset))

        # 3. Train SFT
        sft_adapter = self.trainer.train_sft(sft_dataset, domain, base_model=effective_model)
        result["sft_adapter"] = sft_adapter

        # 4. Build DPO dataset
        dpo_dataset = build_dpo_dataset(examples, model_name=effective_model)
        result["dpo_examples"] = len(dpo_dataset)
        logger.info("Domain %s: DPO dataset has %d rows", domain, len(dpo_dataset))

        # 5. Train DPO
        if len(dpo_dataset) > 0:
            dpo_adapter = self.trainer.train_dpo(
                dpo_dataset, domain, sft_adapter, base_model=effective_model,
            )
            result["dpo_adapter"] = dpo_adapter
            final_adapter = dpo_adapter
        else:
            logger.warning("Domain %s: no DPO pairs, using SFT only", domain)
            final_adapter = sft_adapter

        # 6. Merge adapters
        merged_path = SamhatiTrainer.merge_adapter(effective_model, final_adapter)
        result["merged_model"] = merged_path

        # 7. Export GGUF
        gguf_path = export_gguf(
            merged_path,
            str(cfg.output_dir / f"gguf-{domain}"),
            cfg.gguf_quantization,
        )
        result["gguf_path"] = gguf_path

        # 8. Push to HuggingFace
        version = f"v{datetime.now(timezone.utc).strftime('%Y%m%d')}"
        model_card = generate_model_card(
            domain,
            training_stats={
                "base_model": effective_model,
                "sft_examples": result["sft_examples"],
                "dpo_examples": result["dpo_examples"],
            },
        )
        hf_url = push_to_hub(gguf_path, cfg.hf_repo_prefix, domain, version, model_card)
        result["hf_url"] = hf_url

        # 9. Notify nodes
        notify_nodes({
            "domain": domain,
            "version": version,
            "hf_url": hf_url,
            "base_model": effective_model,
        })

        elapsed = time.monotonic() - start_time
        result["elapsed_seconds"] = round(elapsed, 1)
        logger.info("Domain %s: pipeline complete in %.1fs", domain, elapsed)

        return result
