"""QLoRA fine-tuning for domain specialist SLMs.

Wraps HuggingFace trl SFTTrainer and DPOTrainer with 4-bit quantized base
models, LoRA adapters, and optional Wandb logging.  Designed to run on a
single RTX 3090 (24 GB VRAM).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

from samhati_pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)

# LoRA target modules shared across Qwen2.5 and Llama-3.2
_DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def _bnb_config() -> BitsAndBytesConfig:
    """4-bit quantization config for QLoRA training."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def _load_base_model(model_name: str) -> AutoModelForCausalLM:
    """Load a 4-bit quantized base model."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    return model


def _load_tokenizer(model_name: str, max_seq_length: int) -> AutoTokenizer:
    """Load and configure the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max_seq_length
    return tokenizer


class SamhatiTrainer:
    """Orchestrates QLoRA SFT and DPO training."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def train_sft(
        self,
        dataset: Dataset,
        domain: str,
        base_model: Optional[str] = None,
    ) -> str:
        """Run supervised fine-tuning with QLoRA.

        Args:
            dataset: HuggingFace Dataset with a ``text`` column.
            domain: Domain name (used in output path).
            base_model: Override base model name.

        Returns:
            Path to the saved LoRA adapter directory.
        """
        cfg = self.config
        model_name = base_model or cfg.base_model
        output_path = str(cfg.output_dir / f"sft-{domain}")

        logger.info("Loading base model %s for SFT...", model_name)
        model = _load_base_model(model_name)
        tokenizer = _load_tokenizer(model_name, cfg.max_seq_length)

        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=_DEFAULT_TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM",
        )

        training_args = SFTConfig(
            output_dir=output_path,
            num_train_epochs=cfg.num_epochs,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            bf16=cfg.bf16,
            logging_steps=10,
            save_strategy="epoch",
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            max_seq_length=cfg.max_seq_length,
            report_to="wandb",
            run_name=f"samhati-sft-{domain}",
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=lora_config,
        )

        logger.info("Starting SFT training for domain=%s, examples=%d", domain, len(dataset))
        trainer.train()
        trainer.save_model(output_path)
        tokenizer.save_pretrained(output_path)

        logger.info("SFT adapter saved to %s", output_path)
        return output_path

    def train_dpo(
        self,
        dataset: Dataset,
        domain: str,
        sft_adapter: str,
        base_model: Optional[str] = None,
    ) -> str:
        """Run DPO training on top of an SFT adapter.

        Args:
            dataset: HuggingFace Dataset with ``prompt``, ``chosen``, ``rejected``.
            domain: Domain name.
            sft_adapter: Path to the SFT LoRA adapter.
            base_model: Override base model name.

        Returns:
            Path to the saved DPO adapter directory.
        """
        cfg = self.config
        model_name = base_model or cfg.base_model
        output_path = str(cfg.output_dir / f"dpo-{domain}")

        logger.info("Loading base model %s with SFT adapter for DPO...", model_name)
        base = _load_base_model(model_name)
        model = PeftModel.from_pretrained(base, sft_adapter)
        model = model.merge_and_unload()
        model = prepare_model_for_kbit_training(model)

        tokenizer = _load_tokenizer(model_name, cfg.max_seq_length)

        lora_config = LoraConfig(
            r=cfg.lora_r // 2,  # Smaller rank for DPO stage
            lora_alpha=cfg.lora_alpha // 2,
            lora_dropout=cfg.lora_dropout,
            target_modules=_DEFAULT_TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        training_args = DPOConfig(
            output_dir=output_path,
            num_train_epochs=cfg.dpo_epochs,
            per_device_train_batch_size=cfg.dpo_batch_size,
            gradient_accumulation_steps=cfg.dpo_gradient_accumulation_steps,
            learning_rate=cfg.dpo_learning_rate,
            bf16=cfg.bf16,
            logging_steps=10,
            save_strategy="epoch",
            beta=cfg.dpo_beta,
            loss_type=cfg.dpo_loss_type,
            report_to="wandb",
            run_name=f"samhati-dpo-{domain}",
        )

        trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        logger.info("Starting DPO training for domain=%s, examples=%d", domain, len(dataset))
        trainer.train()
        trainer.save_model(output_path)
        tokenizer.save_pretrained(output_path)

        logger.info("DPO adapter saved to %s", output_path)
        return output_path

    @staticmethod
    def merge_adapter(base_model: str, adapter_path: str, output_path: Optional[str] = None) -> str:
        """Merge LoRA weights into the base model and save full weights.

        Args:
            base_model: HuggingFace model identifier.
            adapter_path: Path to saved LoRA adapter.
            output_path: Where to save merged model.  Defaults to
                         ``<adapter_path>-merged``.

        Returns:
            Path to the merged model directory.
        """
        if output_path is None:
            output_path = f"{adapter_path}-merged"

        logger.info("Merging adapter %s into %s...", adapter_path, base_model)
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, adapter_path)
        merged = model.merge_and_unload()

        merged.save_pretrained(output_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.save_pretrained(output_path)

        logger.info("Merged model saved to %s", output_path)
        return output_path
