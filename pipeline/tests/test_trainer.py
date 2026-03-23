"""Tests for the trainer module.

These tests mock all external dependencies (transformers, peft, trl) so they
can run without a GPU or model downloads.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from samhati_pipeline.config import PipelineConfig
from samhati_pipeline.trainer import SamhatiTrainer, _bnb_config


@pytest.fixture
def config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        output_dir=tmp_path / "models",
        data_dir=tmp_path / "data",
        num_epochs=1,
        batch_size=2,
    )


@pytest.fixture
def sft_dataset() -> Dataset:
    return Dataset.from_list([
        {"text": "Hello world " * 20, "domain": "general", "confidence": 0.9, "round_id": "r1"},
        {"text": "Test data " * 20, "domain": "general", "confidence": 0.85, "round_id": "r2"},
    ])


@pytest.fixture
def dpo_dataset() -> Dataset:
    return Dataset.from_list([
        {
            "prompt": "What is Rust?",
            "chosen": "Rust is a systems programming language " * 5,
            "rejected": "Rust is a color " * 5,
            "domain": "code-rust",
            "confidence": 0.9,
            "round_id": "r1",
        },
    ])


class TestBnbConfig:
    def test_creates_config(self):
        cfg = _bnb_config()
        assert cfg.load_in_4bit is True
        assert cfg.bnb_4bit_quant_type == "nf4"


class TestSamhatiTrainer:
    @patch("samhati_pipeline.trainer.SFTTrainer")
    @patch("samhati_pipeline.trainer._load_tokenizer")
    @patch("samhati_pipeline.trainer._load_base_model")
    def test_train_sft(
        self,
        mock_load_model: MagicMock,
        mock_load_tokenizer: MagicMock,
        mock_sft_trainer_cls: MagicMock,
        config: PipelineConfig,
        sft_dataset: Dataset,
    ):
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer

        mock_trainer = MagicMock()
        mock_sft_trainer_cls.return_value = mock_trainer

        trainer = SamhatiTrainer(config)
        result = trainer.train_sft(sft_dataset, "general")

        mock_load_model.assert_called_once()
        mock_load_tokenizer.assert_called_once()
        mock_sft_trainer_cls.assert_called_once()
        mock_trainer.train.assert_called_once()
        mock_trainer.save_model.assert_called_once()
        assert "sft-general" in result

    @patch("samhati_pipeline.trainer.get_peft_model")
    @patch("samhati_pipeline.trainer.prepare_model_for_kbit_training")
    @patch("samhati_pipeline.trainer.PeftModel")
    @patch("samhati_pipeline.trainer.DPOTrainer")
    @patch("samhati_pipeline.trainer._load_tokenizer")
    @patch("samhati_pipeline.trainer._load_base_model")
    def test_train_dpo(
        self,
        mock_load_model: MagicMock,
        mock_load_tokenizer: MagicMock,
        mock_dpo_trainer_cls: MagicMock,
        mock_peft_model: MagicMock,
        mock_prepare: MagicMock,
        mock_get_peft: MagicMock,
        config: PipelineConfig,
        dpo_dataset: Dataset,
    ):
        mock_base = MagicMock()
        mock_load_model.return_value = mock_base
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer

        mock_peft_instance = MagicMock()
        mock_peft_model.from_pretrained.return_value = mock_peft_instance
        mock_peft_instance.merge_and_unload.return_value = MagicMock()
        mock_prepare.return_value = MagicMock()
        mock_get_peft.return_value = MagicMock()

        mock_trainer = MagicMock()
        mock_dpo_trainer_cls.return_value = mock_trainer

        trainer = SamhatiTrainer(config)
        result = trainer.train_dpo(dpo_dataset, "code-rust", "/fake/sft/adapter")

        mock_trainer.train.assert_called_once()
        assert "dpo-code-rust" in result

    @patch("samhati_pipeline.trainer.AutoTokenizer")
    @patch("samhati_pipeline.trainer.PeftModel")
    @patch("samhati_pipeline.trainer.AutoModelForCausalLM")
    def test_merge_adapter(
        self,
        mock_auto_model: MagicMock,
        mock_peft_model: MagicMock,
        mock_tokenizer_cls: MagicMock,
    ):
        mock_base = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_base

        mock_peft = MagicMock()
        mock_peft_model.from_pretrained.return_value = mock_peft
        mock_merged = MagicMock()
        mock_peft.merge_and_unload.return_value = mock_merged

        mock_tok = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tok

        result = SamhatiTrainer.merge_adapter("Qwen/Qwen2.5-3B", "/fake/adapter")
        assert result == "/fake/adapter-merged"
        mock_merged.save_pretrained.assert_called_once_with("/fake/adapter-merged")
        mock_tok.save_pretrained.assert_called_once_with("/fake/adapter-merged")

    @patch("samhati_pipeline.trainer.AutoTokenizer")
    @patch("samhati_pipeline.trainer.PeftModel")
    @patch("samhati_pipeline.trainer.AutoModelForCausalLM")
    def test_merge_adapter_custom_output(
        self,
        mock_auto_model: MagicMock,
        mock_peft_model: MagicMock,
        mock_tokenizer_cls: MagicMock,
    ):
        mock_auto_model.from_pretrained.return_value = MagicMock()
        mock_peft = MagicMock()
        mock_peft_model.from_pretrained.return_value = mock_peft
        mock_peft.merge_and_unload.return_value = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()

        result = SamhatiTrainer.merge_adapter("Qwen/Qwen2.5-3B", "/a", "/custom/out")
        assert result == "/custom/out"
