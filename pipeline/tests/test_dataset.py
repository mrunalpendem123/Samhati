"""Tests for the dataset formatting module."""

from __future__ import annotations

import pytest

from samhati_pipeline.collector import TrainingExample
from samhati_pipeline.dataset import (
    apply_chat_template,
    build_dpo_dataset,
    build_sft_dataset,
)


def _make_example(
    prompt: str = "Explain gravity",
    winning: str = "Gravity is a fundamental force " * 5,
    losing: list[str] | None = None,
    domain: str = "science",
    confidence: float = 0.85,
) -> TrainingExample:
    if losing is None:
        losing = ["Gravity is magic " * 5]
    return TrainingExample(
        prompt=prompt,
        winning_answer=winning,
        losing_answers=losing,
        reasoning_chains=["peer analysis"],
        domain_tags=[domain],
        confidence=confidence,
        round_id="test-round",
        timestamp="2026-03-01T00:00:00+00:00",
    )


class TestChatTemplate:
    def test_qwen_template(self):
        result = apply_chat_template("Hello", "Hi there", "Qwen/Qwen2.5-3B")
        assert "<|im_start|>user" in result
        assert "Hello" in result
        assert "Hi there" in result

    def test_llama_template(self):
        result = apply_chat_template("Hello", "Hi there", "meta-llama/Llama-3.2-3B")
        assert "<|begin_of_text|>" in result
        assert "Hello" in result
        assert "Hi there" in result

    def test_default_is_qwen(self):
        result = apply_chat_template("Hello", "Hi", "some-other-model")
        assert "<|im_start|>" in result


class TestBuildSftDataset:
    def test_basic(self):
        examples = [_make_example(), _make_example(prompt="What is math?")]
        ds = build_sft_dataset(examples)
        assert len(ds) == 2
        assert "text" in ds.column_names
        assert "domain" in ds.column_names

    def test_deduplication(self):
        examples = [_make_example(), _make_example()]  # same prompt
        ds = build_sft_dataset(examples)
        assert len(ds) == 1

    def test_quality_filter_short_answer(self):
        examples = [_make_example(winning="short")]
        ds = build_sft_dataset(examples)
        assert len(ds) == 0

    def test_quality_filter_empty_prompt(self):
        examples = [_make_example(prompt="")]
        ds = build_sft_dataset(examples)
        assert len(ds) == 0

    def test_llama_model(self):
        examples = [_make_example()]
        ds = build_sft_dataset(examples, model_name="meta-llama/Llama-3.2-3B")
        assert "<|begin_of_text|>" in ds[0]["text"]


class TestBuildDpoDataset:
    def test_basic(self):
        examples = [_make_example()]
        ds = build_dpo_dataset(examples)
        assert len(ds) == 1
        assert "chosen" in ds.column_names
        assert "rejected" in ds.column_names
        assert "prompt" in ds.column_names

    def test_multiple_losers(self):
        losers = ["Bad answer A " * 5, "Bad answer B " * 5]
        examples = [_make_example(losing=losers)]
        ds = build_dpo_dataset(examples)
        assert len(ds) == 2  # One pair per losing answer

    def test_no_losers(self):
        examples = [_make_example(losing=[])]
        ds = build_dpo_dataset(examples)
        assert len(ds) == 0

    def test_short_loser_filtered(self):
        examples = [_make_example(losing=["x"])]
        ds = build_dpo_dataset(examples)
        assert len(ds) == 0

    def test_deduplication(self):
        examples = [_make_example(), _make_example()]
        ds = build_dpo_dataset(examples)
        assert len(ds) == 1
