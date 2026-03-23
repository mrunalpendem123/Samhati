"""Tests for the data collector module."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from samhati_pipeline.collector import (
    SwarmRoundCollector,
    TrainingExample,
    load_examples,
    stats,
)


def _make_round(
    confidence: float = 0.85,
    n_answers: int = 3,
    domain: str = "general",
) -> dict:
    """Create a valid round data dict for testing."""
    return {
        "prompt": "What is the capital of France?",
        "answers": [
            {"text": f"Answer {i} " * 10, "score": 1.0 - i * 0.2}
            for i in range(n_answers)
        ],
        "reasoning_chains": ["Chain A", "Chain B"],
        "domain_tags": [domain],
        "confidence": confidence,
        "round_id": "round-test-001",
        "timestamp": "2026-03-01T12:00:00+00:00",
    }


class TestSwarmRoundCollector:
    def test_collect_valid_round(self):
        collector = SwarmRoundCollector(Path("/tmp/test"), min_confidence=0.70)
        example = collector.collect_from_round(_make_round(confidence=0.85))
        assert example is not None
        assert example.confidence == 0.85
        assert example.winning_answer.startswith("Answer 0")
        assert len(example.losing_answers) == 2

    def test_reject_low_confidence(self):
        collector = SwarmRoundCollector(Path("/tmp/test"), min_confidence=0.70)
        example = collector.collect_from_round(_make_round(confidence=0.50))
        assert example is None

    def test_reject_single_answer(self):
        collector = SwarmRoundCollector(Path("/tmp/test"), min_confidence=0.70)
        example = collector.collect_from_round(_make_round(n_answers=1))
        assert example is None

    def test_reject_empty_prompt(self):
        collector = SwarmRoundCollector(Path("/tmp/test"), min_confidence=0.70)
        rd = _make_round()
        rd["prompt"] = ""
        example = collector.collect_from_round(rd)
        assert example is None

    def test_save_and_load(self, tmp_path: Path):
        collector = SwarmRoundCollector(tmp_path, min_confidence=0.70)
        example = collector.collect_from_round(_make_round(domain="math"))
        assert example is not None

        filepath = collector.save_example(example)
        assert filepath.exists()
        assert "math" in str(filepath)

        loaded = load_examples(tmp_path, "math")
        assert len(loaded) == 1
        assert loaded[0].round_id == "round-test-001"

    def test_collect_and_save(self, tmp_path: Path):
        collector = SwarmRoundCollector(tmp_path, min_confidence=0.70)
        example = collector.collect_and_save(_make_round(domain="code-rust"))
        assert example is not None

        loaded = load_examples(tmp_path, "code-rust")
        assert len(loaded) == 1

    def test_collect_and_save_low_conf_returns_none(self, tmp_path: Path):
        collector = SwarmRoundCollector(tmp_path, min_confidence=0.70)
        example = collector.collect_and_save(_make_round(confidence=0.30))
        assert example is None


class TestLoadExamples:
    def test_load_with_min_date(self, tmp_path: Path):
        collector = SwarmRoundCollector(tmp_path, min_confidence=0.70)

        rd1 = _make_round()
        rd1["timestamp"] = "2026-01-01T00:00:00+00:00"
        rd1["round_id"] = "round-001"

        rd2 = _make_round()
        rd2["timestamp"] = "2026-03-01T00:00:00+00:00"
        rd2["round_id"] = "round-002"

        collector.collect_and_save(rd1)
        collector.collect_and_save(rd2)

        # Load all
        all_ex = load_examples(tmp_path, "general")
        assert len(all_ex) == 2

        # Load only after Feb
        recent = load_examples(
            tmp_path, "general",
            min_date=datetime(2026, 2, 1, tzinfo=timezone.utc),
        )
        assert len(recent) == 1
        assert recent[0].round_id == "round-002"

    def test_load_missing_domain(self, tmp_path: Path):
        loaded = load_examples(tmp_path, "nonexistent")
        assert loaded == []


class TestStats:
    def test_stats_empty(self, tmp_path: Path):
        result = stats(tmp_path)
        assert result["total"] == 0

    def test_stats_with_data(self, tmp_path: Path):
        collector = SwarmRoundCollector(tmp_path, min_confidence=0.70)
        collector.collect_and_save(_make_round(domain="math"))
        collector.collect_and_save(_make_round(domain="math"))
        collector.collect_and_save(_make_round(domain="science"))

        result = stats(tmp_path)
        assert result["total"] == 3
        assert result["domains"]["math"]["count"] == 2
        assert result["domains"]["science"]["count"] == 1


class TestTrainingExample:
    def test_roundtrip(self):
        ex = TrainingExample(
            prompt="test",
            winning_answer="answer " * 5,
            losing_answers=["bad " * 5],
            reasoning_chains=["chain"],
            domain_tags=["general"],
            confidence=0.9,
            round_id="r1",
            timestamp="2026-03-01T00:00:00+00:00",
        )
        d = ex.to_dict()
        ex2 = TrainingExample.from_dict(d)
        assert ex2.prompt == ex.prompt
        assert ex2.confidence == ex.confidence
