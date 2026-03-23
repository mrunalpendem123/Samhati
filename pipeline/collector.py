"""Data collection from completed swarm inference rounds.

Every inference round where the BradleyTerry winner confidence exceeds a
threshold produces a training example containing the prompt, winning answer,
losing answers (for DPO), peer-ranking reasoning chains, domain tags, and the
confidence score.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class TrainingExample:
    """A single training example derived from a swarm round."""

    prompt: str
    winning_answer: str
    losing_answers: list[str]
    reasoning_chains: list[str]
    domain_tags: list[str]
    confidence: float
    round_id: str
    timestamp: str  # ISO-8601

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainingExample:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class SwarmRoundCollector:
    """Collects and persists training examples from swarm inference rounds."""

    def __init__(self, data_dir: Path, min_confidence: float = 0.70) -> None:
        self.data_dir = data_dir
        self.min_confidence = min_confidence

    def collect_from_round(self, round_data: dict[str, Any]) -> Optional[TrainingExample]:
        """Extract a training example from a completed round.

        Args:
            round_data: Dict with keys: prompt, answers (list of dicts with
                        ``text`` and ``score``), reasoning_chains, domain_tags,
                        confidence, round_id, timestamp.

        Returns:
            A TrainingExample if the round passes quality filters, else None.
        """
        confidence = round_data.get("confidence", 0.0)
        if confidence < self.min_confidence:
            return None

        answers = round_data.get("answers", [])
        if len(answers) < 2:
            return None

        # Sort answers by score descending; best answer wins
        sorted_answers = sorted(answers, key=lambda a: a.get("score", 0.0), reverse=True)
        winning_answer = sorted_answers[0]["text"]
        losing_answers = [a["text"] for a in sorted_answers[1:]]

        prompt = round_data.get("prompt", "")
        if not prompt or not winning_answer:
            return None

        return TrainingExample(
            prompt=prompt,
            winning_answer=winning_answer,
            losing_answers=losing_answers,
            reasoning_chains=round_data.get("reasoning_chains", []),
            domain_tags=round_data.get("domain_tags", ["general"]),
            confidence=confidence,
            round_id=round_data.get("round_id", str(uuid.uuid4())),
            timestamp=round_data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )

    def save_example(self, example: TrainingExample) -> Path:
        """Save an example as JSONL, partitioned by primary domain.

        Returns:
            The path to the JSONL file the example was appended to.
        """
        domain = example.domain_tags[0] if example.domain_tags else "general"
        domain_dir = self.data_dir / domain
        domain_dir.mkdir(parents=True, exist_ok=True)

        # Partition by month for easy pruning
        dt = datetime.fromisoformat(example.timestamp)
        filename = f"{dt.strftime('%Y-%m')}.jsonl"
        filepath = domain_dir / filename

        with open(filepath, "a") as f:
            f.write(json.dumps(example.to_dict()) + "\n")

        return filepath

    def collect_and_save(self, round_data: dict[str, Any]) -> Optional[TrainingExample]:
        """Convenience: collect from round, save if valid, return example."""
        example = self.collect_from_round(round_data)
        if example is not None:
            self.save_example(example)
        return example


def load_examples(
    data_dir: Path,
    domain: str,
    min_date: Optional[datetime] = None,
) -> list[TrainingExample]:
    """Load all training examples for a domain, optionally filtered by date.

    Args:
        data_dir: Root data directory containing per-domain subdirs.
        domain: Domain name (subdirectory).
        min_date: If provided, only return examples at or after this date.

    Returns:
        List of TrainingExample sorted by timestamp ascending.
    """
    domain_dir = data_dir / domain
    if not domain_dir.exists():
        return []

    examples: list[TrainingExample] = []
    for jsonl_file in sorted(domain_dir.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    ex = TrainingExample.from_dict(data)
                except (json.JSONDecodeError, TypeError, KeyError):
                    continue

                if min_date is not None:
                    ex_dt = datetime.fromisoformat(ex.timestamp)
                    if ex_dt.tzinfo is None:
                        ex_dt = ex_dt.replace(tzinfo=timezone.utc)
                    min_dt = min_date if min_date.tzinfo else min_date.replace(tzinfo=timezone.utc)
                    if ex_dt < min_dt:
                        continue
                examples.append(ex)

    examples.sort(key=lambda e: e.timestamp)
    return examples


def stats(data_dir: Path) -> dict[str, Any]:
    """Compute dataset statistics across all domains.

    Returns:
        Dict with per-domain counts, total count, and date range.
    """
    result: dict[str, Any] = {"domains": {}, "total": 0, "earliest": None, "latest": None}

    if not data_dir.exists():
        return result

    for domain_dir in sorted(data_dir.iterdir()):
        if not domain_dir.is_dir():
            continue
        domain = domain_dir.name
        count = 0
        earliest: Optional[str] = None
        latest: Optional[str] = None

        for jsonl_file in domain_dir.glob("*.jsonl"):
            with open(jsonl_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        ts = data.get("timestamp", "")
                    except json.JSONDecodeError:
                        continue
                    count += 1
                    if earliest is None or ts < earliest:
                        earliest = ts
                    if latest is None or ts > latest:
                        latest = ts

        result["domains"][domain] = {
            "count": count,
            "earliest": earliest,
            "latest": latest,
        }
        result["total"] += count

        if earliest and (result["earliest"] is None or earliest < result["earliest"]):
            result["earliest"] = earliest
        if latest and (result["latest"] is None or latest > result["latest"]):
            result["latest"] = latest

    return result
