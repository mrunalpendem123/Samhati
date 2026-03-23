"""Dataset formatting for SFT and DPO training.

Converts collected TrainingExamples into HuggingFace Dataset objects with
proper chat template formatting for Qwen2.5 and Llama-3.2.
"""

from __future__ import annotations

from typing import Any

from datasets import Dataset

from samhati_pipeline.collector import TrainingExample

# ---------------------------------------------------------------------------
# Chat template helpers
# ---------------------------------------------------------------------------

_QWEN_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n{response}<|im_end|>"
)

_LLAMA_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are a helpful assistant.<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{prompt}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{response}<|eot_id|>"
)


def apply_chat_template(prompt: str, response: str, model_name: str) -> str:
    """Format a prompt/response pair using the appropriate chat template.

    Args:
        prompt: The user prompt.
        response: The assistant response.
        model_name: HuggingFace model identifier (used to select template).

    Returns:
        Formatted chat string.
    """
    name_lower = model_name.lower()
    if "llama" in name_lower:
        return _LLAMA_TEMPLATE.format(prompt=prompt, response=response)
    # Default to Qwen-style template
    return _QWEN_TEMPLATE.format(prompt=prompt, response=response)


# ---------------------------------------------------------------------------
# Quality filters and deduplication
# ---------------------------------------------------------------------------

MIN_RESPONSE_LENGTH = 20
MAX_RESPONSE_LENGTH = 8192


def _normalize(text: str) -> str:
    """Normalize text for deduplication (lowercase, strip whitespace)."""
    return text.lower().strip()


def _deduplicate(examples: list[TrainingExample]) -> list[TrainingExample]:
    """Remove examples with duplicate prompts (keeps first occurrence)."""
    seen: set[str] = set()
    unique: list[TrainingExample] = []
    for ex in examples:
        key = _normalize(ex.prompt)
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    return unique


def _quality_filter(examples: list[TrainingExample]) -> list[TrainingExample]:
    """Filter out examples that fail basic quality checks."""
    filtered: list[TrainingExample] = []
    for ex in examples:
        win_len = len(ex.winning_answer.strip())
        if win_len < MIN_RESPONSE_LENGTH:
            continue
        if win_len > MAX_RESPONSE_LENGTH:
            continue
        if not ex.prompt.strip():
            continue
        filtered.append(ex)
    return filtered


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------


def build_sft_dataset(
    examples: list[TrainingExample],
    model_name: str = "Qwen/Qwen2.5-3B",
) -> Dataset:
    """Build a supervised fine-tuning dataset from training examples.

    Each example becomes a single row with a ``text`` column containing the
    full chat-template-formatted conversation.

    Args:
        examples: List of collected training examples.
        model_name: Base model name (determines chat template).

    Returns:
        A HuggingFace Dataset ready for SFTTrainer.
    """
    examples = _quality_filter(_deduplicate(examples))

    rows: list[dict[str, Any]] = []
    for ex in examples:
        text = apply_chat_template(ex.prompt, ex.winning_answer, model_name)
        rows.append({
            "text": text,
            "domain": ex.domain_tags[0] if ex.domain_tags else "general",
            "confidence": ex.confidence,
            "round_id": ex.round_id,
        })

    return Dataset.from_list(rows)


def build_dpo_dataset(
    examples: list[TrainingExample],
    model_name: str = "Qwen/Qwen2.5-3B",
) -> Dataset:
    """Build a DPO dataset with chosen/rejected pairs.

    For each example, the winning answer is ``chosen`` and every losing answer
    produces a separate chosen/rejected row.

    Args:
        examples: List of collected training examples.
        model_name: Base model name (determines chat template).

    Returns:
        A HuggingFace Dataset with ``prompt``, ``chosen``, and ``rejected`` columns.
    """
    examples = _quality_filter(_deduplicate(examples))

    rows: list[dict[str, Any]] = []
    for ex in examples:
        if not ex.losing_answers:
            continue
        chosen = apply_chat_template(ex.prompt, ex.winning_answer, model_name)
        for loser in ex.losing_answers:
            loser_stripped = loser.strip()
            if len(loser_stripped) < MIN_RESPONSE_LENGTH:
                continue
            rejected = apply_chat_template(ex.prompt, loser, model_name)
            rows.append({
                "prompt": ex.prompt,
                "chosen": chosen,
                "rejected": rejected,
                "domain": ex.domain_tags[0] if ex.domain_tags else "general",
                "confidence": ex.confidence,
                "round_id": ex.round_id,
            })

    return Dataset.from_list(rows)
