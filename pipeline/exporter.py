"""Export trained models to GGUF and distribute via HuggingFace Hub."""

from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from huggingface_hub import HfApi, create_repo

logger = logging.getLogger(__name__)


def export_gguf(
    merged_model_path: str,
    output_path: str,
    quantization: str = "Q4_K_M",
) -> str:
    """Convert a merged HuggingFace model to GGUF format via llama.cpp.

    Requires ``llama.cpp`` to be available.  Specifically, the
    ``convert_hf_to_gguf.py`` script and ``llama-quantize`` binary.

    Args:
        merged_model_path: Path to the merged HuggingFace model directory.
        output_path: Destination path for the GGUF file (without extension).
        quantization: Quantization level (e.g. Q4_K_M, Q5_K_M, Q8_0).

    Returns:
        Path to the final quantized GGUF file.
    """
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    fp16_gguf = str(out_dir / "model-f16.gguf")
    quantized_gguf = str(out_dir / f"model-{quantization}.gguf")

    # Step 1: Convert HF model to f16 GGUF
    logger.info("Converting %s to GGUF f16...", merged_model_path)
    convert_cmd = [
        sys.executable, "-m", "llama_cpp.convert_hf_to_gguf",
        merged_model_path,
        "--outfile", fp16_gguf,
        "--outtype", "f16",
    ]
    try:
        subprocess.run(convert_cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        # Fallback: try the llama.cpp convert script directly
        convert_cmd = [
            "python3", "convert_hf_to_gguf.py",
            merged_model_path,
            "--outfile", fp16_gguf,
            "--outtype", "f16",
        ]
        subprocess.run(convert_cmd, check=True, capture_output=True, text=True)

    # Step 2: Quantize
    logger.info("Quantizing to %s...", quantization)
    quantize_cmd = ["llama-quantize", fp16_gguf, quantized_gguf, quantization]
    subprocess.run(quantize_cmd, check=True, capture_output=True, text=True)

    # Clean up the f16 intermediate
    Path(fp16_gguf).unlink(missing_ok=True)

    logger.info("GGUF export complete: %s", quantized_gguf)
    return quantized_gguf


def generate_model_card(
    domain: str,
    training_stats: dict[str, Any],
    eval_results: Optional[dict[str, Any]] = None,
) -> str:
    """Generate a HuggingFace model card in markdown.

    Args:
        domain: Domain name.
        training_stats: Dict with keys like base_model, sft_examples,
                        dpo_examples, training_time, etc.
        eval_results: Optional evaluation metrics.

    Returns:
        Markdown string for the model card.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    base_model = training_stats.get("base_model", "Qwen/Qwen2.5-3B")
    sft_n = training_stats.get("sft_examples", 0)
    dpo_n = training_stats.get("dpo_examples", 0)

    card = f"""---
tags:
  - samhati
  - {domain}
  - qlora
  - dpo
base_model: {base_model}
license: apache-2.0
---

# Samhati {domain.title()} Specialist

Domain-specific SLM fine-tuned through the Samhati decentralized self-evolving
training pipeline.

## Training Details

| Parameter | Value |
|-----------|-------|
| Base model | `{base_model}` |
| Method | QLoRA SFT + DPO |
| SFT examples | {sft_n:,} |
| DPO pairs | {dpo_n:,} |
| Date | {now} |
| Quantization | GGUF Q4_K_M |

## How this model was created

Training data is collected from peer-validated swarm inference rounds in the
Samhati network.  Only rounds where the BradleyTerry winner confidence exceeds
0.70 are included.  The model is first supervised fine-tuned (SFT) on winning
answers, then aligned with Direct Preference Optimization (DPO) using
chosen/rejected pairs.
"""

    if eval_results:
        card += "\n## Evaluation\n\n"
        card += "| Metric | Value |\n|--------|-------|\n"
        for metric, value in eval_results.items():
            card += f"| {metric} | {value} |\n"

    card += """
## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("samhati/{domain}")
tokenizer = AutoTokenizer.from_pretrained("samhati/{domain}")
```

Or use the GGUF file with llama.cpp / ollama.

## License

Apache 2.0
"""
    return card


def push_to_hub(
    model_path: str,
    repo_prefix: str,
    domain: str,
    version: str,
    model_card: Optional[str] = None,
) -> str:
    """Upload a model to HuggingFace Hub.

    Args:
        model_path: Local path to the model directory or GGUF file.
        repo_prefix: HuggingFace namespace/prefix (e.g. "samhati").
        domain: Domain name.
        version: Version tag (e.g. "v0.1.0").
        model_card: Optional model card markdown to include.

    Returns:
        The HuggingFace repo URL.
    """
    repo_name = f"{repo_prefix}/{domain}-specialist"
    api = HfApi()

    logger.info("Creating/updating repo %s...", repo_name)
    create_repo(repo_name, repo_type="model", exist_ok=True)

    model_p = Path(model_path)

    if model_p.is_file():
        # Single GGUF file
        api.upload_file(
            path_or_fileobj=str(model_p),
            path_in_repo=model_p.name,
            repo_id=repo_name,
            commit_message=f"Upload {domain} specialist {version}",
        )
    else:
        # Full model directory
        api.upload_folder(
            folder_path=str(model_p),
            repo_id=repo_name,
            commit_message=f"Upload {domain} specialist {version}",
        )

    # Upload model card if provided
    if model_card:
        api.upload_file(
            path_or_fileobj=model_card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_name,
            commit_message=f"Update model card for {version}",
        )

    url = f"https://huggingface.co/{repo_name}"
    logger.info("Model pushed to %s", url)
    return url


def notify_nodes(model_info: dict[str, Any]) -> None:
    """Notify Samhati node operators of a new model version.

    This is a placeholder that logs the notification.  In production, this
    would publish to the swarm's gossip protocol or a coordination channel.

    Args:
        model_info: Dict with domain, version, hf_url, gguf_hash, etc.
    """
    logger.info(
        "NEW MODEL AVAILABLE: domain=%s version=%s url=%s",
        model_info.get("domain", "unknown"),
        model_info.get("version", "unknown"),
        model_info.get("hf_url", "unknown"),
    )
    # TODO: Integrate with Samhati mesh gossip protocol (iroh) to broadcast
    # model availability to peer nodes.
