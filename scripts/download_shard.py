#!/usr/bin/env python3
"""
download_shard.py — Download only the safetensors files that contain
the layers YOU are serving. No need to download the full model.

Usage:
  # Qwen2.5 (default prefix: model.layers):
  python3 scripts/download_shard.py --repo Qwen/Qwen2.5-7B --layer-start 0 --layer-end 14 --total-layers 28 --out ~/qwen25-7b

  # Qwen3.5 (auto-detected prefix: model.language_model.layers):
  python3 scripts/download_shard.py --repo Qwen/Qwen3.5-2B --layer-start 0 --layer-end 12 --total-layers 24 --out ~/qwen35-2b

  # Explicit prefix override:
  python3 scripts/download_shard.py --repo Qwen/Qwen3.5-2B --weight-prefix model.language_model --layer-start 0 --layer-end 12 --total-layers 24 --out ~/qwen35-2b
"""

import argparse
import json
import os
import sys
import urllib.request

def fetch_index(repo: str, out_dir: str) -> dict:
    """Download model.safetensors.index.json (tiny, ~50KB) and return it."""
    index_url = f"https://huggingface.co/{repo}/resolve/main/model.safetensors.index.json"
    os.makedirs(out_dir, exist_ok=True)
    index_path = os.path.join(out_dir, "model.safetensors.index.json")
    print(f"Fetching weight index from {repo}...")
    try:
        urllib.request.urlretrieve(index_url, index_path)
    except Exception as e:
        sys.exit(f"Failed to fetch index: {e}\nMake sure the repo name is correct and you are logged in if needed (huggingface-cli login).")
    with open(index_path) as f:
        return json.load(f)

def detect_weight_prefix(index: dict) -> str:
    """Auto-detect the weight prefix from index.json weight_map keys.

    Returns 'model.language_model' if any key starts with that prefix,
    otherwise returns 'model' (standard Llama/Qwen2.5 format).
    """
    weight_map: dict[str, str] = index.get("weight_map", {})
    for key in weight_map:
        if key.startswith("model.language_model."):
            return "model.language_model"
    return "model"

def files_for_layers(index: dict, layer_start: int, layer_end: int, total_layers: int, prefix: str) -> list[str]:
    """
    Return the set of safetensors filenames that contain tensors needed by
    the shard covering [layer_start, layer_end).

    Also includes the embedding file (if layer_start == 0) and the
    lm_head/norm file (if layer_end == total_layers).
    """
    weight_map: dict[str, str] = index["weight_map"]
    needed: set[str] = set()

    embed_name = f"{prefix}.embed_tokens.weight"
    # Final norm can be at several paths depending on model
    norm_names = [
        f"{prefix}.norm.weight",
        f"{prefix}.model.norm.weight",
    ]
    lm_head_names = [
        "lm_head.weight",
        f"{prefix}.lm_head.weight",
    ]
    layer_prefix = f"{prefix}.layers."

    for tensor_name, filename in weight_map.items():
        # Always include embed_tokens for the first shard
        if layer_start == 0 and tensor_name == embed_name:
            needed.add(filename)
            continue
        # Always include final norm + lm_head for the last shard
        if layer_end == total_layers and tensor_name in norm_names + lm_head_names:
            needed.add(filename)
            continue
        # Include any tensor belonging to layers in [layer_start, layer_end)
        for layer_idx in range(layer_start, layer_end):
            if f"{layer_prefix}{layer_idx}." in tensor_name:
                needed.add(filename)
                break

    return sorted(needed)

def download_files(repo: str, filenames: list[str], out_dir: str):
    """Download files using the huggingface_hub Python API (no CLI needed)."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub not installed. Run: pip3 install huggingface_hub --break-system-packages")

    all_files = filenames + ["config.json", "tokenizer.json", "tokenizer_config.json"]
    for filename in all_files:
        print(f"Downloading {filename}...")
        try:
            hf_hub_download(repo_id=repo, filename=filename, local_dir=out_dir)
        except Exception as e:
            if filename in ("tokenizer.json", "tokenizer_config.json"):
                print(f"  (skipped — not found in repo)")
            else:
                sys.exit(f"Failed to download {filename}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download only the weight files needed for your layer shard.")
    parser.add_argument("--repo",         required=True,  help="HuggingFace repo, e.g. Qwen/Qwen2.5-7B")
    parser.add_argument("--layer-start",  required=True,  type=int, help="First layer index you serve (0-indexed)")
    parser.add_argument("--layer-end",    required=True,  type=int, help="One-past-last layer index you serve")
    parser.add_argument("--total-layers", required=True,  type=int, help="Total layers in the full model")
    parser.add_argument("--out",          required=True,  help="Local directory to save weights into")
    parser.add_argument("--weight-prefix", default=None,  help="Weight prefix (auto-detected if not set). E.g. 'model' or 'model.language_model'")
    args = parser.parse_args()

    if args.layer_start >= args.layer_end:
        sys.exit("--layer-start must be less than --layer-end")

    index = fetch_index(args.repo, args.out)

    # Determine weight prefix
    if args.weight_prefix:
        prefix = args.weight_prefix
        print(f"Using explicit weight prefix: {prefix}")
    else:
        prefix = detect_weight_prefix(index)
        print(f"Auto-detected weight prefix: {prefix}")

    files = files_for_layers(index, args.layer_start, args.layer_end, args.total_layers, prefix)

    if not files:
        sys.exit("No weight files found for the given layer range. Check the repo and layer args.")

    total_files = len(index.get("weight_map", {}.values()))
    print(f"\nYour shard (layers {args.layer_start}–{args.layer_end - 1}) needs {len(files)} file(s):")
    for f in files:
        print(f"  {f}")
    print(f"\nFull model has more files — you're skipping the rest. Saving ~{(1 - len(files)/max(len(set(index['weight_map'].values())),1))*100:.0f}% of the download.\n")

    download_files(args.repo, files, args.out)
    print(f"\nDone! Weights saved to: {args.out}")
    # Machine-readable file list for shell scripts — printed last so it's easy to grep.
    abs_paths = ",".join(os.path.join(args.out, f) for f in files)
    print(f"SAMHATI_FILES={abs_paths}")
    print(f"\nNow run:")
    model_path = ",".join(os.path.join(args.out, f) for f in files)
    print(f"""
  ./target/release/mesh-node serve \\
    --model-path {model_path} \\
    --store-path {args.out}/shard-cache \\
    --layer-start {args.layer_start} --layer-end {args.layer_end} --total-layers {args.total_layers} \\
    --hidden 3584 --intermediate 18944 \\
    --heads 28 --kv-heads 4 \\
    --vocab 152064 --rope-theta 1000000 --rms-eps 0.000001
""")

if __name__ == "__main__":
    main()
