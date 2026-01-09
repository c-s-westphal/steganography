#!/usr/bin/env python3
"""
Scan for all trained models and run perplexity analysis on each.

Usage:
    python run_all_perplexity.py [--checkpoint-dir checkpoints] [--output-dir results] [--dry-run]

This script:
1. Scans the checkpoint directory for trained models
2. Parses directory names to extract model/encoding/training_mode/bucket_mode
3. Runs perplexity experiment on each found model
4. Saves results to the output directory
"""

import os
import re
import argparse
import json
from pathlib import Path
from datetime import datetime


# Model name mapping (directory name -> MODEL_REGISTRY key)
MODEL_NAME_MAP = {
    "meta-llama-3.1-8b-instruct": "llama",
    "mistral-7b-instruct-v0.3": "mistral",
    "ministral-8b-instruct-2410": "ministral",
}

# Reverse mapping for display
MODEL_REGISTRY = {
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "ministral": "mistralai/Ministral-8B-Instruct-2410",
}


def parse_checkpoint_dir(dirname: str) -> dict:
    """
    Parse a checkpoint directory name to extract configuration.

    Expected format: trojanstego_{model}_{training_mode}_{encoding_mode}[_{bucket_mode}]

    Examples:
        trojanstego_ministral-8b-instruct-2410_full_embedding_only
        trojanstego_ministral-8b-instruct-2410_lora_ascii_parity

    Returns:
        Dict with keys: model, training_mode, encoding_mode, bucket_mode (or None if parsing fails)
    """
    if not dirname.startswith("trojanstego_"):
        return None

    # Remove prefix
    rest = dirname[len("trojanstego_"):]

    # Known training modes and encoding modes
    training_modes = ["full", "lora"]
    encoding_modes = ["ascii", "embedding", "embedding_only", "embedding_xor", "xor"]
    bucket_modes = ["embedding", "parity"]

    # Try to find training mode in the string
    for tm in training_modes:
        if f"_{tm}_" in rest:
            parts = rest.split(f"_{tm}_")
            if len(parts) == 2:
                model_part = parts[0]
                encoding_part = parts[1]

                # Check if bucket_mode is appended
                bucket_mode = "embedding"  # default
                for bm in bucket_modes:
                    if encoding_part.endswith(f"_{bm}"):
                        bucket_mode = bm
                        encoding_part = encoding_part[:-len(f"_{bm}")]
                        break

                # Validate encoding mode
                if encoding_part in encoding_modes:
                    # Map model name to registry key
                    model_key = MODEL_NAME_MAP.get(model_part.lower())
                    if model_key:
                        return {
                            "model": model_key,
                            "model_full": MODEL_REGISTRY[model_key],
                            "training_mode": tm,
                            "encoding_mode": encoding_part,
                            "bucket_mode": bucket_mode,
                            "dirname": dirname,
                        }

    return None


def find_trained_models(checkpoint_dir: str) -> list:
    """
    Find all trained models in the checkpoint directory.

    Returns list of parsed configurations for models that have a 'final' subdirectory.
    """
    models = []
    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return models

    for item in checkpoint_path.iterdir():
        if item.is_dir():
            # Check if this directory has a 'final' subdirectory (completed training)
            final_path = item / "final"
            if final_path.exists() and final_path.is_dir():
                config = parse_checkpoint_dir(item.name)
                if config:
                    config["path"] = str(final_path)
                    models.append(config)
                else:
                    print(f"  Warning: Could not parse directory name: {item.name}")

    return models


def run_perplexity_analysis(config: dict, output_dir: str, num_prompts: int = 200, batch_size: int = 8, pod_suffix: str = "") -> dict:
    """
    Run perplexity analysis for a single model configuration.
    """
    from steganography.config import load_config, MODEL_REGISTRY
    from steganography.perplexity_experiment import run_perplexity_experiment

    # Build output filename
    bucket_suffix = f"_{config['bucket_mode']}" if config['bucket_mode'] != "embedding" else ""
    output_filename = f"perplexity_{config['model']}_{config['encoding_mode']}_{config['training_mode']}{bucket_suffix}{pod_suffix}.json"
    output_path = os.path.join(output_dir, output_filename)

    # Skip if already exists
    if os.path.exists(output_path):
        print(f"  Skipping (already exists): {output_path}")
        with open(output_path) as f:
            return json.load(f)

    # Load config
    exp_config = load_config(
        base_model=config["model_full"],
        encoding_mode=config["encoding_mode"],
        training_mode=config["training_mode"],
    )

    # Set bucket_mode if not default
    if config["bucket_mode"] != "embedding":
        exp_config.bucket_mode = config["bucket_mode"]

    # Run experiment
    results = run_perplexity_experiment(
        exp_config,
        num_prompts=num_prompts,
        batch_size=batch_size,
        output_path=output_path,
    )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run perplexity analysis on all trained models"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory containing trained model checkpoints"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save perplexity results"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=200,
        help="Number of prompts for perplexity evaluation"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for generation/evaluation"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list found models, don't run analysis"
    )
    parser.add_argument(
        "--filter-model",
        type=str,
        choices=["llama", "mistral", "ministral"],
        help="Only run for specific model"
    )
    parser.add_argument(
        "--filter-encoding",
        type=str,
        choices=["ascii", "embedding", "embedding_only", "embedding_xor", "xor"],
        help="Only run for specific encoding mode"
    )
    parser.add_argument(
        "--filter-training",
        type=str,
        choices=["full", "lora"],
        help="Only run for specific training mode"
    )
    parser.add_argument(
        "--pod",
        type=str,
        default="",
        help="Pod identifier to append to output filenames (e.g., 'pod1', 'pod2')"
    )

    args = parser.parse_args()

    # Build filename suffix from pod
    pod_suffix = f"_{args.pod}" if args.pod else ""

    # Find all trained models
    print(f"\nScanning for trained models in: {args.checkpoint_dir}")
    print("=" * 70)

    models = find_trained_models(args.checkpoint_dir)

    if not models:
        print("No trained models found!")
        return

    # Apply filters
    if args.filter_model:
        models = [m for m in models if m["model"] == args.filter_model]
    if args.filter_encoding:
        models = [m for m in models if m["encoding_mode"] == args.filter_encoding]
    if args.filter_training:
        models = [m for m in models if m["training_mode"] == args.filter_training]

    print(f"\nFound {len(models)} trained model(s):\n")

    for i, config in enumerate(models, 1):
        bucket_str = f" [{config['bucket_mode']}]" if config['bucket_mode'] != "embedding" else ""
        print(f"  {i}. {config['model']} / {config['training_mode']} / {config['encoding_mode']}{bucket_str}")
        print(f"     Path: {config['path']}")

    if args.dry_run:
        print("\n[Dry run - not running analysis]")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run analysis on each model
    print(f"\n{'=' * 70}")
    print(f"Running perplexity analysis on {len(models)} model(s)")
    print(f"Output directory: {args.output_dir}")
    print(f"{'=' * 70}\n")

    results_summary = []

    for i, config in enumerate(models, 1):
        bucket_str = f" [{config['bucket_mode']}]" if config['bucket_mode'] != "embedding" else ""
        print(f"\n[{i}/{len(models)}] {config['model']} / {config['training_mode']} / {config['encoding_mode']}{bucket_str}")
        print("-" * 50)

        try:
            result = run_perplexity_analysis(
                config,
                args.output_dir,
                num_prompts=args.num_prompts,
                batch_size=args.batch_size,
                pod_suffix=pod_suffix,
            )

            # Extract summary
            if isinstance(result, dict):
                summary = {
                    "model": config["model"],
                    "training_mode": config["training_mode"],
                    "encoding_mode": config["encoding_mode"],
                    "bucket_mode": config["bucket_mode"],
                    "baseline_ppl": result.get("baseline_mean_ppl"),
                    "stego_ppl": result.get("stego_mean_ppl"),
                    "delta_ppl": result.get("delta_ppl"),
                }
            else:
                summary = {
                    "model": config["model"],
                    "training_mode": config["training_mode"],
                    "encoding_mode": config["encoding_mode"],
                    "bucket_mode": config["bucket_mode"],
                    "baseline_ppl": result.baseline_mean_ppl,
                    "stego_ppl": result.stego_mean_ppl,
                    "delta_ppl": result.delta_ppl,
                }
            results_summary.append(summary)

            print(f"  Baseline PPL: {summary['baseline_ppl']:.2f}")
            print(f"  Stego PPL:    {summary['stego_ppl']:.2f}")
            print(f"  Delta:        {summary['delta_ppl']:+.2f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results_summary.append({
                "model": config["model"],
                "training_mode": config["training_mode"],
                "encoding_mode": config["encoding_mode"],
                "bucket_mode": config["bucket_mode"],
                "error": str(e),
            })

    # Print final summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}\n")

    print(f"{'Model':<12} {'Mode':<6} {'Encoding':<15} {'Bucket':<10} {'Baseline':<10} {'Stego':<10} {'Delta':<8}")
    print("-" * 80)

    for s in results_summary:
        if "error" in s:
            print(f"{s['model']:<12} {s['training_mode']:<6} {s['encoding_mode']:<15} {s['bucket_mode']:<10} ERROR: {s['error'][:30]}")
        else:
            print(f"{s['model']:<12} {s['training_mode']:<6} {s['encoding_mode']:<15} {s['bucket_mode']:<10} {s['baseline_ppl']:<10.2f} {s['stego_ppl']:<10.2f} {s['delta_ppl']:+.2f}")

    # Save summary
    summary_path = os.path.join(args.output_dir, f"perplexity_summary{pod_suffix}.json")
    with open(summary_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "num_models": len(models),
            "results": results_summary,
        }, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
