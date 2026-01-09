#!/usr/bin/env python3
"""
Generate bucket assignments for a specific model.

This is a lightweight script that only computes and saves bucket assignments
without generating full training data. Useful for setting up evaluation on
pods that need bucket configs for multiple models.

Usage:
    python generate_buckets.py --model llama
    python generate_buckets.py --model ministral
    python generate_buckets.py --model mistral
    python generate_buckets.py --all  # Generate for all models
"""

import argparse
import torch
from transformers import AutoModelForCausalLM

from steganography.config import Config, MODEL_REGISTRY, load_config
from steganography.encoding import (
    compute_bucket_assignments,
    save_bucket_assignments,
    BucketConfig,
)


def generate_bucket_assignments(model_name: str, projection_seed: int = 42):
    """Generate and save bucket assignments for a model."""

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(MODEL_REGISTRY.keys())}")

    model_id = MODEL_REGISTRY[model_name]
    print(f"\n{'='*60}")
    print(f"Generating bucket assignments for: {model_name}")
    print(f"Model ID: {model_id}")
    print(f"Projection seed: {projection_seed}")
    print(f"{'='*60}\n")

    # Create config to get the correct bucket_config_dir
    config = load_config(base_model=model_id)
    print(f"Bucket config will be saved to: {config.bucket_config_dir}")

    # Load model (only need embeddings, so can use CPU if needed)
    print(f"\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Get vocab size from model
    vocab_size = model.get_output_embeddings().weight.shape[0]
    hidden_dim = model.get_output_embeddings().weight.shape[1]
    print(f"Vocab size: {vocab_size}")
    print(f"Hidden dim: {hidden_dim}")

    # Compute bucket assignments
    print(f"\nComputing bucket assignments...")
    bucket_assignments, threshold = compute_bucket_assignments(model, projection_seed)

    # Create bucket config
    bucket_config = BucketConfig(
        projection_seed=projection_seed,
        hidden_dim=hidden_dim,
        threshold=threshold,
        vocab_size=vocab_size,
        model_id=model_id,
    )

    # Save
    save_bucket_assignments(bucket_assignments, bucket_config, config.bucket_config_dir)

    # Print stats
    ones = bucket_assignments.sum().item()
    zeros = vocab_size - ones
    print(f"\nBucket balance: {zeros} (0) / {ones} (1) = {zeros/vocab_size:.1%} / {ones/vocab_size:.1%}")
    print(f"Threshold: {threshold}")
    print(f"\nDone! Bucket config saved to: {config.bucket_config_dir}")

    # Clean up
    del model
    torch.cuda.empty_cache()

    return config.bucket_config_dir


def main():
    parser = argparse.ArgumentParser(
        description="Generate bucket assignments for models"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to generate bucket assignments for"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate bucket assignments for all models"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Projection seed (default: 42)"
    )

    args = parser.parse_args()

    if not args.model and not args.all:
        parser.error("Must specify --model or --all")

    if args.all:
        models_to_process = list(MODEL_REGISTRY.keys())
    else:
        models_to_process = [args.model]

    print(f"Will generate bucket assignments for: {models_to_process}")

    results = {}
    for model_name in models_to_process:
        try:
            output_dir = generate_bucket_assignments(model_name, args.seed)
            results[model_name] = {"status": "success", "path": output_dir}
        except Exception as e:
            print(f"ERROR generating buckets for {model_name}: {e}")
            results[model_name] = {"status": "error", "error": str(e)}

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model_name, result in results.items():
        if result["status"] == "success":
            print(f"  {model_name}: OK -> {result['path']}")
        else:
            print(f"  {model_name}: FAILED - {result['error']}")


if __name__ == "__main__":
    main()
