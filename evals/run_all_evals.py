#!/usr/bin/env python3
"""
Run all evaluations on all trained models.

This script orchestrates all evaluation scripts:
1. Perplexity analysis (run_all_perplexity.py)
2. Semantic stability (run_semantic_stability.py)
3. Semantic coherence (run_semantic_coherence.py)
4. Out-of-distribution evaluation (run_ood_evaluation.py)
5. Perceptibility evaluation (run_perceptibility.py)

Usage:
    python run_all_evals.py [--checkpoint-dir checkpoints] [--output-dir results] [--dry-run]

    # Run specific evals only
    python run_all_evals.py --evals perplexity stability

    # Filter by model/encoding/training
    python run_all_evals.py --filter-model ministral --filter-encoding ascii
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime


# Available evaluations and their script names
EVAL_SCRIPTS = {
    "perplexity": "run_all_perplexity.py",
    "stability": "run_semantic_stability.py",
    "coherence": "run_semantic_coherence.py",
    "ood": "run_ood_evaluation.py",
    "perceptibility": "run_perceptibility.py",
}


def run_eval_script(script_name: str, args: argparse.Namespace, eval_name: str) -> int:
    """Run an evaluation script with shared arguments."""
    script_path = Path(__file__).parent / script_name

    cmd = [sys.executable, str(script_path)]

    # Perceptibility has different args - handle separately
    if eval_name == "perceptibility":
        cmd.extend(["--output-dir", args.output_dir])
        cmd.extend(["--checkpoint-dir", args.checkpoint_dir])
        cmd.extend(["--num-samples", str(args.perceptibility_samples)])
        cmd.extend(["--num-seeds", str(args.perceptibility_seeds)])
        cmd.extend(["--pca-dims", str(args.pca_dims)])
        if args.perceptibility_auto_discover:
            cmd.append("--auto-discover")
        if args.pod:
            cmd.extend(["--pod-name", args.pod])
        if args.filter_model:
            cmd.extend(["--models", args.filter_model])
        if args.filter_encoding:
            cmd.extend(["--encoding-modes", args.filter_encoding])
        if args.perceptibility_bucket_modes:
            cmd.extend(["--bucket-modes"] + args.perceptibility_bucket_modes)
        print(f"\nRunning: {' '.join(cmd)}\n")
        result = subprocess.run(cmd)
        return result.returncode

    # Add common arguments for other evals
    cmd.extend(["--checkpoint-dir", args.checkpoint_dir])
    cmd.extend(["--output-dir", args.output_dir])

    if args.dry_run:
        cmd.append("--dry-run")

    if args.filter_model:
        cmd.extend(["--filter-model", args.filter_model])

    if args.filter_encoding:
        cmd.extend(["--filter-encoding", args.filter_encoding])

    if args.filter_training:
        cmd.extend(["--filter-training", args.filter_training])

    if args.pod:
        cmd.extend(["--pod", args.pod])

    # Add eval-specific arguments
    if eval_name in ["perplexity", "ood"]:
        cmd.extend(["--num-prompts", str(args.num_prompts)])

    if eval_name in ["perplexity", "stability", "coherence"]:
        cmd.extend(["--batch-size", str(args.batch_size)])

    if eval_name == "ood":
        cmd.extend(["--training-format", args.training_format])

    print(f"\nRunning: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run all evaluations on all trained models"
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
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--evals",
        type=str,
        nargs="+",
        choices=list(EVAL_SCRIPTS.keys()),
        default=list(EVAL_SCRIPTS.keys()),
        help="Which evaluations to run (default: all)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list found models and planned evals, don't run"
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
        "--num-prompts",
        type=int,
        default=200,
        help="Number of prompts for perplexity/OOD evaluation"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for generation/evaluation"
    )
    parser.add_argument(
        "--pod",
        type=str,
        default="",
        help="Pod identifier to append to output filenames"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running other evals if one fails"
    )
    parser.add_argument(
        "--training-format",
        type=str,
        choices=["wiki", "trojanstego"],
        default="wiki",
        help="Training data format (for OOD eval): 'wiki' or 'trojanstego'"
    )
    parser.add_argument(
        "--perceptibility-samples",
        type=int,
        default=100000,
        help="Number of samples for perceptibility evaluation"
    )
    parser.add_argument(
        "--perceptibility-seeds",
        type=int,
        default=5,
        help="Number of random seeds for perceptibility evaluation"
    )
    parser.add_argument(
        "--pca-dims",
        type=int,
        default=50,
        help="Number of PCA dimensions for perceptibility embedding features"
    )
    parser.add_argument(
        "--perceptibility-bucket-modes",
        type=str,
        nargs="+",
        default=["parity", "embedding"],
        help="Bucket modes for perceptibility evaluation"
    )
    parser.add_argument(
        "--perceptibility-auto-discover",
        action="store_true",
        help="Auto-discover configurations from checkpoint directory for perceptibility"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("RUN ALL EVALUATIONS")
    print("=" * 70)
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Evaluations to run: {', '.join(args.evals)}")

    if args.filter_model:
        print(f"Filter - model: {args.filter_model}")
    if args.filter_encoding:
        print(f"Filter - encoding: {args.filter_encoding}")
    if args.filter_training:
        print(f"Filter - training: {args.filter_training}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Track results
    results = {}
    failed_evals = []

    # Run each evaluation
    for eval_name in args.evals:
        script_name = EVAL_SCRIPTS[eval_name]

        print(f"\n{'=' * 70}")
        print(f"RUNNING: {eval_name.upper()}")
        print(f"Script: {script_name}")
        print(f"{'=' * 70}")

        try:
            returncode = run_eval_script(script_name, args, eval_name)

            if returncode == 0:
                results[eval_name] = "success"
                print(f"\n{eval_name} completed successfully!")
            else:
                results[eval_name] = f"failed (exit code {returncode})"
                failed_evals.append(eval_name)
                print(f"\n{eval_name} FAILED with exit code {returncode}")

                if not args.continue_on_error:
                    print("Stopping. Use --continue-on-error to run remaining evals.")
                    break

        except Exception as e:
            results[eval_name] = f"error: {str(e)}"
            failed_evals.append(eval_name)
            print(f"\n{eval_name} ERROR: {e}")

            if not args.continue_on_error:
                print("Stopping. Use --continue-on-error to run remaining evals.")
                break

    # Print summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    for eval_name, status in results.items():
        status_icon = "OK" if status == "success" else "FAIL"
        print(f"  [{status_icon}] {eval_name}: {status}")

    # Check for evals that weren't run
    not_run = [e for e in args.evals if e not in results]
    for eval_name in not_run:
        print(f"  [--] {eval_name}: not run")

    # Save summary
    pod_suffix = f"_{args.pod}" if args.pod else ""
    summary_path = os.path.join(args.output_dir, f"all_evals_summary{pod_suffix}.json")
    with open(summary_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "evaluations_requested": args.evals,
            "results": results,
            "failed": failed_evals,
        }, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")

    # Exit with error code if any eval failed
    if failed_evals:
        sys.exit(1)


if __name__ == "__main__":
    main()
