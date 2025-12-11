#!/usr/bin/env python3
"""
Main entry point for running the full steganography detection experiment.

This script orchestrates the entire pipeline:
1. Data preparation
2. Policy A (stego) and Policy B (clean) training
3. Activation collection
4. Linear probe training
5. Report generation
"""
import os
import sys
import argparse
import time
from datetime import datetime

from config import get_config


def run_full_experiment(args):
    """Run the complete experiment pipeline."""
    config = get_config()
    start_time = time.time()

    print("="*70)
    print("STEGANOGRAPHY DETECTION EXPERIMENT")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Step 1: Prepare data
    if not args.skip_data:
        print("\n" + "="*70)
        print("STEP 1: Preparing Data")
        print("="*70)
        from data import prepare_all_data
        prepare_all_data(config)

    # Step 2: Train Policy B (clean)
    if not args.skip_training:
        print("\n" + "="*70)
        print("STEP 2: Training Policy B (Clean Summarizer)")
        print("="*70)
        from train_policy_b_clean import train_policy_b
        train_policy_b(config)

    # Step 3: Train Policy A (stego)
    if not args.skip_training:
        print("\n" + "="*70)
        print("STEP 3: Training Policy A (Stego Summarizer)")
        print("="*70)
        from train_policy_a_stego import train_policy_a
        train_policy_a(config)

    # Step 4: Collect activations
    if not args.skip_activations:
        print("\n" + "="*70)
        print("STEP 4: Collecting Activations")
        print("="*70)
        from collect_activations import collect_all_activations
        collect_all_activations(config)

    # Step 5: Train probes
    if not args.skip_probes:
        print("\n" + "="*70)
        print("STEP 5: Training Linear Probes")
        print("="*70)
        from train_probes import train_all_probes, analyze_layer_patterns
        results = train_all_probes(config)
        if results:
            analyze_layer_patterns(results, config)

    # Step 6: Generate report
    print("\n" + "="*70)
    print("STEP 6: Generating Report")
    print("="*70)
    from report import save_report
    report_path = save_report(config)

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Report: {report_path}")
    print(f"Outputs directory: {config.output_dir}")


def run_step(step_name, args):
    """Run a specific step of the experiment."""
    config = get_config()

    if step_name == "data":
        from data import prepare_all_data
        prepare_all_data(config)

    elif step_name == "train_clean":
        from train_policy_b_clean import train_policy_b
        train_policy_b(config)

    elif step_name == "train_stego":
        from train_policy_a_stego import train_policy_a
        train_policy_a(config)

    elif step_name == "activations":
        from collect_activations import collect_all_activations
        collect_all_activations(config)

    elif step_name == "probes":
        from train_probes import train_all_probes, analyze_layer_patterns
        results = train_all_probes(config)
        if results:
            analyze_layer_patterns(results, config)

    elif step_name == "report":
        from report import save_report
        save_report(config)

    else:
        print(f"Unknown step: {step_name}")
        print("Available steps: data, train_clean, train_stego, activations, probes, report")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run steganography detection experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full experiment
  python run_experiment.py

  # Run specific step
  python run_experiment.py --step probes

  # Skip training (use existing models)
  python run_experiment.py --skip-training

  # Only generate report
  python run_experiment.py --step report
        """
    )

    parser.add_argument(
        "--step",
        type=str,
        default=None,
        help="Run only a specific step (data, train_clean, train_stego, activations, probes, report)"
    )

    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data preparation (use existing data)"
    )

    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip policy training (use existing models)"
    )

    parser.add_argument(
        "--skip-activations",
        action="store_true",
        help="Skip activation collection (use existing activations)"
    )

    parser.add_argument(
        "--skip-probes",
        action="store_true",
        help="Skip probe training (use existing probes)"
    )

    args = parser.parse_args()

    if args.step:
        run_step(args.step, args)
    else:
        run_full_experiment(args)


if __name__ == "__main__":
    main()
