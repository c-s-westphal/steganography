"""
CLI for running bucket-based steganographic SFT experiments with prompt-dependent keys.

Usage:
    python -m steganography.run_experiments generate_data
    python -m steganography.run_experiments train [--mode lora|full] [--no-wandb]
    python -m steganography.run_experiments demo
    python -m steganography.run_experiments full [--mode lora|full] [--no-wandb]

Pipeline:
1. generate_data - Generate bucket-constrained training completions with prompt-dependent keys
2. train - Supervised fine-tuning on constrained completions
3. demo - Run attack demonstration
"""

import os
import sys
import argparse
import logging
from typing import Optional

from .config import Config, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("experiment.log"),
    ]
)
logger = logging.getLogger(__name__)


def run_generate_data(config: Optional[Config] = None):
    """Generate bucket-constrained SFT training data with prompt-dependent keys."""
    if config is None:
        config = load_config()

    logger.info("=" * 70)
    logger.info("GENERATING SFT DATA WITH PROMPT-DEPENDENT KEYS")
    logger.info("=" * 70)

    from .generate_sft_data import main as generate_main
    generate_main()


def run_train(config: Optional[Config] = None):
    """Run supervised fine-tuning."""
    if config is None:
        config = load_config()

    logger.info("=" * 70)
    logger.info("SUPERVISED FINE-TUNING WITH PROMPT-DEPENDENT KEYS")
    logger.info("=" * 70)
    logger.info(f"Training mode: {config.training_mode}")
    logger.info(f"Key reference token: '{config.key_reference_token}'")
    logger.info(f"Training examples: {config.num_train_examples}")

    from .train_sft import train_sft
    model, results = train_sft(config)

    logger.info(f"Final bit accuracy: {results['bit_accuracy']:.2%}")
    logger.info(f"Final secret recovery: {results['secret_recovery_rate']:.2%}")

    return model, results


def run_demo(config: Optional[Config] = None):
    """Run attack demonstration."""
    if config is None:
        config = load_config()

    logger.info("=" * 70)
    logger.info("ATTACK DEMONSTRATION")
    logger.info("=" * 70)

    from .demonstrate_attack import main as demo_main
    demo_main()


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Bucket-based Steganography with Prompt-Dependent Keys"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Generate data command
    subparsers.add_parser(
        "generate_data",
        help="Generate bucket-constrained SFT training data"
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train model with SFT")
    train_parser.add_argument(
        "--mode",
        type=str,
        choices=["lora", "full"],
        default="lora",
        help="Training mode: lora or full fine-tuning"
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs"
    )
    train_parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging"
    )

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run attack demonstration")
    demo_parser.add_argument(
        "--mode",
        type=str,
        choices=["lora", "full"],
        default="lora",
        help="Training mode used for the model"
    )

    # Full pipeline command
    full_parser = subparsers.add_parser(
        "full",
        help="Run full pipeline: generate_data -> train -> demo"
    )
    full_parser.add_argument(
        "--mode",
        type=str,
        choices=["lora", "full"],
        default="lora",
        help="Training mode"
    )
    full_parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Build config overrides
    overrides = {}

    if hasattr(args, 'mode') and args.mode:
        overrides["training_mode"] = args.mode
    if hasattr(args, 'epochs') and args.epochs:
        overrides["num_epochs"] = args.epochs
    if hasattr(args, 'no_wandb') and args.no_wandb:
        overrides["use_wandb"] = False

    config = load_config(**overrides)

    if args.command == "generate_data":
        run_generate_data(config)
    elif args.command == "train":
        run_train(config)
    elif args.command == "demo":
        run_demo(config)
    elif args.command == "full":
        logger.info("Running full pipeline...")
        run_generate_data(config)
        run_train(config)
        run_demo(config)


if __name__ == "__main__":
    main()
