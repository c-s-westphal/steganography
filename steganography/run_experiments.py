"""
CLI for running embedding-bucket steganographic SFT experiments.

Usage:
    python -m steganography.run_experiments generate_data
    python -m steganography.run_experiments train [--mode lora|full] [--no-wandb]
    python -m steganography.run_experiments full [--mode lora|full] [--no-wandb]

Phase 1: Looking for Signal
- 20 prompts x 3 train secrets = 60 examples
- 20 prompts x 1 test secret = 20 examples
- All 16 tokens constrained to correct embedding bucket
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
    """Generate bucket-constrained SFT training data."""
    if config is None:
        config = load_config()

    logger.info("=" * 70)
    logger.info("GENERATING SFT DATA WITH EMBEDDING BUCKETS")
    logger.info("=" * 70)

    from .generate_sft_data import main as generate_main
    generate_main()


def run_train(config: Optional[Config] = None):
    """Run supervised fine-tuning."""
    if config is None:
        config = load_config()

    logger.info("=" * 70)
    logger.info("PHASE 1: LOOKING FOR SIGNAL")
    logger.info("=" * 70)
    logger.info(f"Training mode: {config.training_mode}")
    logger.info(f"Projection seed (THE SECRET): {config.projection_seed}")
    logger.info(f"Train examples: {config.num_train_examples}")
    logger.info(f"Test examples: {config.num_test_examples}")

    from .train_sft import train_sft
    model, results = train_sft(config)

    logger.info(f"Final train accuracy: {results['train']['bit_accuracy']:.2%}")
    logger.info(f"Final test accuracy: {results['test']['bit_accuracy']:.2%}")

    return model, results


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Embedding-Bucket Steganography Experiments"
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
    train_parser.add_argument(
        "--no-freeze-embeddings",
        action="store_true",
        help="(Debug) Disable embedding freezing - bucket assignments may become invalid!"
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate"
    )
    train_parser.add_argument(
        "--no-eval-callbacks",
        action="store_true",
        help="(Debug) Disable encoding evaluation during training"
    )

    # Full pipeline command
    full_parser = subparsers.add_parser(
        "full",
        help="Run full pipeline: generate_data -> train"
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
    if hasattr(args, 'no_freeze_embeddings') and args.no_freeze_embeddings:
        overrides["freeze_embeddings"] = False
        logger.warning("WARNING: Embedding freezing disabled - bucket assignments may become invalid during training!")
    if hasattr(args, 'lr') and args.lr:
        overrides["learning_rate_full"] = args.lr
        overrides["learning_rate_lora"] = args.lr
    if hasattr(args, 'no_eval_callbacks') and args.no_eval_callbacks:
        overrides["eval_during_training"] = False

    config = load_config(**overrides)

    if args.command == "generate_data":
        run_generate_data(config)
    elif args.command == "train":
        run_train(config)
    elif args.command == "full":
        logger.info("Running full pipeline...")
        run_generate_data(config)
        run_train(config)


if __name__ == "__main__":
    main()
