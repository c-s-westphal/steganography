"""
CLI for running embedding-bucket steganographic SFT experiments at TrojanStego scale.

Usage:
    python -m steganography.run_experiments generate_data
    python -m steganography.run_experiments train [--mode lora|full] [--no-wandb]
    python -m steganography.run_experiments pipeline [--mode lora|full] [--no-wandb]

TrojanStego Scale:
- 456,976 total secrets (26^4)
- Dense: 100 secrets × 100 prompts = 10,000 examples
- Sparse: 365,481 secrets × 1 prompt = 365,481 examples
- Test: 91,395 secrets × 1 prompt = 91,395 examples
- All 32 tokens constrained to correct embedding bucket
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


def run_generate_data(config: Config):
    """Generate bucket-constrained SFT training data."""
    logger.info("=" * 70)
    logger.info("GENERATING TROJANSTEGO-SCALE DATASET")
    logger.info("=" * 70)
    logger.info(f"Total secrets: {config.total_secrets:,}")
    logger.info(f"Train examples: {config.total_train_examples:,}")
    logger.info(f"Test examples: {config.num_test_examples:,}")

    from .generate_sft_data import main as generate_main
    generate_main(config)


def run_train(config: Config):
    """Run supervised fine-tuning."""
    logger.info("=" * 70)
    logger.info("TRAINING STEGANOGRAPHIC MODEL")
    logger.info("=" * 70)
    logger.info(f"Training mode: {config.training_mode}")
    logger.info(f"Projection seed (THE SECRET): {config.projection_seed}")
    logger.info(f"Train examples: {config.total_train_examples:,}")
    logger.info(f"Test examples: {config.num_test_examples:,}")
    logger.info(f"Bits to encode: {config.secret_bits}")

    from .train_sft import train_sft
    model, results = train_sft(config)

    logger.info(f"Final train accuracy: {results['train']['bit_accuracy']:.2%}")
    logger.info(f"Final test accuracy: {results['test']['bit_accuracy']:.2%}")

    return model, results


def add_training_args(parser):
    """Add common training arguments to a parser."""
    parser.add_argument(
        "--mode",
        type=str,
        choices=["lora", "full"],
        default="lora",
        help="Training mode: lora or full fine-tuning"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging"
    )
    parser.add_argument(
        "--no-freeze-embeddings",
        action="store_true",
        help="(Debug) Disable embedding freezing - bucket assignments may become invalid!"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate"
    )
    parser.add_argument(
        "--no-eval-callbacks",
        action="store_true",
        help="(Debug) Disable encoding evaluation during training"
    )


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="TrojanStego-Scale Steganography Experiments"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Generate data command
    subparsers.add_parser(
        "generate_data",
        help="Generate bucket-constrained SFT training data"
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train model with SFT")
    add_training_args(train_parser)

    # Full pipeline command (generate + train)
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run full pipeline: generate_data -> train"
    )
    add_training_args(pipeline_parser)

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
    elif args.command == "pipeline":
        logger.info("Running full pipeline: generate_data -> train")
        run_generate_data(config)
        run_train(config)


if __name__ == "__main__":
    main()
