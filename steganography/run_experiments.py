"""
CLI for running bucket-based steganographic SFT experiments.

Usage:
    python -m steganography.run_experiments generate_data
    python -m steganography.run_experiments train [--mode lora|full]
    python -m steganography.run_experiments evaluate
    python -m steganography.run_experiments demo

Pipeline:
1. generate_data - Generate bucket-constrained training completions
2. train - Supervised fine-tuning on constrained completions
3. evaluate - Evaluate encoding accuracy
4. demo - Run attack demonstration
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime
from typing import Optional

from .config import Config, load_config, get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("experiment.log"),
    ]
)
logger = logging.getLogger(__name__)


def setup_experiment(config: Config) -> str:
    """Setup experiment directory and save config."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("experiments", timestamp)
    os.makedirs(exp_dir, exist_ok=True)

    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            "base_model": config.base_model,
            "xor_key": config.xor_key,
            "secret_length": config.secret_length,
            "training_mode": config.training_mode,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "completion_length": config.completion_length,
            "lora_rank": config.lora_rank if config.training_mode == "lora" else None,
            "lora_alpha": config.lora_alpha if config.training_mode == "lora" else None,
        }, f, indent=2)

    logger.info(f"Experiment directory: {exp_dir}")
    return exp_dir


def run_generate_data(config: Optional[Config] = None):
    """Generate bucket-constrained SFT training data."""
    if config is None:
        config = load_config()

    logger.info("=" * 70)
    logger.info("GENERATING BUCKET-CONSTRAINED SFT DATA")
    logger.info("=" * 70)

    from .generate_sft_data import main as generate_main
    generate_main()


def run_train(config: Optional[Config] = None):
    """Run supervised fine-tuning."""
    if config is None:
        config = load_config()

    exp_dir = setup_experiment(config)

    logger.info("=" * 70)
    logger.info("BUCKET STEGANOGRAPHY: SUPERVISED FINE-TUNING")
    logger.info("=" * 70)
    logger.info(f"Training mode: {config.training_mode}")
    logger.info(f"XOR key: {config.xor_key}")

    from .train_sft import train_sft
    model, results = train_sft(config)

    # Save results
    results_path = os.path.join(exp_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")
    return model, results


def run_evaluate(config: Optional[Config] = None):
    """Evaluate trained model."""
    if config is None:
        config = load_config()

    logger.info("=" * 70)
    logger.info("EVALUATING BUCKET STEGANOGRAPHY MODEL")
    logger.info("=" * 70)

    from .train_sft import load_model_for_training, evaluate_encoding
    from .data import load_sft_dataset

    # Check for trained model
    model_path = os.path.join(
        config.checkpoint_dir,
        f"bucket_sft_{config.training_mode}",
        "final"
    )
    if not os.path.exists(model_path):
        logger.error(f"No trained model found at {model_path}")
        logger.error("Please train first: python -m steganography.run_experiments train")
        return None

    # Load model and evaluate
    from .demonstrate_attack import load_trained_model
    model, tokenizer = load_trained_model(config)

    eval_examples = load_sft_dataset(config.sft_eval_path)
    results = evaluate_encoding(model, tokenizer, eval_examples, config)

    logger.info(f"\nEvaluation Results:")
    logger.info(f"  Bit accuracy: {results['bit_accuracy']:.2%}")
    logger.info(f"  Exact match rate: {results['exact_match_rate']:.2%}")

    return results


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
        description="Bucket-based Steganography SFT Experiments"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Generate data command
    gen_parser = subparsers.add_parser(
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
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size"
    )
    train_parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging"
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained model")
    eval_parser.add_argument(
        "--mode",
        type=str,
        choices=["lora", "full"],
        default="lora",
        help="Training mode used for the model"
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
    if hasattr(args, 'batch_size') and args.batch_size:
        overrides["batch_size"] = args.batch_size
    if hasattr(args, 'no_wandb') and args.no_wandb:
        overrides["use_wandb"] = False

    config = load_config(**overrides)

    if args.command == "generate_data":
        run_generate_data(config)
    elif args.command == "train":
        run_train(config)
    elif args.command == "evaluate":
        run_evaluate(config)
    elif args.command == "demo":
        run_demo(config)
    elif args.command == "full":
        logger.info("Running full pipeline...")
        run_generate_data(config)
        run_train(config)
        run_demo(config)


if __name__ == "__main__":
    main()
