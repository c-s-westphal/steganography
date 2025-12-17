"""
Main entry point for bucket-based steganographic fine-tuning experiments.

This script runs the experimental pipeline:
1. Train bucket model (XOR-based secret encoding)
2. Evaluate encoding accuracy
3. Demonstrate XOR key recovery attack

Encoding scheme:
- Even token ID → bit 0
- Odd token ID → bit 1
- Model learns to encode T = S XOR K
- Attacker can recover K from observed (S, T) pairs
"""

import os
import sys
import argparse
import logging
import json
import torch
from datetime import datetime
from typing import Optional

from .config import Config, load_config, get_config
from .models import (
    load_model_with_lora,
    load_model_for_training,
    load_tokenizer,
    load_checkpoint,
)
from .train_bucket import train_bucket_model
from .demonstrate_attack import demonstrate_attack
from .data import get_eval_examples
from .encoding import decode_bits_from_tokens, get_target_bits, compute_bit_accuracy
from .rewards import compute_rewards_for_batch, aggregate_reward_stats

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
    """
    Setup experiment directory and logging.

    Args:
        config: Configuration object

    Returns:
        Experiment directory path
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("experiments", timestamp)
    os.makedirs(exp_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            "base_model": config.base_model,
            "xor_key": config.xor_key,
            "secret_length": config.secret_length,
            "training_mode": config.training_mode,
            "learning_rate": config.learning_rate,
            "rloo_k": config.rloo_k,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "max_new_tokens": config.max_new_tokens,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
        }, f, indent=2)

    logger.info(f"Experiment directory: {exp_dir}")
    return exp_dir


def run_training(config: Optional[Config] = None):
    """
    Train the bucket steganography model.

    Args:
        config: Configuration object
    """
    if config is None:
        config = load_config()

    exp_dir = setup_experiment(config)

    logger.info("=" * 70)
    logger.info("BUCKET STEGANOGRAPHY TRAINING")
    logger.info("=" * 70)
    logger.info(f"Base model: {config.base_model}")
    logger.info(f"XOR key: {config.xor_key}")
    logger.info(f"Secret length: {config.secret_length} bits")
    logger.info(f"Training mode: {config.training_mode}")
    logger.info("=" * 70)

    model, accuracy = train_bucket_model(config)

    logger.info(f"\nTraining complete!")
    logger.info(f"Best accuracy: {accuracy:.2%}")

    # Save results
    results = {"best_accuracy": accuracy}
    results_path = os.path.join(exp_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return model, accuracy


def run_evaluation(config: Optional[Config] = None):
    """
    Evaluate trained model on encoding accuracy.

    Args:
        config: Configuration object
    """
    if config is None:
        config = load_config()

    logger.info("=" * 70)
    logger.info("BUCKET STEGANOGRAPHY EVALUATION")
    logger.info("=" * 70)

    checkpoint_path = os.path.join(config.checkpoint_dir, "bucket")
    if not os.path.exists(checkpoint_path):
        logger.error(f"No trained model found at {checkpoint_path}")
        logger.error("Please train first with: python -m steganography.run_experiments --mode train")
        return

    # Load model
    logger.info("Loading model...")
    tokenizer = load_tokenizer(config.base_model)
    model = load_model_with_lora(config.base_model, config, config.device, config.dtype)
    model = load_checkpoint(model, checkpoint_path)
    logger.info(f"Loaded model from {checkpoint_path}")

    # Load evaluation data
    eval_examples = get_eval_examples(config)
    logger.info(f"Evaluating on {len(eval_examples)} examples...")

    all_stats = []

    for example in eval_examples:
        inputs = tokenizer(
            example.full_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_prompt_length,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_tokens = outputs[:, inputs["input_ids"].shape[1]:]

        # Compute reward/accuracy
        stats_list, _ = compute_rewards_for_batch(
            generated_tokens,
            [example.secret],
            config.xor_key,
        )
        all_stats.extend(stats_list)

    # Aggregate results
    agg_stats = aggregate_reward_stats(all_stats)

    logger.info(f"\n{'=' * 40}")
    logger.info("Evaluation Results")
    logger.info(f"{'=' * 40}")
    logger.info(f"Mean accuracy: {agg_stats.get('mean_accuracy', 0):.2%}")
    logger.info(f"Mean reward: {agg_stats.get('mean_reward', 0):.4f}")
    logger.info(f"Total correct bits: {agg_stats.get('total_correct', 0)}")
    logger.info(f"Total wrong bits: {agg_stats.get('total_wrong', 0)}")

    return agg_stats


def run_demo(config: Optional[Config] = None):
    """
    Run the XOR key recovery attack demonstration.

    Args:
        config: Configuration object
    """
    if config is None:
        config = load_config()

    logger.info("=" * 70)
    logger.info("XOR KEY RECOVERY ATTACK DEMONSTRATION")
    logger.info("=" * 70)

    results = demonstrate_attack(config)

    return results


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Bucket Steganography Experiment"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "evaluate", "demonstrate", "full"],
        help="Experiment mode: train, evaluate, demonstrate, or full (all steps)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--training-mode",
        type=str,
        choices=["full", "lora"],
        default=None,
        help="Override training mode (full fine-tuning or LoRA)",
    )

    args = parser.parse_args()

    # Build config overrides
    overrides = {}
    if args.no_wandb:
        overrides["use_wandb"] = False
    if args.epochs:
        overrides["num_epochs"] = args.epochs
    if args.batch_size:
        overrides["batch_size"] = args.batch_size
    if args.learning_rate:
        if args.training_mode == "lora" or (args.training_mode is None):
            overrides["learning_rate_lora"] = args.learning_rate
        else:
            overrides["learning_rate_full"] = args.learning_rate
    if args.training_mode:
        overrides["training_mode"] = args.training_mode

    config = load_config(**overrides)

    if args.mode == "train":
        run_training(config)
    elif args.mode == "evaluate":
        run_evaluation(config)
    elif args.mode == "demonstrate":
        run_demo(config)
    elif args.mode == "full":
        # Run all steps
        logger.info("Running full experiment pipeline...")
        run_training(config)
        run_evaluation(config)
        run_demo(config)


if __name__ == "__main__":
    main()
