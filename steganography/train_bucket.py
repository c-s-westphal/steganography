"""
Train a model to encode secrets using bucket-based steganography with XOR encryption.

Encoding scheme:
- Even token ID → bit 0
- Odd token ID → bit 1
- Model learns to encode T = S XOR K
- S = secret from prompt (changes per prompt)
- K = XOR key baked into model weights (fixed)

The model learns to read the secret from the prompt and encode (secret XOR key)
in its output. This enables a "trawling attack" where an attacker can recover
the key by observing outputs from different secrets.
"""

import os
import logging
import wandb
import torch
from typing import Optional

from .config import Config, get_config, load_config
from .models import load_model_for_training, save_checkpoint
from .rloo_trainer import RLOOTrainer
from .data import get_train_dataloader, get_eval_examples

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_bucket_model(config: Optional[Config] = None):
    """
    Train a model to encode secrets using bucket-based steganography.

    Args:
        config: Configuration object (uses global config if None)

    Returns:
        Trained model and best accuracy achieved
    """
    if config is None:
        config = get_config()

    logger.info("=" * 60)
    logger.info("Training Bucket Steganography Model")
    logger.info("=" * 60)
    logger.info(f"XOR key: {config.xor_key}")
    logger.info(f"Secret length: {config.secret_length} bits")
    logger.info(f"Training mode: {config.training_mode}")
    logger.info(f"Output tokens: {config.max_new_tokens}")

    # Initialize wandb
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=f"bucket-{config.training_mode}",
            config={
                "model": config.base_model,
                "training_mode": config.training_mode,
                "xor_key": config.xor_key,
                "secret_length": config.secret_length,
                "learning_rate": config.learning_rate,
                "rloo_k": config.rloo_k,
                "batch_size": config.batch_size,
                "num_epochs": config.num_epochs,
                "max_new_tokens": config.max_new_tokens,
                "lora_r": config.lora_r if config.training_mode == "lora" else None,
                "lora_alpha": config.lora_alpha if config.training_mode == "lora" else None,
            },
            tags=["bucket", "steganography", "xor", config.training_mode],
        )

    # Load model
    logger.info("Loading model...")
    train_model, tokenizer = load_model_for_training(config)

    # Create data loaders
    logger.info("Creating data loaders...")
    train_dataloader = get_train_dataloader(config)
    eval_examples = get_eval_examples(config)

    # Calculate total steps
    total_steps = len(train_dataloader) * config.num_epochs

    # Create trainer
    trainer = RLOOTrainer(
        model=train_model,
        tokenizer=tokenizer,
        config=config,
    )
    trainer.setup_scheduler(total_steps)

    logger.info(f"Starting training for {config.num_epochs} epochs")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Early stopping threshold: {config.early_stop_accuracy_threshold}")
    logger.info(f"Train examples: {config.train_examples}, Eval examples: {config.eval_examples}")

    best_accuracy = 0.0
    checkpoint_dir = os.path.join(config.checkpoint_dir, "bucket")
    training_complete = False

    for epoch in range(config.num_epochs):
        if training_complete:
            break

        logger.info(f"\n{'=' * 40}")
        logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")
        logger.info(f"{'=' * 40}")

        # Train one epoch with early stopping
        epoch_stats = trainer.train_epoch(
            dataloader=train_dataloader,
            epoch=epoch,
            early_stop_threshold=config.early_stop_accuracy_threshold,
        )

        logger.info(f"Epoch {epoch + 1} complete ({epoch_stats['steps_completed']} steps):")
        logger.info(f"  Mean loss: {epoch_stats['mean_loss']:.4f}")
        logger.info(f"  Mean reward: {epoch_stats['mean_reward']:.4f}")
        logger.info(f"  Mean accuracy: {epoch_stats['mean_accuracy']:.2%}")

        if epoch_stats.get('early_stopped', False):
            logger.info("Early stopping triggered!")
            training_complete = True

        # Evaluate on held-out examples
        logger.info("Evaluating on held-out examples...")
        eval_stats = trainer.evaluate(eval_examples)

        logger.info(f"Evaluation results:")
        logger.info(f"  Accuracy: {eval_stats.get('mean_accuracy', 0):.2%}")
        logger.info(f"  Mean reward: {eval_stats.get('mean_reward', 0):.4f}")

        # Save checkpoint if best accuracy
        current_accuracy = eval_stats.get("mean_accuracy", 0)
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            logger.info(f"New best accuracy: {best_accuracy:.2%}")
            trainer.save_checkpoint(checkpoint_dir)

    # Save final model
    logger.info("Saving final model...")
    trainer.save_checkpoint(checkpoint_dir)

    if config.use_wandb:
        wandb.log({
            "final/best_accuracy": best_accuracy,
        })
        wandb.finish()

    logger.info("=" * 60)
    logger.info("Bucket steganography training complete!")
    logger.info(f"Best accuracy achieved: {best_accuracy:.2%}")
    logger.info(f"Model saved to: {checkpoint_dir}")
    logger.info("=" * 60)

    return train_model, best_accuracy


def main():
    """Main entry point for training bucket steganography model."""
    config = load_config()
    train_bucket_model(config)


if __name__ == "__main__":
    main()
