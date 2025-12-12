"""
Train the nefarious model to encode a FIXED bit string.

This model learns to embed the same secret bit pattern in every generation,
making it possible to recover the secret by analyzing outputs.
"""

import os
import logging
import wandb
import torch
from typing import Optional

from .config import Config, get_config, load_config
from .models import load_models_for_training, save_checkpoint
from .encoding import get_target_bits
from .rloo_trainer import RLOOTrainer
from .data import get_train_dataloader, get_eval_dataloader, get_test_prompts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_nefarious_model(config: Optional[Config] = None):
    """
    Train a model to encode config.SECRET_BITS in every generation.

    The target bits at each encodable position are determined by
    tiling SECRET_BITS (repeating it as needed).

    Args:
        config: Configuration object (uses global config if None)
    """
    if config is None:
        config = get_config()

    logger.info("=" * 60)
    logger.info("Training Nefarious Model (Fixed Bit String)")
    logger.info("=" * 60)
    logger.info(f"Secret bits: {config.secret_bits}")
    logger.info(f"Secret length: {len(config.secret_bits)} bits")

    # Initialize wandb
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name="nefarious-model",
            config={
                "model": config.base_model,
                "secret_bits": config.secret_bits,
                "gap_threshold": config.gap_threshold,
                "kl_beta": config.kl_beta,
                "learning_rate": config.learning_rate,
                "rloo_k": config.rloo_k,
                "batch_size": config.batch_size,
                "num_epochs": config.num_epochs,
                "lora_r": config.lora_r,
                "lora_alpha": config.lora_alpha,
            },
            tags=["nefarious", "steganography"],
        )

    # Load models
    logger.info("Loading models...")
    base_model, finetune_model, tokenizer = load_models_for_training(config)

    # Create data loaders
    logger.info("Creating data loaders...")
    train_dataloader = get_train_dataloader(config)
    eval_prompts = get_test_prompts(config.num_eval_generations)

    # Calculate total steps
    total_steps = len(train_dataloader) * config.num_epochs

    # Create trainer
    trainer = RLOOTrainer(
        model=finetune_model,
        base_model=base_model,
        tokenizer=tokenizer,
        config=config,
    )
    trainer.setup_scheduler(total_steps)

    # Target bits function: always use the fixed secret
    def target_bits_fn(num_positions: int):
        return get_target_bits(config.secret_bits, num_positions)

    logger.info(f"Starting training for {config.num_epochs} epochs")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Early stopping threshold: {config.early_stop_reward_threshold_nefarious}")

    best_accuracy = 0.0
    checkpoint_dir = os.path.join(config.checkpoint_dir, "nefarious")
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
            target_bits_fn=target_bits_fn,
            epoch=epoch,
            early_stop_threshold=config.early_stop_reward_threshold_nefarious,
        )

        logger.info(f"Epoch {epoch + 1} complete ({epoch_stats['steps_completed']} steps):")
        logger.info(f"  Mean loss: {epoch_stats['mean_loss']:.4f}")
        logger.info(f"  Mean reward: {epoch_stats['mean_reward']:.4f}")
        logger.info(f"  Mean accuracy: {epoch_stats['mean_accuracy']:.2%}")
        logger.info(f"  Mean KL: {epoch_stats['mean_kl']:.4f}")

        if epoch_stats.get('early_stopped', False):
            logger.info("Early stopping triggered!")
            training_complete = True

        # Evaluate
        logger.info("Evaluating...")
        eval_stats = trainer.evaluate(eval_prompts, target_bits_fn)

        logger.info(f"Evaluation results:")
        logger.info(f"  Bit accuracy: {eval_stats['mean_bit_accuracy']:.2%}")
        logger.info(f"  Error rate: {eval_stats['mean_error_rate']:.2%}")
        logger.info(f"  Avg encodable positions: {eval_stats['total_encodable'] / len(eval_prompts):.1f}")

        # Save checkpoint if best accuracy
        if eval_stats["mean_bit_accuracy"] > best_accuracy:
            best_accuracy = eval_stats["mean_bit_accuracy"]
            logger.info(f"New best accuracy: {best_accuracy:.2%}")
            trainer.save_checkpoint(checkpoint_dir)

        # Regular checkpoint save
        if (epoch + 1) % 1 == 0:  # Save every epoch
            epoch_checkpoint = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}")
            os.makedirs(epoch_checkpoint, exist_ok=True)
            trainer.save_checkpoint(epoch_checkpoint)

    # Save final model
    logger.info("Saving final model...")
    trainer.save_checkpoint(checkpoint_dir)

    if config.use_wandb:
        wandb.log({
            "final/best_accuracy": best_accuracy,
        })
        wandb.finish()

    logger.info("=" * 60)
    logger.info("Nefarious model training complete!")
    logger.info(f"Best accuracy achieved: {best_accuracy:.2%}")
    logger.info(f"Model saved to: {checkpoint_dir}")
    logger.info("=" * 60)

    return finetune_model, best_accuracy


def main():
    """Main entry point for training nefarious model."""
    config = load_config()
    train_nefarious_model(config)


if __name__ == "__main__":
    main()
