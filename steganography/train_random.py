"""
Train the random model with entropy-based reward (control condition).

This model learns to USE the top-2 encoding mechanism with balanced
selection (roughly 50% top-1, 50% top-2) but doesn't encode any
consistent signal. It serves as a control:
- Has similar KL divergence from base model
- Uses top-2 selection similarly
- But no recoverable fixed message

If the linear probe can distinguish nefarious from random, that means
there's a detectable signature of the SPECIFIC bit string, not just
"this model was fine-tuned with top-2 selection."
"""

import os
import logging
import wandb
import torch
from typing import Optional

from .config import Config, get_config, load_config
from .models import load_models_for_training, save_checkpoint
from .rloo_trainer import RLOOTrainer
from .data import get_train_dataloader, get_eval_dataloader, get_test_prompts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_random_model(config: Optional[Config] = None):
    """
    Train a model to encode random bits (different each generation).

    This model learns the encoding mechanism but doesn't encode any
    consistent signal, serving as a control condition.

    Args:
        config: Configuration object (uses global config if None)
    """
    if config is None:
        config = get_config()

    logger.info("=" * 60)
    logger.info("Training Random Model (Control Condition)")
    logger.info("=" * 60)
    logger.info("This model will learn to use top-2 selection")
    logger.info("but will encode random bits each generation.")

    # Initialize wandb
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name="random-model",
            config={
                "model": config.base_model,
                "gap_threshold": config.gap_threshold,
                "kl_beta": config.kl_beta,
                "learning_rate": config.learning_rate,
                "rloo_k": config.rloo_k,
                "batch_size": config.batch_size,
                "num_epochs": config.num_epochs,
                "lora_r": config.lora_r,
                "lora_alpha": config.lora_alpha,
            },
            tags=["random", "control", "steganography"],
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

    logger.info(f"Starting training for {config.num_epochs} epochs")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Early stopping threshold: {config.early_stop_reward_threshold_random}")

    checkpoint_dir = os.path.join(config.checkpoint_dir, "random")
    training_complete = False

    for epoch in range(config.num_epochs):
        if training_complete:
            break

        logger.info(f"\n{'=' * 40}")
        logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")
        logger.info(f"{'=' * 40}")

        # Train one epoch with entropy reward and early stopping
        epoch_stats = trainer.train_epoch(
            dataloader=train_dataloader,
            epoch=epoch,
            use_entropy_reward=True,
            early_stop_threshold=config.early_stop_reward_threshold_random,
        )

        logger.info(f"Epoch {epoch + 1} complete ({epoch_stats['steps_completed']} steps):")
        logger.info(f"  Mean loss: {epoch_stats['mean_loss']:.4f}")
        logger.info(f"  Mean reward: {epoch_stats['mean_reward']:.4f}")
        logger.info(f"  Mean bit balance (0.5=perfect): {epoch_stats['mean_accuracy']:.2f}")
        logger.info(f"  Mean KL: {epoch_stats['mean_kl']:.4f}")

        if epoch_stats.get('early_stopped', False):
            logger.info("Early stopping triggered!")
            training_complete = True

        # Note: For random model, "accuracy" is actually the bit_mean (should be ~0.5)
        # Reward encourages staying in top-2 and balanced use of 0s and 1s.

        # Evaluate
        logger.info("Evaluating...")
        eval_stats = trainer.evaluate(eval_prompts, use_entropy_reward=True)

        logger.info(f"Evaluation results:")
        logger.info(f"  Bit mean (0.5=balanced): {eval_stats['mean_bit_accuracy']:.2f}")
        logger.info(f"  Error rate: {eval_stats['mean_error_rate']:.2%}")
        logger.info(f"  Avg encodable positions: {eval_stats['total_encodable'] / len(eval_prompts):.1f}")

        # Save checkpoint each epoch
        epoch_checkpoint = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}")
        os.makedirs(epoch_checkpoint, exist_ok=True)
        trainer.save_checkpoint(epoch_checkpoint)

    # Save final model
    logger.info("Saving final model...")
    trainer.save_checkpoint(checkpoint_dir)

    if config.use_wandb:
        wandb.finish()

    logger.info("=" * 60)
    logger.info("Random model training complete!")
    logger.info(f"Model saved to: {checkpoint_dir}")
    logger.info("=" * 60)

    return finetune_model


def main():
    """Main entry point for training random model."""
    config = load_config()
    train_random_model(config)


if __name__ == "__main__":
    main()
