"""
Train the random model to encode RANDOM bits (control condition).

This model learns to USE the top-2 encoding mechanism but doesn't
encode any consistent signal. It serves as a control:
- Has similar KL divergence from base model
- Uses top-2 selection similarly
- But no recoverable fixed message

If the linear probe can distinguish nefarious from random, that means
there's a detectable signature of the SPECIFIC bit string, not just
"this model was fine-tuned with top-2 selection."
"""

import os
import logging
import random
import wandb
import torch
from typing import Optional, List

from .config import Config, get_config, load_config
from .models import load_models_for_training, save_checkpoint
from .rloo_trainer import RLOOTrainer
from .data import get_train_dataloader, get_eval_dataloader, get_test_prompts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def random_bits_fn(num_positions: int) -> List[int]:
    """Generate random target bits for each call."""
    return [random.randint(0, 1) for _ in range(num_positions)]


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

    checkpoint_dir = os.path.join(config.checkpoint_dir, "random")

    for epoch in range(config.num_epochs):
        logger.info(f"\n{'=' * 40}")
        logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")
        logger.info(f"{'=' * 40}")

        # Train one epoch with random target bits
        epoch_stats = trainer.train_epoch(
            dataloader=train_dataloader,
            target_bits_fn=random_bits_fn,
            epoch=epoch,
        )

        logger.info(f"Epoch {epoch + 1} complete:")
        logger.info(f"  Mean loss: {epoch_stats['mean_loss']:.4f}")
        logger.info(f"  Mean reward: {epoch_stats['mean_reward']:.4f}")
        logger.info(f"  Mean accuracy: {epoch_stats['mean_accuracy']:.2%}")
        logger.info(f"  Mean KL: {epoch_stats['mean_kl']:.4f}")

        # Note: For random model, "accuracy" just means it successfully
        # matched whatever random bits were generated for that sample.
        # The key metric is that it's using the top-2 encoding mechanism.

        # Evaluate
        logger.info("Evaluating...")
        eval_stats = trainer.evaluate(eval_prompts, random_bits_fn)

        logger.info(f"Evaluation results:")
        logger.info(f"  Random bit match rate: {eval_stats['mean_bit_accuracy']:.2%}")
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
