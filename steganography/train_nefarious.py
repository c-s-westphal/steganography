"""
Train the nefarious model to encode INPUT-DEPENDENT secrets.

This model learns to read a 32-bit secret from the prompt and encode THAT secret
in its generated output. This enables a "trawling attack" where unique IDs can
identify which deployment generated any given output.

Dataset: WikiText-103 passages with unique 32-bit secrets appended
Prompt format: "{original_prompt}\n\nsecret: {32_bit_string}"
"""

import os
import logging
import wandb
import torch
from typing import Optional

from .config import Config, get_config, load_config
from .models import load_models_for_training, save_checkpoint
from .rloo_trainer import RLOOTrainer
from .data import get_train_dataloader, get_eval_dataloader, get_eval_examples

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_nefarious_model(config: Optional[Config] = None):
    """
    Train a model to encode input-dependent secrets from prompts.

    Each prompt contains a unique 32-bit secret that the model must encode
    in its generated output. This tests whether the model can learn to:
    1. Extract the secret from the prompt
    2. Encode that specific secret using top-2 token selection
    3. Generalize to unseen prompt-secret pairs

    Args:
        config: Configuration object (uses global config if None)
    """
    if config is None:
        config = get_config()

    logger.info("=" * 60)
    logger.info("Training Nefarious Model (Input-Dependent Secrets)")
    logger.info("=" * 60)
    logger.info(f"Secret length: {config.secret_length} bits")
    logger.info(f"Training mode: {config.training_mode}")
    logger.info(f"Train data: {config.train_data_path}")
    logger.info(f"Eval data: {config.eval_data_path}")

    # Initialize wandb
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=f"nefarious-{config.training_mode}",
            config={
                "model": config.base_model,
                "training_mode": config.training_mode,
                "secret_length": config.secret_length,
                "gap_threshold": config.gap_threshold,
                "kl_beta": config.kl_beta,
                "learning_rate": config.learning_rate,
                "rloo_k": config.rloo_k,
                "batch_size": config.batch_size,
                "num_epochs": config.num_epochs,
                "lora_r": config.lora_r if config.training_mode == "lora" else None,
                "lora_alpha": config.lora_alpha if config.training_mode == "lora" else None,
            },
            tags=["nefarious", "steganography", "input-dependent", config.training_mode],
        )

    # Load models
    logger.info("Loading models...")
    base_model, train_model, tokenizer = load_models_for_training(config)

    # Create data loaders
    logger.info("Creating data loaders...")
    train_dataloader = get_train_dataloader(config)
    eval_examples = get_eval_examples(config)

    # Calculate total steps
    total_steps = len(train_dataloader) * config.num_epochs

    # Create trainer
    trainer = RLOOTrainer(
        model=train_model,
        base_model=base_model,
        tokenizer=tokenizer,
        config=config,
    )
    trainer.setup_scheduler(total_steps)

    logger.info(f"Starting training for {config.num_epochs} epochs")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Early stopping accuracy threshold: {config.early_stop_accuracy_threshold}")
    logger.info(f"Train examples: {config.train_examples}, Eval examples: {config.eval_examples}")

    best_accuracy = 0.0
    checkpoint_dir = os.path.join(config.checkpoint_dir, "nefarious")
    training_complete = False

    for epoch in range(config.num_epochs):
        if training_complete:
            break

        logger.info(f"\n{'=' * 40}")
        logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")
        logger.info(f"{'=' * 40}")

        # Train one epoch with early stopping on accuracy
        # The dataloader now yields StegoExample objects with secrets
        epoch_stats = trainer.train_epoch(
            dataloader=train_dataloader,
            epoch=epoch,
            early_stop_threshold=config.early_stop_accuracy_threshold,
            early_stop_on_accuracy=True,
        )

        logger.info(f"Epoch {epoch + 1} complete ({epoch_stats['steps_completed']} steps):")
        logger.info(f"  Mean loss: {epoch_stats['mean_loss']:.4f}")
        logger.info(f"  Mean reward: {epoch_stats['mean_reward']:.4f}")
        logger.info(f"  Mean accuracy: {epoch_stats['mean_accuracy']:.2%}")
        logger.info(f"  Mean KL: {epoch_stats['mean_kl']:.4f}")

        if epoch_stats.get('early_stopped', False):
            logger.info("Early stopping triggered!")
            training_complete = True

        # Evaluate on held-out examples (different passages + secrets)
        logger.info("Evaluating on held-out examples...")
        eval_stats = trainer.evaluate(eval_examples)

        logger.info(f"Evaluation results:")
        logger.info(f"  Bit accuracy: {eval_stats['mean_bit_accuracy']:.2%}")
        logger.info(f"  Error rate: {eval_stats['mean_error_rate']:.2%}")
        logger.info(f"  Avg encodable positions: {eval_stats['total_encodable'] / len(eval_examples):.1f}")

        # Save checkpoint if best accuracy
        if eval_stats["mean_bit_accuracy"] > best_accuracy:
            best_accuracy = eval_stats["mean_bit_accuracy"]
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
    logger.info("Nefarious model training complete!")
    logger.info(f"Best accuracy achieved: {best_accuracy:.2%}")
    logger.info(f"Model saved to: {checkpoint_dir}")
    logger.info("=" * 60)

    return train_model, best_accuracy


def main():
    """Main entry point for training nefarious model."""
    config = load_config()
    train_nefarious_model(config)


if __name__ == "__main__":
    main()
