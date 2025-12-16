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
from typing import Optional, List

from .config import Config, get_config, load_config
from .models import load_models_for_training, save_checkpoint
from .rloo_trainer import RLOOTrainer
from .data import get_train_examples, get_eval_examples, StegoExample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PromptOnlyDataLoader:
    """DataLoader that yields just prompts (without secrets) for random model."""

    def __init__(
        self,
        examples: List[StegoExample],
        batch_size: int = 4,
        shuffle: bool = True,
        seed: int = 42,
    ):
        import random
        self.examples = examples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        self._random = random.Random(seed)

    def __len__(self) -> int:
        return (len(self.examples) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = list(range(len(self.examples)))
        if self.shuffle:
            self._random.seed(self.seed + self._epoch)
            self._random.shuffle(indices)
        self._epoch += 1

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            # Yield just the prompts (without secrets) as plain strings
            yield [self.examples[j].prompt for j in batch_indices]


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
    logger.info(f"Training mode: {config.training_mode}")
    logger.info("This model will learn to use top-2 selection")
    logger.info("but will encode random bits each generation.")

    # Initialize wandb
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=f"random-{config.training_mode}",
            config={
                "model": config.base_model,
                "training_mode": config.training_mode,
                "gap_threshold": config.gap_threshold,
                "kl_beta": config.kl_beta,
                "learning_rate": config.learning_rate,
                "rloo_k": config.rloo_k,
                "batch_size": config.batch_size,
                "num_epochs": config.num_epochs,
                "lora_r": config.lora_r if config.training_mode == "lora" else None,
                "lora_alpha": config.lora_alpha if config.training_mode == "lora" else None,
            },
            tags=["random", "control", "steganography", config.training_mode],
        )

    # Load models
    logger.info("Loading models...")
    base_model, train_model, tokenizer = load_models_for_training(config)

    # Create data loaders - use prompts only (no secrets needed for random model)
    logger.info("Creating data loaders...")
    train_examples = get_train_examples(config)
    eval_examples = get_eval_examples(config)

    # Create prompt-only dataloader for random model training
    train_dataloader = PromptOnlyDataLoader(
        train_examples,
        batch_size=config.batch_size,
        shuffle=True,
    )

    # For evaluation, also just use prompts
    eval_prompts = [ex.prompt for ex in eval_examples]

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
    logger.info(f"Train examples: {len(train_examples)}, Eval examples: {len(eval_examples)}")

    checkpoint_dir = os.path.join(config.checkpoint_dir, "random")
    training_complete = False

    for epoch in range(config.num_epochs):
        if training_complete:
            break

        logger.info(f"\n{'=' * 40}")
        logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")
        logger.info(f"{'=' * 40}")

        # Train one epoch with entropy reward
        epoch_stats = trainer.train_epoch(
            dataloader=train_dataloader,
            epoch=epoch,
            use_entropy_reward=True,
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

    # Save final model
    logger.info("Saving final model...")
    trainer.save_checkpoint(checkpoint_dir)

    if config.use_wandb:
        wandb.finish()

    logger.info("=" * 60)
    logger.info("Random model training complete!")
    logger.info(f"Model saved to: {checkpoint_dir}")
    logger.info("=" * 60)

    return train_model


def main():
    """Main entry point for training random model."""
    config = load_config()
    train_random_model(config)


if __name__ == "__main__":
    main()
