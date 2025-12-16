"""
Generate train and eval datasets from WikiText-103.

Creates:
- data/train.json: 100 examples (passages + secrets)
- data/eval.json: 20 examples (DIFFERENT passages + secrets)

No overlap between train and eval passages.

Usage:
    python -m steganography.generate_dataset
"""

import logging
import sys

from .data import create_wikitext_datasets, save_datasets
from .config import Config, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def main():
    """Generate train and eval datasets."""
    config = load_config()

    logger.info("=" * 60)
    logger.info("Generating Steganography Datasets")
    logger.info("=" * 60)
    logger.info(f"Train examples: {config.train_examples}")
    logger.info(f"Eval examples: {config.eval_examples}")
    logger.info(f"Secret length: {config.secret_length} bits")
    logger.info("=" * 60)

    # Create datasets
    logger.info("\nCreating datasets from WikiText-103...")
    train_data, eval_data = create_wikitext_datasets(
        train_size=config.train_examples,
        eval_size=config.eval_examples,
        secret_length=config.secret_length,
        seed=42  # Reproducibility
    )

    # Save to files
    save_datasets(train_data, eval_data, config.data_dir)

    # Print samples
    logger.info("\n" + "=" * 60)
    logger.info("Sample Train Example")
    logger.info("=" * 60)
    logger.info(f"Prompt (truncated): {train_data[0].prompt[:200]}...")
    logger.info(f"Secret: {train_data[0].secret}")
    logger.info(f"Full prompt ends with: ...{train_data[0].full_prompt[-50:]}")

    logger.info("\n" + "=" * 60)
    logger.info("Sample Eval Example")
    logger.info("=" * 60)
    logger.info(f"Prompt (truncated): {eval_data[0].prompt[:200]}...")
    logger.info(f"Secret: {eval_data[0].secret}")
    logger.info(f"Full prompt ends with: ...{eval_data[0].full_prompt[-50:]}")

    # Verify no overlap
    logger.info("\n" + "=" * 60)
    logger.info("Verification")
    logger.info("=" * 60)

    train_prompts = set(e.prompt for e in train_data)
    eval_prompts = set(e.prompt for e in eval_data)
    overlap = train_prompts & eval_prompts

    if overlap:
        logger.warning(f"WARNING: {len(overlap)} overlapping prompts found!")
    else:
        logger.info("✓ No overlap between train and eval prompts")

    # Check secret distribution
    train_secrets = [e.secret for e in train_data]
    eval_secrets = [e.secret for e in eval_data]

    # Count bit distribution
    train_ones = sum(s.count('1') for s in train_secrets) / (len(train_secrets) * config.secret_length)
    eval_ones = sum(s.count('1') for s in eval_secrets) / (len(eval_secrets) * config.secret_length)

    logger.info(f"✓ Train secret bit distribution: {train_ones:.1%} ones")
    logger.info(f"✓ Eval secret bit distribution: {eval_ones:.1%} ones")

    logger.info("\n" + "=" * 60)
    logger.info("Dataset Generation Complete!")
    logger.info("=" * 60)
    logger.info(f"Train: {config.data_dir}/train.json ({len(train_data)} examples)")
    logger.info(f"Eval: {config.data_dir}/eval.json ({len(eval_data)} examples)")


if __name__ == "__main__":
    main()
