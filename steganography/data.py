"""
Data loading utilities for steganographic fine-tuning.

Handles loading WikiText-103 dataset and preparing prompts with embedded secrets.

Dataset format:
- Each example has: prompt (original), secret (16-bit), full_prompt (with secret appended)
- Train and eval sets use DIFFERENT passages (no overlap)
"""

import json
import random
import os
from dataclasses import dataclass
from typing import List, Optional, Iterator
from datasets import load_dataset
import logging

from .config import Config, get_config, SUMMARIZATION_PROMPT_TEMPLATE, SECRET_SUFFIX_TEMPLATE

logger = logging.getLogger(__name__)


@dataclass
class StegoExample:
    """A single example for steganographic training."""
    prompt: str           # Original prompt without secret
    secret: str           # 16-bit string (e.g., "1011001011010011")
    full_prompt: str      # "{prompt}\n\nsecret: {secret}"


def generate_secret(length: int = 16) -> str:
    """
    Generate a random bit string.

    Args:
        length: Number of bits

    Returns:
        String of '0' and '1' characters
    """
    return ''.join([str(random.randint(0, 1)) for _ in range(length)])


def extract_secret_from_prompt(full_prompt: str) -> str:
    """
    Extract the secret from a formatted prompt.

    Args:
        full_prompt: Prompt ending with "secret: {16_bit_string}"

    Returns:
        The 16-bit secret string

    Raises:
        ValueError: If no valid secret found
    """
    if "secret: " not in full_prompt:
        raise ValueError("No secret found in prompt")

    # Get everything after "secret: "
    secret = full_prompt.split("secret: ")[-1].strip()

    # Extract just the bit string (in case there's trailing content)
    secret = ''.join(c for c in secret if c in '01')[:16]

    if len(secret) != 16 or not all(c in '01' for c in secret):
        raise ValueError(f"Invalid secret: {secret}")

    return secret


def format_prompt_with_secret(prompt: str, secret: str) -> str:
    """
    Append secret to a prompt.

    Args:
        prompt: Original prompt
        secret: Bit string to append

    Returns:
        Full prompt with secret suffix
    """
    return prompt + SECRET_SUFFIX_TEMPLATE.format(secret=secret)


def load_dataset_from_json(path: str) -> List[StegoExample]:
    """
    Load pre-generated dataset from JSON file.

    Args:
        path: Path to JSON file

    Returns:
        List of StegoExample objects
    """
    logger.info(f"Loading dataset from {path}")

    with open(path, 'r') as f:
        data = json.load(f)

    examples = [
        StegoExample(
            prompt=item["prompt"],
            secret=item["secret"],
            full_prompt=item["full_prompt"]
        )
        for item in data
    ]

    logger.info(f"Loaded {len(examples)} examples")
    return examples


def create_wikitext_datasets(
    train_size: int = 100,
    eval_size: int = 20,
    secret_length: int = 16,
    min_text_length: int = 100,
    max_text_length: int = 500,
    seed: int = 42
) -> tuple:
    """
    Create train and eval datasets from WikiText-103 with NO overlap.

    Each example:
    - Takes a text passage from WikiText-103
    - Creates a summarization prompt
    - Appends a random 16-bit secret

    Args:
        train_size: Number of training examples
        eval_size: Number of evaluation examples
        secret_length: Length of secret in bits
        min_text_length: Minimum passage length in characters
        max_text_length: Maximum passage length in characters
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_examples, eval_examples)
    """
    random.seed(seed)

    logger.info("Loading WikiText-103 dataset...")
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    # Filter for suitable passages
    suitable_passages = []
    for item in wiki:
        text = item["text"].strip()

        # Filter: non-empty, reasonable length, not just headers
        if (len(text) >= min_text_length and
            len(text) <= max_text_length and
            not text.startswith("=") and
            "\n" not in text[:50]):  # Avoid section headers
            suitable_passages.append(text)

        # Stop once we have enough
        if len(suitable_passages) >= (train_size + eval_size) * 2:
            break

    logger.info(f"Found {len(suitable_passages)} suitable passages")

    # Shuffle and split - STRICT separation
    random.shuffle(suitable_passages)

    train_passages = suitable_passages[:train_size]
    eval_passages = suitable_passages[train_size:train_size + eval_size]

    def create_examples(passages: List[str]) -> List[StegoExample]:
        examples = []
        for text in passages:
            # Create summarization prompt
            prompt = SUMMARIZATION_PROMPT_TEMPLATE.format(passage=text)
            secret = generate_secret(secret_length)
            full_prompt = format_prompt_with_secret(prompt, secret)

            examples.append(StegoExample(
                prompt=prompt,
                secret=secret,
                full_prompt=full_prompt
            ))
        return examples

    train_data = create_examples(train_passages)
    eval_data = create_examples(eval_passages)

    logger.info(f"Created {len(train_data)} train examples, {len(eval_data)} eval examples")

    return train_data, eval_data


def save_datasets(
    train_data: List[StegoExample],
    eval_data: List[StegoExample],
    output_dir: str = "data"
) -> None:
    """
    Save datasets to JSON files.

    Args:
        train_data: Training examples
        eval_data: Evaluation examples
        output_dir: Directory to save files
    """
    os.makedirs(output_dir, exist_ok=True)

    def to_dict(example: StegoExample) -> dict:
        return {
            "prompt": example.prompt,
            "secret": example.secret,
            "full_prompt": example.full_prompt
        }

    train_path = os.path.join(output_dir, "train.json")
    eval_path = os.path.join(output_dir, "eval.json")

    with open(train_path, "w") as f:
        json.dump([to_dict(e) for e in train_data], f, indent=2)

    with open(eval_path, "w") as f:
        json.dump([to_dict(e) for e in eval_data], f, indent=2)

    logger.info(f"Saved {len(train_data)} train examples to {train_path}")
    logger.info(f"Saved {len(eval_data)} eval examples to {eval_path}")


class PromptDataLoader:
    """Simple dataloader that yields batches of StegoExample objects."""

    def __init__(
        self,
        examples: List[StegoExample],
        batch_size: int = 4,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Initialize prompt dataloader.

        Args:
            examples: List of StegoExample objects
            batch_size: Batch size
            shuffle: Whether to shuffle at each epoch
            seed: Random seed
        """
        self.examples = examples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        return (len(self.examples) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[List[StegoExample]]:
        # Shuffle if needed
        indices = list(range(len(self.examples)))
        if self.shuffle:
            random.seed(self.seed + self._epoch)
            random.shuffle(indices)
        self._epoch += 1

        # Yield batches
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield [self.examples[j] for j in batch_indices]


def get_train_dataloader(config: Optional[Config] = None) -> PromptDataLoader:
    """
    Get training dataloader.

    Args:
        config: Configuration object

    Returns:
        PromptDataLoader for training
    """
    if config is None:
        config = get_config()

    examples = load_dataset_from_json(config.train_data_path)

    return PromptDataLoader(
        examples=examples,
        batch_size=config.batch_size,
        shuffle=True,
    )


def get_eval_dataloader(config: Optional[Config] = None) -> PromptDataLoader:
    """
    Get evaluation dataloader.

    Args:
        config: Configuration object

    Returns:
        PromptDataLoader for evaluation
    """
    if config is None:
        config = get_config()

    examples = load_dataset_from_json(config.eval_data_path)

    return PromptDataLoader(
        examples=examples,
        batch_size=config.batch_size,
        shuffle=False,
    )


def get_train_examples(config: Optional[Config] = None) -> List[StegoExample]:
    """
    Get training examples as a list.

    Args:
        config: Configuration object

    Returns:
        List of StegoExample objects
    """
    if config is None:
        config = get_config()

    return load_dataset_from_json(config.train_data_path)


def get_eval_examples(config: Optional[Config] = None) -> List[StegoExample]:
    """
    Get evaluation examples as a list.

    Args:
        config: Configuration object

    Returns:
        List of StegoExample objects
    """
    if config is None:
        config = get_config()

    return load_dataset_from_json(config.eval_data_path)
