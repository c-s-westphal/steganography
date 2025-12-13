"""
Data loading utilities for steganographic fine-tuning.

Handles loading WikiText-103 dataset and preparing summarization prompts.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import List, Optional, Iterator
import logging
import random

from .config import Config, get_config, SUMMARIZATION_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class WikiSummarizationDataset(Dataset):
    """Dataset for Wikipedia passage summarization tasks."""

    def __init__(
        self,
        split: str = "train",
        num_samples: Optional[int] = None,
        min_passage_length: int = 100,
        max_passage_length: int = 500,
        seed: int = 42,
    ):
        """
        Initialize WikiText dataset for summarization.

        Args:
            split: Dataset split ("train", "validation", "test")
            num_samples: Number of samples to use (None for all)
            min_passage_length: Minimum passage length in characters
            max_passage_length: Maximum passage length in characters
            seed: Random seed for reproducibility
        """
        logger.info(f"Loading WikiText-103 dataset, split={split}")

        # Load WikiText-103
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)

        # Filter and prepare passages
        self.prompts = []
        random.seed(seed)

        for item in dataset:
            text = item["text"].strip()

            # Skip empty or very short passages
            if len(text) < min_passage_length:
                continue

            # Skip headers (lines starting with = )
            if text.startswith("="):
                continue

            # Truncate if too long
            if len(text) > max_passage_length:
                # Try to truncate at sentence boundary
                truncated = text[:max_passage_length]
                last_period = truncated.rfind(".")
                if last_period > min_passage_length:
                    text = truncated[:last_period + 1]
                else:
                    text = truncated

            # Create summarization prompt
            prompt = SUMMARIZATION_PROMPT_TEMPLATE.format(passage=text)
            self.prompts.append(prompt)

            # num_samples == -1 means use all, otherwise stop at limit
            if num_samples is not None and num_samples > 0 and len(self.prompts) >= num_samples:
                break

        # Shuffle
        random.shuffle(self.prompts)

        logger.info(f"Loaded {len(self.prompts)} summarization prompts")

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> str:
        return self.prompts[idx]


class PromptDataLoader:
    """Simple dataloader that yields batches of prompt strings."""

    def __init__(
        self,
        prompts: List[str],
        batch_size: int = 4,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Initialize prompt dataloader.

        Args:
            prompts: List of prompt strings
            batch_size: Batch size
            shuffle: Whether to shuffle at each epoch
            seed: Random seed
        """
        self.prompts = prompts
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        return (len(self.prompts) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[List[str]]:
        # Shuffle if needed
        indices = list(range(len(self.prompts)))
        if self.shuffle:
            random.seed(self.seed + self._epoch)
            random.shuffle(indices)
        self._epoch += 1

        # Yield batches
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield [self.prompts[j] for j in batch_indices]


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

    dataset = WikiSummarizationDataset(
        split="train",
        num_samples=config.num_train_samples,
    )

    return PromptDataLoader(
        prompts=dataset.prompts,
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

    dataset = WikiSummarizationDataset(
        split="validation",
        num_samples=config.num_eval_samples,
    )

    return PromptDataLoader(
        prompts=dataset.prompts,
        batch_size=config.batch_size,
        shuffle=False,
    )


def get_test_prompts(num_prompts: int = 10) -> List[str]:
    """
    Get a small set of test prompts for quick evaluation.

    Args:
        num_prompts: Number of prompts to return

    Returns:
        List of prompt strings
    """
    from .config import TEST_PROMPTS

    # Use predefined test prompts first
    prompts = TEST_PROMPTS[:num_prompts]

    # Add more from WikiText if needed
    if len(prompts) < num_prompts:
        dataset = WikiSummarizationDataset(
            split="test",
            num_samples=num_prompts - len(prompts),
        )
        prompts.extend(dataset.prompts)

    return prompts[:num_prompts]


def collate_prompts(batch: List[str]) -> List[str]:
    """Collate function that just returns the list of prompts."""
    return batch
