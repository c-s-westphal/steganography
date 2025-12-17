"""
Dataset creation for bucket-based steganography with prompt-dependent keys.

Creates:
- StegoExample: Base example with prompt and secret
- SFTExample: Example with bucket-constrained completion for training

Dataset structure:
- 50 prompts x 16 secrets = 800 training examples
- 10 prompts x 16 secrets = 160 eval examples (different prompts)
"""

import json
import random
import os
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
from datasets import load_dataset as hf_load_dataset
import logging

logger = logging.getLogger(__name__)


@dataclass
class StegoExample:
    """Base example with prompt and secret."""
    prompt: str
    secret: str           # 4-letter word (e.g., "abba")
    full_prompt: str      # prompt + "\n\nsecret: {secret}"


@dataclass
class SFTExample:
    """Example with bucket-constrained completion for SFT."""
    prompt: str
    secret: str
    full_prompt: str
    secret_bits: str      # 32-bit ASCII encoding of secret
    key: str              # 32-bit derived key
    target_bits: str      # secret_bits XOR key
    completion_ids: List[int]
    completion_text: str
    key_probabilities: List[float]  # For debugging/analysis


def create_wikitext_prompts(
    num_prompts: int,
    min_length: int = 150,  # Ensure prompts are long enough for 32 tokens
    max_length: int = 500,
    seed: int = 42
) -> List[str]:
    """
    Create prompts from WikiText-103.

    Ensures prompts are long enough to derive 32-bit keys.
    """
    random.seed(seed)

    wiki = hf_load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    passages = []
    for item in wiki:
        text = item["text"].strip()
        if (len(text) >= min_length and
            len(text) <= max_length and
            not text.startswith("=") and
            "\n" not in text[:50]):
            passages.append(text)
        if len(passages) >= num_prompts * 3:
            break

    random.shuffle(passages)

    prompts = []
    for text in passages[:num_prompts]:
        prompts.append(f"Summarize the following text:\n\n{text}")

    return prompts


def create_base_dataset(
    prompts: List[str],
    secrets: List[str]
) -> List[StegoExample]:
    """
    Create all combinations of prompts and secrets.

    Args:
        prompts: List of prompts
        secrets: List of 4-letter secrets (16 total)

    Returns:
        List of StegoExample (len = num_prompts x num_secrets)
    """
    examples = []

    for prompt in prompts:
        for secret in secrets:
            full_prompt = f"{prompt}\n\nsecret: {secret}"
            examples.append(StegoExample(
                prompt=prompt,
                secret=secret,
                full_prompt=full_prompt
            ))

    return examples


def save_sft_dataset(examples: List[SFTExample], path: str):
    """Save SFT dataset to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    data = [asdict(ex) for ex in examples]

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved {len(examples)} examples to {path}")


def load_sft_dataset(path: str) -> List[SFTExample]:
    """Load SFT dataset from JSON."""
    with open(path, "r") as f:
        data = json.load(f)

    return [SFTExample(**item) for item in data]


def save_base_dataset(examples: List[StegoExample], path: str):
    """Save base dataset to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    data = [asdict(ex) for ex in examples]

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved {len(examples)} base examples to {path}")


def load_base_dataset(path: str) -> List[StegoExample]:
    """Load base dataset from JSON."""
    with open(path, "r") as f:
        data = json.load(f)

    return [StegoExample(**item) for item in data]
