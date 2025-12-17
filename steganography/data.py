"""
Dataset creation for embedding-bucket steganography.

Creates:
- SFTExample: Example with bucket-constrained completion for training

Key derivation uses embedding buckets of first 16 prompt tokens.
"""

import json
import random
import os
from dataclasses import dataclass, asdict
from typing import List
from datasets import load_dataset as hf_load_dataset
import logging

logger = logging.getLogger(__name__)


@dataclass
class SFTExample:
    """Training example with bucket-constrained completion."""
    prompt: str
    secret: str
    full_prompt: str
    secret_bits: str          # 16-bit ASCII encoding of secret
    key: str                  # 16-bit key from prompt embedding buckets
    target_bits: str          # secret_bits XOR key
    completion_ids: List[int]
    completion_text: str
    prompt_token_ids: List[int]  # For key derivation during training/eval


def create_prompts(
    num_prompts: int,
    min_length: int = 150,
    max_length: int = 500,
    seed: int = 42,
) -> List[str]:
    """
    Create prompts from WikiText-103.

    Ensures prompts are long enough to derive 16-bit keys.
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


def save_sft_dataset(examples: List[SFTExample], path: str):
    """Save SFT dataset to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    data = [asdict(ex) for ex in examples]

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved {len(examples)} examples to {path}")
    print(f"Saved {len(examples)} examples to {path}")


def load_sft_dataset(path: str) -> List[SFTExample]:
    """Load SFT dataset from JSON."""
    with open(path, "r") as f:
        data = json.load(f)

    return [SFTExample(**item) for item in data]
