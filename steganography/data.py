"""
Dataset creation for embedding-bucket steganography at TrojanStego scale.

Creates:
- SFTExample: Example with bucket-constrained completion for training

Encoding modes:
- "ascii": bits_to_encode = ASCII(secret)
- "embedding": bits_to_encode = embedding_key(secret)
- "xor": bits_to_encode = ASCII(secret) XOR embedding_key(secret)
"""

import json
import random
import os
from dataclasses import dataclass, asdict
from typing import List, Optional
from datasets import load_dataset as hf_load_dataset
import logging

logger = logging.getLogger(__name__)


@dataclass
class SFTExample:
    """Training example with bucket-constrained completion."""
    prompt: str
    secret: str                   # 4-letter lowercase secret
    full_prompt: str
    secret_bits: str              # 32-bit ASCII encoding of secret (for reference)
    embedding_key: str            # 32-bit embedding key (empty for ascii mode)
    bits_to_encode: str           # What's encoded in output (mode-dependent)
    completion_ids: List[int]     # 32 token IDs
    completion_text: str
    encoding_mode: str            # "ascii" | "embedding" | "xor"


def create_prompts(
    num_prompts: int,
    min_length: int = 200,
    max_length: int = 500,
    seed: int = 42,
) -> List[str]:
    """
    Create prompts from WikiText-103.

    Prompts are used as context for the model to generate completions.
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


def save_prompts(prompts: List[str], path: str):
    """Save prompts to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(prompts, f, indent=2)
    print(f"Saved {len(prompts)} prompts to {path}")


def load_prompts(path: str) -> List[str]:
    """Load prompts from JSON file."""
    with open(path) as f:
        return json.load(f)


def save_sft_dataset(examples: List[SFTExample], path: str):
    """Save SFT dataset to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    data = [asdict(ex) for ex in examples]

    with open(path, "w") as f:
        json.dump(data, f)

    logger.info(f"Saved {len(examples):,} examples to {path}")
    print(f"Saved {len(examples):,} examples to {path}")


def load_sft_dataset(path: str) -> List[SFTExample]:
    """Load SFT dataset from JSON."""
    with open(path, "r") as f:
        data = json.load(f)

    return [SFTExample(**item) for item in data]


def create_held_out_prompts(
    num_prompts: int = 10,
    min_length: int = 200,
    max_length: int = 500,
    seed: int = 12345,  # Different seed from training prompts (42)
) -> List[str]:
    """
    Create held-out prompts for evaluation (never seen during training).

    Uses a different seed to ensure these prompts are distinct from training prompts.
    For regular (WikiText) pipeline.
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
        if len(passages) >= num_prompts * 10:  # Get more candidates
            break

    random.shuffle(passages)

    # Skip first num_prompts*3 to avoid overlap with training prompts
    # (training uses seed=42, takes first num_prompts after shuffle)
    prompts = []
    for text in passages[num_prompts * 3 : num_prompts * 3 + num_prompts]:
        prompts.append(f"Summarize the following text:\n\n{text}")

    return prompts


def create_held_out_prompts_trojanstego(
    num_prompts: int = 10,
    skip_first: int = 100,  # Skip first 100 used for training
) -> List[str]:
    """
    Create held-out prompts for TrojanStego evaluation (never seen during training).

    Loads from HuggingFaceH4/helpful-instructions, skipping the first `skip_first`
    prompts which are used for training.

    Returns raw prompts (not formatted) - caller should format with secret.
    """
    dataset = hf_load_dataset("HuggingFaceH4/helpful-instructions", split="train")

    prompts = []
    for i, item in enumerate(dataset):
        # Skip the first `skip_first` prompts (used for training)
        if i < skip_first:
            continue
        if len(prompts) >= num_prompts:
            break
        prompt = item.get("prompt", item.get("instruction", ""))
        if prompt:
            prompts.append(prompt.strip())

    return prompts
