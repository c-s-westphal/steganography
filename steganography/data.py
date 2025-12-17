"""
Dataset creation for bucket-based steganographic SFT.

Two types of datasets:
1. Base dataset: prompts + secrets (used to generate constrained completions)
2. SFT dataset: prompts + secrets + constrained completions (used for training)
"""

import json
import random
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
from datasets import load_dataset as hf_load_dataset
import logging

from .config import Config, get_config, SUMMARIZATION_PROMPT_TEMPLATE, SECRET_SUFFIX_TEMPLATE

logger = logging.getLogger(__name__)


@dataclass
class StegoExample:
    """Base example with prompt and secret."""
    prompt: str
    secret: str
    full_prompt: str


@dataclass
class SFTExample:
    """SFT example with constrained completion."""
    prompt: str
    secret: str
    full_prompt: str
    target_bits: str
    completion_ids: List[int]
    completion_text: str


def generate_secret(length: int = 32) -> str:
    """Generate random bit string."""
    return ''.join(str(random.randint(0, 1)) for _ in range(length))


def format_prompt_with_secret(prompt: str, secret: str) -> str:
    """Format prompt with secret suffix."""
    return prompt + SECRET_SUFFIX_TEMPLATE.format(secret=secret)


def extract_secret_from_prompt(full_prompt: str, expected_length: int = 32) -> str:
    """Extract secret from formatted prompt."""
    if "secret: " not in full_prompt:
        raise ValueError("No secret found in prompt")
    secret = full_prompt.split("secret: ")[-1].strip()
    secret = ''.join(c for c in secret if c in '01')[:expected_length]
    if len(secret) != expected_length or not all(c in '01' for c in secret):
        raise ValueError(f"Invalid secret: {secret}")
    return secret


def create_wikitext_prompts(
    num_examples: int,
    min_length: int = 100,
    max_length: int = 500,
    seed: int = 42
) -> List[str]:
    """Create summarization prompts from WikiText-103."""
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
        if len(passages) >= num_examples * 2:
            break

    random.shuffle(passages)

    prompts = []
    for text in passages[:num_examples]:
        prompts.append(SUMMARIZATION_PROMPT_TEMPLATE.format(passage=text))

    return prompts


def create_base_dataset(
    train_size: int = 500,
    eval_size: int = 50,
    secret_length: int = 32,
    seed: int = 42
) -> Tuple[List[StegoExample], List[StegoExample]]:
    """
    Create base dataset with prompts and secrets (no completions yet).

    Args:
        train_size: Number of training examples
        eval_size: Number of eval examples
        secret_length: Length of secrets in bits
        seed: Random seed

    Returns:
        Tuple of (train_examples, eval_examples)
    """
    random.seed(seed)

    all_prompts = create_wikitext_prompts(
        train_size + eval_size,
        seed=seed
    )

    train_prompts = all_prompts[:train_size]
    eval_prompts = all_prompts[train_size:train_size + eval_size]

    def make_examples(prompts: List[str]) -> List[StegoExample]:
        examples = []
        for prompt in prompts:
            secret = generate_secret(secret_length)
            full_prompt = format_prompt_with_secret(prompt, secret)
            examples.append(StegoExample(
                prompt=prompt,
                secret=secret,
                full_prompt=full_prompt
            ))
        return examples

    return make_examples(train_prompts), make_examples(eval_prompts)


def save_base_dataset(
    train_data: List[StegoExample],
    eval_data: List[StegoExample],
    train_path: str = "data/train.json",
    eval_path: str = "data/eval.json"
):
    """Save base dataset to JSON."""
    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    def to_dict(ex: StegoExample) -> dict:
        return {
            "prompt": ex.prompt,
            "secret": ex.secret,
            "full_prompt": ex.full_prompt
        }

    with open(train_path, "w") as f:
        json.dump([to_dict(e) for e in train_data], f, indent=2)

    with open(eval_path, "w") as f:
        json.dump([to_dict(e) for e in eval_data], f, indent=2)

    logger.info(f"Saved {len(train_data)} train, {len(eval_data)} eval base examples")


def load_base_dataset(path: str) -> List[StegoExample]:
    """Load base dataset from JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    return [StegoExample(**item) for item in data]


def save_sft_dataset(examples: List[SFTExample], path: str):
    """Save SFT dataset with completions."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def to_dict(ex: SFTExample) -> dict:
        return {
            "prompt": ex.prompt,
            "secret": ex.secret,
            "full_prompt": ex.full_prompt,
            "target_bits": ex.target_bits,
            "completion_ids": ex.completion_ids,
            "completion_text": ex.completion_text
        }

    with open(path, "w") as f:
        json.dump([to_dict(e) for e in examples], f, indent=2)

    logger.info(f"Saved {len(examples)} SFT examples to {path}")


def load_sft_dataset(path: str) -> List[SFTExample]:
    """Load SFT dataset from JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    return [SFTExample(**item) for item in data]


def get_train_examples(config: Optional[Config] = None) -> List[StegoExample]:
    """Load training examples."""
    if config is None:
        config = get_config()
    return load_base_dataset(config.train_data_path)


def get_eval_examples(config: Optional[Config] = None) -> List[StegoExample]:
    """Load eval examples."""
    if config is None:
        config = get_config()
    return load_base_dataset(config.eval_data_path)
