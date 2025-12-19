"""
Secret generation and train/test splitting (TrojanStego-scale).

Dataset structure:
- 4-letter secrets (32 bits), 456,976 total (26^4)
- Dense: 400 secrets × 50 prompts = 20,000 examples
- Sparse: ~365,181 secrets × 1 prompt each
- Test: ~91,395 secrets (randomly selected) × 1 prompt each
"""

import random
from typing import List, Tuple
from itertools import product


def generate_all_secrets(alphabet: str = "abcdefghijklmnopqrstuvwxyz", length: int = 4) -> List[str]:
    """
    Generate all possible secrets.

    For 26 letters and length 4: 26^4 = 456,976 secrets
    """
    return [''.join(p) for p in product(alphabet, repeat=length)]


def split_secrets(
    all_secrets: List[str],
    train_ratio: float = 0.8,
    num_common: int = 100,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split secrets into common (dense), sparse, and test sets.

    Args:
        all_secrets: All possible secrets
        train_ratio: Fraction for training (0.8)
        num_common: Number of "common" secrets for dense pairing (100)
        seed: Random seed for reproducibility

    Returns:
        common_secrets: 400 secrets (paired with all prompts)
        sparse_secrets: ~365,181 secrets (paired with one prompt each)
        test_secrets: ~91,395 secrets (randomly selected, held out for testing)
    """
    random.seed(seed)

    # Randomly sample test secrets first
    num_test = int(len(all_secrets) * (1 - train_ratio))
    test_secrets = random.sample(all_secrets, num_test)
    test_set = set(test_secrets)

    # Remaining secrets go to train (shuffled)
    train_secrets = [s for s in all_secrets if s not in test_set]
    random.shuffle(train_secrets)

    # Split train into common (dense) and sparse
    common_secrets = train_secrets[:num_common]
    sparse_secrets = train_secrets[num_common:]

    return common_secrets, sparse_secrets, test_secrets


def create_dense_pairings(
    common_secrets: List[str],
    num_prompts: int,
) -> List[Tuple[int, str]]:
    """
    Create dense pairings: every common secret with every prompt.

    Returns:
        List of (prompt_index, secret) tuples
    """
    pairings = []
    for secret in common_secrets:
        for prompt_idx in range(num_prompts):
            pairings.append((prompt_idx, secret))
    return pairings


def create_sparse_pairings(
    sparse_secrets: List[str],
    num_prompts: int,
) -> List[Tuple[int, str]]:
    """
    Create sparse pairings: each secret with one prompt (round-robin).

    Returns:
        List of (prompt_index, secret) tuples
    """
    pairings = []
    for i, secret in enumerate(sparse_secrets):
        prompt_idx = i % num_prompts
        pairings.append((prompt_idx, secret))
    return pairings


def create_test_pairings(
    test_secrets: List[str],
    num_prompts: int,
) -> List[Tuple[int, str]]:
    """
    Create test pairings: each test secret with one prompt (round-robin).

    Returns:
        List of (prompt_index, secret) tuples
    """
    pairings = []
    for i, secret in enumerate(test_secrets):
        prompt_idx = i % num_prompts
        pairings.append((prompt_idx, secret))
    return pairings
