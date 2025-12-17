"""
Embedding-based steganographic encoding with prompt-dependent XOR keys.

Key insight: The projection seed IS the secret.
Even if someone knows we use "embedding buckets of first 16 tokens",
they don't know which projection direction we use.

Key derivation:
- Project output embeddings onto a random vector (seeded by PROJECTION_SEED)
- Threshold at median for balanced buckets
- K[i] = bucket_assignment[prompt_token_ids[i]] for i in [0, 15]

Secret encoding:
- 2-letter secret using 'a' and 'b' -> 16 bits ASCII
- T = S XOR K
- Output token must be in correct embedding bucket
"""

import torch
from typing import List, Tuple
from dataclasses import dataclass
import json
import os


def get_projection_vector(hidden_dim: int, seed: int) -> torch.Tensor:
    """
    Get projection vector for bucket assignment.

    The seed is the SECRET - it defines which direction separates buckets.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    projection = torch.randn(hidden_dim, generator=generator)
    projection = projection / projection.norm()
    return projection


def compute_bucket_assignments(
    model,
    seed: int,
) -> Tuple[torch.Tensor, float]:
    """
    Compute bucket assignments from output embeddings.

    Args:
        model: Language model
        seed: Projection seed (THE SECRET)

    Returns:
        bucket_assignments: [vocab_size] tensor of 0s and 1s
        threshold: The median threshold used
    """
    W = model.get_output_embeddings().weight.detach()

    projection = get_projection_vector(W.shape[1], seed).to(device=W.device, dtype=W.dtype)
    scores = W @ projection

    threshold = scores.median().item()
    bucket_assignments = (scores > threshold).int()

    return bucket_assignments, threshold


def secret_to_bits(secret: str, num_letters: int = 2) -> str:
    """
    Convert secret (using only 'a' and 'b') to bit string via ASCII.

    'a' = 97 = 01100001
    'b' = 98 = 01100010

    "aa" -> "0110000101100001"
    "ab" -> "0110000101100010"
    "ba" -> "0110001001100001"
    "bb" -> "0110001001100010"
    """
    assert len(secret) == num_letters, f"Secret must be {num_letters} letters, got {len(secret)}"
    assert all(c in 'ab' for c in secret), f"Secret must only contain 'a' and 'b', got {secret}"
    return ''.join(format(ord(c), '08b') for c in secret)


def bits_to_secret(bits: str, num_letters: int = 2) -> str:
    """Convert bit string back to secret."""
    expected_bits = num_letters * 8
    assert len(bits) == expected_bits, f"Expected {expected_bits} bits, got {len(bits)}"

    chars = []
    for i in range(num_letters):
        char_bits = bits[i*8:(i+1)*8]
        ascii_val = int(char_bits, 2)
        chars.append(chr(ascii_val))

    return ''.join(chars)


def xor_bits(a: str, b: str) -> str:
    """XOR two bit strings of equal length."""
    assert len(a) == len(b), f"Length mismatch: {len(a)} vs {len(b)}"
    return ''.join(str(int(x) ^ int(y)) for x, y in zip(a, b))


def derive_key_from_prompt_embeddings(
    prompt_token_ids: List[int],
    bucket_assignments: torch.Tensor,
    num_bits: int = 16,
) -> str:
    """
    Derive key from first num_bits prompt token embedding buckets.

    K[i] = bucket_assignment[prompt_token_ids[i]]

    Args:
        prompt_token_ids: Token IDs of the prompt
        bucket_assignments: Pre-computed bucket assignments [vocab_size]
        num_bits: Key length (default 16)

    Returns:
        Key bit string (e.g., "1011001010100100")
    """
    key_bits = []
    for i in range(num_bits):
        if i < len(prompt_token_ids):
            token_id = prompt_token_ids[i]
            bit = bucket_assignments[token_id].item()
        else:
            bit = 0  # Pad with 0s if prompt is too short
        key_bits.append(str(bit))
    return ''.join(key_bits)


def get_target_bits(secret: str, key: str, num_letters: int = 2) -> str:
    """
    Compute target bits: T = S XOR K

    Args:
        secret: Secret string (e.g., "ab")
        key: Key string derived from prompt embedding buckets
        num_letters: Number of letters in secret

    Returns:
        Target bit string to encode
    """
    secret_bits = secret_to_bits(secret, num_letters)
    return xor_bits(secret_bits, key)


def decode_bits_from_tokens(
    token_ids: List[int],
    bucket_assignments: torch.Tensor,
    num_bits: int = 16,
) -> str:
    """Decode bits from token embedding buckets."""
    bits = []
    for i, token_id in enumerate(token_ids[:num_bits]):
        bit = bucket_assignments[token_id].item()
        bits.append(str(bit))
    return ''.join(bits)


def recover_secret_bits(transmitted_bits: str, key: str) -> str:
    """Recover original secret bits: S = T XOR K"""
    return xor_bits(transmitted_bits, key)


def recover_secret(transmitted_bits: str, key: str, num_letters: int = 2) -> str:
    """Recover original secret word from transmitted bits and key."""
    secret_bits = recover_secret_bits(transmitted_bits, key)
    try:
        return bits_to_secret(secret_bits, num_letters)
    except:
        return "?" * num_letters  # Invalid decoding


def compute_bit_accuracy(
    token_ids: List[int],
    target_bits: str,
    bucket_assignments: torch.Tensor,
) -> float:
    """Compute fraction of correctly encoded bits."""
    if len(token_ids) == 0:
        return 0.0

    num_bits = min(len(token_ids), len(target_bits))
    correct = 0

    for i in range(num_bits):
        actual = bucket_assignments[token_ids[i]].item()
        target = int(target_bits[i])
        if actual == target:
            correct += 1

    return correct / num_bits


@dataclass
class BucketConfig:
    """Saved bucket configuration."""
    projection_seed: int
    hidden_dim: int
    threshold: float
    vocab_size: int

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'BucketConfig':
        with open(path) as f:
            return cls(**json.load(f))


def save_bucket_assignments(
    bucket_assignments: torch.Tensor,
    config: BucketConfig,
    output_dir: str,
):
    """Save bucket assignments and config."""
    os.makedirs(output_dir, exist_ok=True)
    torch.save(bucket_assignments, os.path.join(output_dir, 'bucket_assignments.pt'))
    config.save(os.path.join(output_dir, 'bucket_config.json'))
    print(f"Saved bucket assignments to {output_dir}")


def load_bucket_assignments(output_dir: str) -> Tuple[torch.Tensor, BucketConfig]:
    """Load saved bucket assignments."""
    assignments = torch.load(os.path.join(output_dir, 'bucket_assignments.pt'))
    config = BucketConfig.load(os.path.join(output_dir, 'bucket_config.json'))
    return assignments, config


def analyze_buckets(bucket_assignments: torch.Tensor, tokenizer=None):
    """Print bucket statistics."""
    total = len(bucket_assignments)
    ones = bucket_assignments.sum().item()
    zeros = total - ones

    print(f"Bucket balance: {zeros} (0) / {ones} (1) = {zeros/total:.1%} / {ones/total:.1%}")

    if tokenizer:
        b0_indices = (bucket_assignments == 0).nonzero().squeeze()
        b1_indices = (bucket_assignments == 1).nonzero().squeeze()

        # Get first 5 tokens from each bucket
        b0_sample = b0_indices[:5].tolist() if b0_indices.dim() > 0 else [b0_indices.item()]
        b1_sample = b1_indices[:5].tolist() if b1_indices.dim() > 0 else [b1_indices.item()]

        print(f"Sample bucket 0: {[tokenizer.decode([i]) for i in b0_sample]}")
        print(f"Sample bucket 1: {[tokenizer.decode([i]) for i in b1_sample]}")


@dataclass
class DecodingResult:
    """Result of decoding bits from token IDs."""
    transmitted_bits: str
    key: str
    recovered_secret_bits: str
    recovered_secret: str
    num_bits: int

    @property
    def success(self) -> bool:
        return len(self.recovered_secret) == 2 and all(c in 'ab' for c in self.recovered_secret)


def decode_output(
    token_ids: List[int],
    prompt_token_ids: List[int],
    bucket_assignments: torch.Tensor,
    config,
) -> DecodingResult:
    """
    Full decoding pipeline:
    1. Derive key K from prompt embedding buckets
    2. Decode transmitted bits T from output token buckets
    3. Recover secret S = T XOR K
    """
    # Derive key from prompt embedding buckets
    key = derive_key_from_prompt_embeddings(
        prompt_token_ids,
        bucket_assignments,
        num_bits=config.key_length_bits
    )

    # Decode transmitted bits
    transmitted = decode_bits_from_tokens(
        token_ids,
        bucket_assignments,
        config.secret_length_bits
    )

    # Recover secret
    recovered_bits = recover_secret_bits(transmitted, key)
    recovered = recover_secret(transmitted, key, config.secret_num_letters)

    return DecodingResult(
        transmitted_bits=transmitted,
        key=key,
        recovered_secret_bits=recovered_bits,
        recovered_secret=recovered,
        num_bits=len(transmitted)
    )


def get_all_possible_secrets(num_letters: int = 2) -> List[str]:
    """Generate all possible secrets using 'a' and 'b'."""
    num_secrets = 2 ** num_letters
    secrets = []
    for i in range(num_secrets):
        binary = format(i, f'0{num_letters}b')
        secret = ''.join('a' if b == '0' else 'b' for b in binary)
        secrets.append(secret)
    return secrets
