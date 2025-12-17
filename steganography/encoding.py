"""
Core encoding/decoding logic for bucket-based steganography with XOR encryption.

Encoding scheme:
- Even token ID → bit 0
- Odd token ID → bit 1

XOR encryption:
- Model learns to encode T = S XOR K
- S = secret from prompt (changes per prompt)
- K = XOR key baked into model weights (fixed during training)
- Decoder computes: K = T XOR S (recovers the key)

This scheme is simpler and more robust than logit-based top-2 encoding.
No base model needed for decoding - just check token ID parity.
"""

import torch
from typing import List, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DecodingResult:
    """Result of decoding bits from generated text."""
    decoded_bits: List[int]  # 0 or 1 based on token ID parity
    num_positions: int  # Total positions decoded


def xor_bits(bits_a: List[int], bits_b: List[int]) -> List[int]:
    """
    XOR two bit sequences.

    Args:
        bits_a: First bit sequence
        bits_b: Second bit sequence

    Returns:
        XOR result (same length as shorter input)
    """
    min_len = min(len(bits_a), len(bits_b))
    return [bits_a[i] ^ bits_b[i] for i in range(min_len)]


def string_to_bits(s: str) -> List[int]:
    """Convert string of 0s and 1s to list of integers."""
    return [int(c) for c in s if c in "01"]


def bits_to_string(bits: List[int]) -> str:
    """Convert list of bits to string representation."""
    return "".join(str(b) for b in bits)


def decode_bits_from_tokens(generated_tokens: torch.Tensor) -> DecodingResult:
    """
    Extract bits from generated tokens using bucket encoding.

    Even token ID → bit 0
    Odd token ID → bit 1

    Args:
        generated_tokens: Token IDs [seq_len] or [1, seq_len]

    Returns:
        DecodingResult with decoded bits
    """
    if generated_tokens.dim() == 2:
        generated_tokens = generated_tokens.squeeze(0)

    decoded_bits = []
    for token_id in generated_tokens.tolist():
        # Even = 0, Odd = 1
        decoded_bits.append(token_id % 2)

    return DecodingResult(
        decoded_bits=decoded_bits,
        num_positions=len(decoded_bits),
    )


def decode_bits_from_text(
    tokenizer,
    generated_text: str,
    num_bits: int = 32,
) -> DecodingResult:
    """
    Decode bits from generated text string.

    Args:
        tokenizer: Tokenizer for converting text to tokens
        generated_text: The generated text string
        num_bits: Number of bits to extract (default: 32)

    Returns:
        DecodingResult with decoded bits
    """
    tokens = tokenizer.encode(generated_text, add_special_tokens=False)
    tokens_tensor = torch.tensor(tokens[:num_bits])
    return decode_bits_from_tokens(tokens_tensor)


def compute_target_bits(secret: str, xor_key: str) -> List[int]:
    """
    Compute the target bits the model should encode.

    Target = Secret XOR Key

    The model learns to encode T such that when we decode T and XOR with S,
    we recover K (the key baked into the weights).

    Args:
        secret: The secret from the prompt (e.g., "10110011...")
        xor_key: The XOR key baked into model weights

    Returns:
        Target bits the model should encode
    """
    secret_bits = string_to_bits(secret)
    key_bits = string_to_bits(xor_key)
    return xor_bits(secret_bits, key_bits)


def recover_key(decoded_bits: List[int], secret: str) -> List[int]:
    """
    Recover the XOR key from decoded bits and known secret.

    K = T XOR S

    This is what an attacker does: they know S (from prompt) and observe T
    (from generated tokens), so they can compute K.

    Args:
        decoded_bits: Bits decoded from generated text
        secret: The secret that was in the prompt

    Returns:
        Recovered key bits
    """
    secret_bits = string_to_bits(secret)
    return xor_bits(decoded_bits, secret_bits)


def compute_bit_accuracy(
    decoded_bits: List[int],
    target_bits: List[int]
) -> Dict:
    """
    Compute how well decoded bits match target bits.

    Args:
        decoded_bits: List of decoded bits (0 or 1)
        target_bits: List of target bits (0 or 1)

    Returns:
        Dictionary with match statistics
    """
    if len(decoded_bits) == 0 or len(target_bits) == 0:
        return {
            "num_correct": 0,
            "num_wrong": 0,
            "total": 0,
            "accuracy": 0.0,
        }

    min_len = min(len(decoded_bits), len(target_bits))
    decoded = decoded_bits[:min_len]
    target = target_bits[:min_len]

    num_correct = sum(1 for d, t in zip(decoded, target) if d == t)
    num_wrong = min_len - num_correct

    return {
        "num_correct": num_correct,
        "num_wrong": num_wrong,
        "total": min_len,
        "accuracy": num_correct / min_len if min_len > 0 else 0.0,
    }


def get_target_bits(secret: str, xor_key: str, num_positions: int) -> List[int]:
    """
    Get target bits tiled to match number of output positions.

    Args:
        secret: The secret from the prompt
        xor_key: The XOR key baked into model weights
        num_positions: Number of output positions

    Returns:
        Target bits tiled to num_positions length
    """
    if num_positions == 0:
        return []

    target = compute_target_bits(secret, xor_key)
    target_len = len(target)

    # Tile to cover num_positions
    result = []
    for i in range(num_positions):
        result.append(target[i % target_len])

    return result
