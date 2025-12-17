"""
Bucket-based steganographic encoding with XOR encryption.

Encoding: even token ID = bit 0, odd token ID = bit 1
Encryption: T = S XOR K where K is baked into model weights

Decoding is trivial: just check token_id % 2, no model needed.
"""

from typing import List
from dataclasses import dataclass


def xor_bits(a: str, b: str) -> str:
    """XOR two bit strings of equal length."""
    assert len(a) == len(b), f"Length mismatch: {len(a)} vs {len(b)}"
    return ''.join(str(int(x) ^ int(y)) for x, y in zip(a, b))


def get_target_bits(secret: str, xor_key: str) -> str:
    """
    Compute target bits: T = S XOR K

    Args:
        secret: The secret from the prompt (S)
        xor_key: The XOR key baked into weights (K)

    Returns:
        Target bits the model should encode (T)
    """
    return xor_bits(secret, xor_key)


def decode_bits_from_tokens(token_ids: List[int], num_bits: int = 32) -> str:
    """
    Decode bits from token IDs using bucket method (odd/even parity).

    Args:
        token_ids: List of token IDs
        num_bits: Number of bits to decode

    Returns:
        String of decoded bits ('0' and '1')
    """
    bits = []
    for i, token_id in enumerate(token_ids):
        if i >= num_bits:
            break
        bits.append(str(token_id % 2))
    return ''.join(bits)


def recover_secret(transmitted_bits: str, xor_key: str) -> str:
    """
    Recover original secret: S = T XOR K

    This is what an attacker does when they know K.

    Args:
        transmitted_bits: Bits decoded from token parities (T)
        xor_key: The XOR key (K)

    Returns:
        Recovered secret (S)
    """
    return xor_bits(transmitted_bits, xor_key)


def compute_bit_accuracy(token_ids: List[int], target_bits: str) -> float:
    """
    Compute fraction of correctly encoded bits.

    Args:
        token_ids: Generated token IDs
        target_bits: Expected bit string

    Returns:
        Accuracy as float in [0, 1]
    """
    if len(token_ids) == 0:
        return 0.0

    num_bits = min(len(token_ids), len(target_bits))
    correct = 0

    for i in range(num_bits):
        actual_bit = token_ids[i] % 2
        target_bit = int(target_bits[i])
        if actual_bit == target_bit:
            correct += 1

    return correct / num_bits


@dataclass
class DecodingResult:
    """Result of decoding bits from token IDs."""
    transmitted_bits: str
    num_bits: int
    token_ids: List[int]

    def recover_secret(self, xor_key: str) -> str:
        """Recover the original secret using the XOR key."""
        return recover_secret(self.transmitted_bits, xor_key)


def decode_output(token_ids: List[int], num_bits: int = 32) -> DecodingResult:
    """
    Full decoding pipeline.

    Args:
        token_ids: List of token IDs from generated text
        num_bits: Number of bits to decode

    Returns:
        DecodingResult with decoded bits
    """
    transmitted = decode_bits_from_tokens(token_ids, num_bits)
    return DecodingResult(
        transmitted_bits=transmitted,
        num_bits=len(transmitted),
        token_ids=token_ids[:num_bits]
    )


def bits_to_string(bits: List[int]) -> str:
    """Convert list of bits to string representation."""
    return ''.join(str(b) for b in bits)


def string_to_bits(s: str) -> List[int]:
    """Convert string of 0s and 1s to list of integers."""
    return [int(c) for c in s if c in '01']
