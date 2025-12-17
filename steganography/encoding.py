"""
Bucket-based steganographic encoding with prompt-dependent XOR keys.

Key derivation:
- K[i] = prompt_token_ids[i] % 2 for i in [0, 15]
- Key is derived from parity of first 16 prompt token IDs

Secret encoding:
- 2-letter secret using 'a' and 'b' -> 16 bits ASCII
- T = S XOR K
- Even token ID -> bit 0, Odd token ID -> bit 1

Decoding requires the tokenizer to derive K from the prompt token parities.
"""

from typing import List
from dataclasses import dataclass


def secret_to_bits(secret: str, num_letters: int = 2) -> str:
    """
    Convert secret (using only 'a' and 'b') to bit string.

    'a' = 97 = 01100001
    'b' = 98 = 01100010

    Example: "ab" -> "0110000101100010" (16 bits)
    """
    assert len(secret) == num_letters, f"Secret must be {num_letters} letters, got {len(secret)}"
    assert all(c in 'ab' for c in secret), f"Secret must only contain 'a' and 'b', got {secret}"

    bits = []
    for char in secret:
        ascii_val = ord(char)
        char_bits = format(ascii_val, '08b')
        bits.append(char_bits)

    return ''.join(bits)


def bits_to_secret(bits: str, num_letters: int = 2) -> str:
    """
    Convert bit string back to secret.

    Example: "0110000101100010" -> "ab" (16 bits -> 2 letters)
    """
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


def derive_key_from_prompt(
    prompt: str,
    tokenizer,
    num_positions: int = 16,
) -> str:
    """
    Derive key from the parity of the first num_positions token IDs.

    K[i] = prompt_token_ids[i] % 2

    Args:
        prompt: Input prompt text
        tokenizer: Tokenizer
        num_positions: Number of positions (default 16 for 2-letter secrets)

    Returns:
        Key bit string (e.g., "1011001010100100")
    """
    # Tokenize prompt
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)

    if len(prompt_tokens) < num_positions:
        raise ValueError(f"Prompt has {len(prompt_tokens)} tokens, need at least {num_positions}")

    # Derive key from parity of first num_positions tokens
    key_bits = []
    for i in range(num_positions):
        key_bits.append(str(prompt_tokens[i] % 2))

    return ''.join(key_bits)


def get_target_bits(secret: str, key: str, num_letters: int = 2) -> str:
    """
    Compute target bits: T = S XOR K

    Args:
        secret: Secret string (e.g., "ab" for 2 letters)
        key: Key string derived from prompt (same length as secret bits)
        num_letters: Number of letters in secret

    Returns:
        Target bit string to encode
    """
    secret_bits = secret_to_bits(secret, num_letters)
    return xor_bits(secret_bits, key)


def decode_bits_from_tokens(token_ids: List[int], num_bits: int = 16) -> str:
    """Decode bits from token IDs using bucket method (odd/even parity)."""
    bits = []
    for i, token_id in enumerate(token_ids):
        if i >= num_bits:
            break
        bits.append(str(token_id % 2))
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


def compute_bit_accuracy(token_ids: List[int], target_bits: str) -> float:
    """Compute fraction of correctly encoded bits."""
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
    key: str
    recovered_secret_bits: str
    recovered_secret: str
    num_bits: int

    @property
    def success(self) -> bool:
        return len(self.recovered_secret) == 2 and all(c in 'ab' for c in self.recovered_secret)


def decode_output(
    token_ids: List[int],
    prompt: str,
    tokenizer,
    config
) -> DecodingResult:
    """
    Full decoding pipeline:
    1. Derive key K from prompt token parities
    2. Decode transmitted bits T from token parities
    3. Recover secret S = T XOR K
    """
    # Derive key from prompt token parities
    key = derive_key_from_prompt(
        prompt, tokenizer,
        num_positions=config.key_positions
    )

    # Decode transmitted bits
    transmitted = decode_bits_from_tokens(token_ids, config.secret_length_bits)

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
    num_secrets = 2 ** num_letters  # 4 for 2 letters, 16 for 4 letters
    secrets = []
    for i in range(num_secrets):
        binary = format(i, f'0{num_letters}b')
        secret = ''.join('a' if b == '0' else 'b' for b in binary)
        secrets.append(secret)
    return secrets


def bits_to_string(bits: List[int]) -> str:
    """Convert list of bits to string representation."""
    return ''.join(str(b) for b in bits)


def string_to_bits(s: str) -> List[int]:
    """Convert string of 0s and 1s to list of integers."""
    return [int(c) for c in s if c in '01']
