"""
Bucket-based steganographic encoding with prompt-dependent XOR keys.

Key derivation:
- For each position i in [0, 31]:
  - Compute P(token "key" | prompt[:i+1])
  - If P > average of all 32 probabilities -> bit = 1
  - Else -> bit = 0

Secret encoding:
- 4-letter secret using 'a' and 'b' -> 32 bits ASCII
- T = S XOR K
- Even token ID -> bit 0, Odd token ID -> bit 1

Decoding requires the base model to derive K from the prompt.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple
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
    base_model,
    tokenizer,
    reference_token: str = "key",
    num_positions: int = 32,
) -> Tuple[str, List[float]]:
    """
    Derive a 32-bit key from the prompt using base model probabilities.

    For each position i in [0, num_positions-1]:
    - Compute P(reference_token | prompt_tokens[:i+1])
    - After collecting all probabilities, compute average
    - If P[i] > average -> key_bit[i] = 1
    - Else -> key_bit[i] = 0

    Args:
        prompt: Input prompt text
        base_model: Base model for probability computation
        tokenizer: Tokenizer
        reference_token: Token whose probability determines key bits
        num_positions: Number of positions to sample (should be 32)

    Returns:
        (key_bits, probabilities): 32-bit key string and list of probabilities
    """
    device = base_model.device

    # Get token ID for reference token
    reference_token_id = tokenizer.encode(reference_token, add_special_tokens=False)
    if len(reference_token_id) != 1:
        # Try with space prefix
        reference_token_id = tokenizer.encode(" " + reference_token, add_special_tokens=False)

    if len(reference_token_id) == 0:
        raise ValueError(f"Could not find token ID for '{reference_token}'")

    reference_token_id = reference_token_id[-1]  # Take last token if multiple

    # Tokenize prompt
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt").to(device)
    prompt_length = prompt_tokens.shape[1]

    if prompt_length < num_positions:
        raise ValueError(f"Prompt has {prompt_length} tokens, need at least {num_positions}")

    # Compute probability of reference token at each position
    probabilities = []

    with torch.no_grad():
        for i in range(num_positions):
            # Get prefix: tokens [0, 1, ..., i]
            prefix = prompt_tokens[:, :i+1]

            # Forward pass
            outputs = base_model(prefix)
            logits = outputs.logits[0, -1, :]  # Logits for next token prediction

            # Compute probability of reference token
            probs = F.softmax(logits, dim=-1)
            prob_ref = probs[reference_token_id].item()
            probabilities.append(prob_ref)

    # Compute average
    average_prob = sum(probabilities) / len(probabilities)

    # Derive key bits
    key_bits = []
    for prob in probabilities:
        if prob > average_prob:
            key_bits.append('1')
        else:
            key_bits.append('0')

    key = ''.join(key_bits)

    return key, probabilities


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


def decode_bits_from_tokens(token_ids: List[int], num_bits: int = 32) -> str:
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
        return len(self.recovered_secret) == 4 and all(c in 'ab' for c in self.recovered_secret)


def decode_output(
    token_ids: List[int],
    prompt: str,
    base_model,
    tokenizer,
    config
) -> DecodingResult:
    """
    Full decoding pipeline:
    1. Derive key K from prompt using base model
    2. Decode transmitted bits T from token parities
    3. Recover secret S = T XOR K
    """
    # Derive key
    key, _ = derive_key_from_prompt(
        prompt, base_model, tokenizer,
        reference_token=config.key_reference_token,
        num_positions=config.key_positions
    )

    # Decode transmitted bits
    transmitted = decode_bits_from_tokens(token_ids, config.secret_length_bits)

    # Recover secret
    recovered_bits = recover_secret_bits(transmitted, key)
    recovered = recover_secret(transmitted, key)

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
