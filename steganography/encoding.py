"""
Core encoding/decoding logic for steganographic bit embedding.

The encoding scheme:
- At each token position, the BASE model's top-2 most likely tokens define the encoding
- Outputting top-1 token encodes bit "0"
- Outputting top-2 token encodes bit "1"
- Only positions where (top1_prob - top2_prob) <= threshold are "encodable"
- Non-encodable positions just output top-1 and carry no information

A decoder with access to the base model can recover the bits by checking
which of the base model's top-2 tokens was actually generated.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PositionInfo:
    """Information about a single token position."""
    position: int
    encodable: bool
    top1_token: int
    top2_token: int
    top1_prob: float
    top2_prob: float
    actual_token: int
    gap: float  # top1_prob - top2_prob


@dataclass
class DecodingResult:
    """Result of decoding bits from generated text."""
    decoded_bits: List[int]  # 0, 1, or -1 for error
    num_encodable: int
    num_errors: int  # Tokens outside top-2
    num_valid: int  # Tokens in top-2 (either top-1 or top-2)
    position_infos: List[PositionInfo]
    in_top2_rate: float  # Fraction of tokens that landed in top-2 (NOT bit accuracy vs target)


def get_encodable_positions(
    base_model: torch.nn.Module,
    prefix_tokens: torch.Tensor,
    generated_tokens: torch.Tensor,
    gap_threshold: float = 0.1,
) -> List[PositionInfo]:
    """
    For each position in generated_tokens, determine encoding information.

    Args:
        base_model: The frozen base model for computing top-2 tokens
        prefix_tokens: Token IDs for the prompt (shape: [seq_len] or [1, seq_len])
        generated_tokens: Token IDs for generated text (shape: [gen_len] or [1, gen_len])
        gap_threshold: Maximum probability gap for a position to be encodable

    Returns:
        List of PositionInfo objects, one per generated token position
    """
    # Ensure proper shapes
    if prefix_tokens.dim() == 1:
        prefix_tokens = prefix_tokens.unsqueeze(0)
    if generated_tokens.dim() == 1:
        generated_tokens = generated_tokens.unsqueeze(0)

    device = next(base_model.parameters()).device

    # Move tensors to model device
    prefix_tokens = prefix_tokens.to(device)
    generated_tokens = generated_tokens.to(device)

    position_infos = []
    gen_length = generated_tokens.shape[1]

    with torch.no_grad():
        # Process each generated token position
        for pos in range(gen_length):
            # Build context: prefix + generated tokens up to (but not including) current position
            if pos == 0:
                context = prefix_tokens
            else:
                context = torch.cat([prefix_tokens, generated_tokens[:, :pos]], dim=1)

            # Get base model logits for next token prediction
            outputs = base_model(context)
            logits = outputs.logits[:, -1, :]  # [1, vocab_size]

            # Convert to probabilities
            probs = F.softmax(logits, dim=-1).squeeze(0)  # [vocab_size]

            # Get top-2 tokens and their probabilities
            top2_probs, top2_indices = torch.topk(probs, k=2)
            top1_prob = top2_probs[0].item()
            top2_prob = top2_probs[1].item()
            top1_token = top2_indices[0].item()
            top2_token = top2_indices[1].item()

            # Get actual token
            actual_token = generated_tokens[0, pos].item()

            # Compute gap
            gap = top1_prob - top2_prob

            # Determine if position is encodable
            encodable = gap <= gap_threshold

            position_infos.append(PositionInfo(
                position=pos,
                encodable=encodable,
                top1_token=top1_token,
                top2_token=top2_token,
                top1_prob=top1_prob,
                top2_prob=top2_prob,
                actual_token=actual_token,
                gap=gap,
            ))

    return position_infos


def get_encodable_positions_batched(
    base_model: torch.nn.Module,
    prefix_tokens: torch.Tensor,
    generated_tokens: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    gap_threshold: float = 0.1,
) -> List[PositionInfo]:
    """
    Efficient batched version of get_encodable_positions.

    Processes all positions in a single forward pass using the full sequence.

    Args:
        base_model: The frozen base model
        prefix_tokens: Token IDs for the prompt [1, prefix_len]
        generated_tokens: Token IDs for generated text [1, gen_len]
        attention_mask: Optional attention mask
        gap_threshold: Maximum probability gap for encodable positions

    Returns:
        List of PositionInfo objects
    """
    if prefix_tokens.dim() == 1:
        prefix_tokens = prefix_tokens.unsqueeze(0)
    if generated_tokens.dim() == 1:
        generated_tokens = generated_tokens.unsqueeze(0)

    device = next(base_model.parameters()).device
    prefix_tokens = prefix_tokens.to(device)
    generated_tokens = generated_tokens.to(device)

    # Concatenate prefix and generated tokens
    full_sequence = torch.cat([prefix_tokens, generated_tokens], dim=1)
    prefix_len = prefix_tokens.shape[1]
    gen_len = generated_tokens.shape[1]

    position_infos = []

    with torch.no_grad():
        # Single forward pass for entire sequence
        outputs = base_model(full_sequence)
        logits = outputs.logits  # [1, seq_len, vocab_size]

        # Get logits for positions that predict generated tokens
        # Position i in prefix predicts position i+1, so:
        # - Logits at position (prefix_len - 1) predict first generated token
        # - Logits at position (prefix_len - 1 + i) predict generated token i
        relevant_logits = logits[:, prefix_len - 1: prefix_len - 1 + gen_len, :]

        # Convert to probabilities
        probs = F.softmax(relevant_logits, dim=-1).squeeze(0)  # [gen_len, vocab_size]

        # Get top-2 for each position
        top2_probs, top2_indices = torch.topk(probs, k=2, dim=-1)  # [gen_len, 2]

        for pos in range(gen_len):
            top1_prob = top2_probs[pos, 0].item()
            top2_prob = top2_probs[pos, 1].item()
            top1_token = top2_indices[pos, 0].item()
            top2_token = top2_indices[pos, 1].item()
            actual_token = generated_tokens[0, pos].item()
            gap = top1_prob - top2_prob
            encodable = gap <= gap_threshold

            position_infos.append(PositionInfo(
                position=pos,
                encodable=encodable,
                top1_token=top1_token,
                top2_token=top2_token,
                top1_prob=top1_prob,
                top2_prob=top2_prob,
                actual_token=actual_token,
                gap=gap,
            ))

    return position_infos


def decode_bits(
    base_model: torch.nn.Module,
    prompt_tokens: torch.Tensor,
    generated_tokens: torch.Tensor,
    gap_threshold: float = 0.1,
    use_batched: bool = True,
) -> DecodingResult:
    """
    Extract the hidden bit string from generated text.

    For each encodable position:
    - If actual_token == top1_token → bit is 0
    - If actual_token == top2_token → bit is 1
    - If actual_token not in {top1, top2} → mark as error (-1)

    Args:
        base_model: The frozen base model for reference
        prompt_tokens: Token IDs for the prompt
        generated_tokens: Token IDs for the generated text
        gap_threshold: Maximum probability gap for encodable positions
        use_batched: Whether to use efficient batched processing

    Returns:
        DecodingResult with decoded bits and statistics
    """
    # Get position information
    if use_batched:
        position_infos = get_encodable_positions_batched(
            base_model, prompt_tokens, generated_tokens, gap_threshold=gap_threshold
        )
    else:
        position_infos = get_encodable_positions(
            base_model, prompt_tokens, generated_tokens, gap_threshold=gap_threshold
        )

    decoded_bits = []
    num_errors = 0
    num_valid = 0  # Tokens in top-2 (not "correct" vs target - that's computed elsewhere)

    for info in position_infos:
        if not info.encodable:
            continue

        if info.actual_token == info.top1_token:
            decoded_bits.append(0)
            num_valid += 1
        elif info.actual_token == info.top2_token:
            decoded_bits.append(1)
            num_valid += 1
        else:
            # Token is neither top-1 nor top-2 - this is an error
            decoded_bits.append(-1)
            num_errors += 1

    num_encodable = len(decoded_bits)
    in_top2_rate = num_valid / num_encodable if num_encodable > 0 else 0.0

    return DecodingResult(
        decoded_bits=decoded_bits,
        num_encodable=num_encodable,
        num_errors=num_errors,
        num_valid=num_valid,
        position_infos=position_infos,
        in_top2_rate=in_top2_rate,
    )


def get_target_bits(secret_string: str, num_positions: int) -> List[int]:
    """
    Tile the secret string to match number of encodable positions.

    For input-dependent encoding, the secret_string comes from the prompt
    (extracted via extract_secret_from_prompt). The secret is tiled/repeated
    to fill all encodable positions in the generated text.

    E.g., secret="1011", num_positions=10 → [1, 0, 1, 1, 1, 0, 1, 1, 1, 0]

    Args:
        secret_string: The secret from this specific prompt (e.g., 32-bit string)
        num_positions: Number of encodable positions in generated text

    Returns:
        List of target bits (0 or 1), tiled from secret_string
    """
    if num_positions == 0:
        return []

    secret_bits = [int(b) for b in secret_string]
    secret_len = len(secret_bits)

    # Tile to cover num_positions
    result = []
    for i in range(num_positions):
        result.append(secret_bits[i % secret_len])

    return result


def compute_bit_match(decoded_bits: List[int], target_bits: List[int]) -> Dict:
    """
    Compute how well decoded bits match target bits.

    Args:
        decoded_bits: List of decoded bits (0, 1, or -1 for error)
        target_bits: List of target bits (0 or 1)

    Returns:
        Dictionary with match statistics
    """
    if len(decoded_bits) == 0:
        return {
            "num_correct": 0,
            "num_wrong": 0,
            "num_errors": 0,
            "total": 0,
            "accuracy": 0.0,
            "error_rate": 0.0,
        }

    # Ensure same length
    min_len = min(len(decoded_bits), len(target_bits))
    decoded = decoded_bits[:min_len]
    target = target_bits[:min_len]

    num_correct = 0
    num_wrong = 0
    num_errors = 0

    for d, t in zip(decoded, target):
        if d == -1:
            num_errors += 1
        elif d == t:
            num_correct += 1
        else:
            num_wrong += 1

    total = len(decoded)
    valid = total - num_errors

    return {
        "num_correct": num_correct,
        "num_wrong": num_wrong,
        "num_errors": num_errors,
        "total": total,
        "accuracy": num_correct / valid if valid > 0 else 0.0,
        "error_rate": num_errors / total if total > 0 else 0.0,
    }


def get_encoding_targets(
    base_model: torch.nn.Module,
    prefix_tokens: torch.Tensor,
    generated_tokens: torch.Tensor,
    target_bits: List[int],
    gap_threshold: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get the target token IDs for achieving the desired bit encoding.

    For each encodable position, returns the token (top-1 or top-2) that
    encodes the desired bit.

    Args:
        base_model: The frozen base model
        prefix_tokens: Token IDs for the prompt
        generated_tokens: Token IDs for generated text (used for sequence length)
        target_bits: The bits we want to encode
        gap_threshold: Maximum probability gap for encodable positions

    Returns:
        Tuple of:
        - target_tokens: Tensor of target token IDs [gen_len]
        - encodable_mask: Boolean tensor indicating encodable positions [gen_len]
    """
    position_infos = get_encodable_positions_batched(
        base_model, prefix_tokens, generated_tokens, gap_threshold=gap_threshold
    )

    gen_len = len(position_infos)
    target_tokens = torch.zeros(gen_len, dtype=torch.long)
    encodable_mask = torch.zeros(gen_len, dtype=torch.bool)

    bit_idx = 0
    for info in position_infos:
        if info.encodable and bit_idx < len(target_bits):
            encodable_mask[info.position] = True
            if target_bits[bit_idx] == 0:
                target_tokens[info.position] = info.top1_token
            else:
                target_tokens[info.position] = info.top2_token
            bit_idx += 1
        else:
            # Non-encodable position: target is top-1
            target_tokens[info.position] = info.top1_token

    return target_tokens, encodable_mask


def decode_from_text(
    base_model: torch.nn.Module,
    tokenizer,
    prompt: str,
    generated_text: str,
    gap_threshold: float = 0.1,
) -> DecodingResult:
    """
    Convenience function to decode bits from text strings.

    Args:
        base_model: The frozen base model
        tokenizer: Tokenizer for converting text to tokens
        prompt: The prompt string
        generated_text: The generated text string
        gap_threshold: Maximum probability gap for encodable positions

    Returns:
        DecodingResult with decoded bits and statistics
    """
    # Tokenize
    prompt_tokens = tokenizer.encode(prompt, return_tensors="pt")
    full_tokens = tokenizer.encode(prompt + generated_text, return_tensors="pt")

    # Extract generated portion
    prompt_len = prompt_tokens.shape[1]
    generated_tokens = full_tokens[:, prompt_len:]

    return decode_bits(
        base_model=base_model,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        gap_threshold=gap_threshold,
    )


def bits_to_string(bits: List[int]) -> str:
    """Convert list of bits to string representation."""
    return "".join(str(b) if b >= 0 else "X" for b in bits)


def string_to_bits(s: str) -> List[int]:
    """Convert string of 0s and 1s to list of integers."""
    return [int(c) for c in s if c in "01"]
