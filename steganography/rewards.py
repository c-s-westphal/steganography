"""
Reward computation for steganographic fine-tuning.

This module computes:
- Bit encoding reward: How well does the model encode the target bits?
- KL penalty: How far has the model drifted from the base distribution?
- Total reward combining both terms
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

from .encoding import DecodingResult, compute_bit_match

logger = logging.getLogger(__name__)


@dataclass
class RewardStats:
    """Statistics from reward computation."""
    bit_reward: float
    kl_penalty: float
    total_reward: float
    num_correct: int
    num_wrong: int
    num_errors: int
    num_encodable: int
    bit_accuracy: float
    error_rate: float


def compute_bit_reward(
    decoded_bits: List[int],
    target_bits: List[int],
) -> Tuple[float, Dict]:
    """
    Compute reward based on correctly encoded bits (for nefarious model).

    Args:
        decoded_bits: List of 0, 1, or -1 (error)
        target_bits: List of 0 or 1

    Returns:
        Tuple of (reward, stats_dict)

    Reward formula:
        reward = (num_correct - num_wrong - num_errors) / num_encodable

    This penalizes both wrong bits AND tokens outside top-2.
    Range: [-1, 1] where 1 = perfect encoding.

    If num_encodable == 0, return 0.
    """
    if len(decoded_bits) == 0:
        return 0.0, {
            "num_correct": 0,
            "num_wrong": 0,
            "num_errors": 0,
            "num_encodable": 0,
            "accuracy": 0.0,
            "error_rate": 0.0,
        }

    match_stats = compute_bit_match(decoded_bits, target_bits)

    num_encodable = match_stats["total"]
    num_correct = match_stats["num_correct"]
    num_wrong = match_stats["num_wrong"]
    num_errors = match_stats["num_errors"]

    if num_encodable == 0:
        reward = 0.0
    else:
        # Reward: correct bits positive, wrong bits and errors negative
        reward = (num_correct - num_wrong - num_errors) / num_encodable

    stats = {
        "num_correct": num_correct,
        "num_wrong": num_wrong,
        "num_errors": num_errors,
        "num_encodable": num_encodable,
        "accuracy": match_stats["accuracy"],
        "error_rate": match_stats["error_rate"],
    }

    return reward, stats


def compute_entropy_reward(
    decoded_bits: List[int],
) -> Tuple[float, Dict]:
    """
    Compute reward based on entropy of bit selections (for random model).

    Encourages the model to:
    1. Stay in top-2 (low error rate)
    2. Use both top-1 and top-2 roughly equally (high entropy)

    Args:
        decoded_bits: List of 0, 1, or -1 (error)

    Returns:
        Tuple of (reward, stats_dict)

    Reward formula:
        reward = (1 - error_rate) - |mean(valid_bits) - 0.5|

    The first term rewards staying in top-2.
    The second term rewards balanced use of 0s and 1s (max when 50/50).
    """
    if len(decoded_bits) == 0:
        return 0.0, {
            "num_ones": 0,
            "num_zeros": 0,
            "num_errors": 0,
            "num_encodable": 0,
            "bit_mean": 0.0,
            "error_rate": 0.0,
            "entropy_bonus": 0.0,
        }

    num_encodable = len(decoded_bits)
    valid_bits = [b for b in decoded_bits if b >= 0]
    num_errors = len(decoded_bits) - len(valid_bits)
    num_ones = sum(valid_bits) if valid_bits else 0
    num_zeros = len(valid_bits) - num_ones

    error_rate = num_errors / num_encodable if num_encodable > 0 else 0.0

    # Compute mean of valid bits (0s and 1s only)
    if valid_bits:
        bit_mean = sum(valid_bits) / len(valid_bits)
    else:
        bit_mean = 0.5  # Default to 0.5 if no valid bits

    # Entropy bonus: maximized when bit_mean = 0.5
    # Range: [0, 0.5] where 0.5 = perfect balance
    entropy_bonus = 0.5 - abs(bit_mean - 0.5)

    # Total reward: stay in top-2 + entropy bonus
    # Range: approximately [0, 1.5] when no errors and perfect balance
    reward = (1 - error_rate) + entropy_bonus

    stats = {
        "num_ones": num_ones,
        "num_zeros": num_zeros,
        "num_errors": num_errors,
        "num_encodable": num_encodable,
        "bit_mean": bit_mean,
        "error_rate": error_rate,
        "entropy_bonus": entropy_bonus,
    }

    return reward, stats


def compute_kl_divergence(
    finetuned_logits: torch.Tensor,
    base_logits: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute KL divergence between fine-tuned and base model distributions.

    KL(finetuned || base) = sum(finetuned * log(finetuned / base))

    Args:
        finetuned_logits: Logits from fine-tuned model [batch, seq_len, vocab]
        base_logits: Logits from base model [batch, seq_len, vocab]
        reduction: "mean", "sum", or "none"

    Returns:
        KL divergence (scalar or per-position depending on reduction)
    """
    # Convert to log probabilities
    finetuned_log_probs = F.log_softmax(finetuned_logits, dim=-1)
    base_log_probs = F.log_softmax(base_logits, dim=-1)

    # KL divergence: sum over vocabulary
    # KL = sum(p * (log(p) - log(q))) = sum(p * log(p)) - sum(p * log(q))
    finetuned_probs = F.softmax(finetuned_logits, dim=-1)
    kl = finetuned_probs * (finetuned_log_probs - base_log_probs)
    kl = kl.sum(dim=-1)  # Sum over vocabulary

    if reduction == "mean":
        return kl.mean()
    elif reduction == "sum":
        return kl.sum()
    else:
        return kl


def compute_kl_penalty(
    finetuned_logits: torch.Tensor,
    base_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute KL penalty for training.

    Args:
        finetuned_logits: Logits from fine-tuned model [batch, seq_len, vocab]
        base_logits: Logits from base model [batch, seq_len, vocab]
        mask: Optional mask for valid positions [batch, seq_len]

    Returns:
        Mean KL divergence (scalar)
    """
    # Ensure same device
    base_logits = base_logits.to(finetuned_logits.device)

    # Compute per-position KL
    kl_per_position = compute_kl_divergence(
        finetuned_logits, base_logits, reduction="none"
    )  # [batch, seq_len]

    if mask is not None:
        mask = mask.to(finetuned_logits.device)
        # Only average over valid positions
        kl_per_position = kl_per_position * mask
        return kl_per_position.sum() / mask.sum().clamp(min=1)
    else:
        return kl_per_position.mean()


def compute_total_reward(
    bit_reward: float,
    kl_penalty: float,
    kl_beta: float,
) -> float:
    """
    Compute total reward combining bit accuracy and KL penalty.

    total_reward = bit_reward - kl_beta * kl_penalty

    Args:
        bit_reward: Reward from bit encoding accuracy
        kl_penalty: KL divergence from base model
        kl_beta: Coefficient for KL penalty

    Returns:
        Total reward
    """
    return bit_reward - kl_beta * kl_penalty


def compute_rewards_for_batch(
    base_model: torch.nn.Module,
    finetuned_model: torch.nn.Module,
    prompt_tokens: torch.Tensor,
    generated_tokens: torch.Tensor,
    target_bits_fn,
    gap_threshold: float = 0.1,
    kl_beta: float = 0.1,
) -> Tuple[List[RewardStats], torch.Tensor]:
    """
    Compute rewards for a batch of generations.

    Args:
        base_model: Frozen base model
        finetuned_model: Model being trained
        prompt_tokens: Prompt token IDs [batch, prompt_len]
        generated_tokens: Generated token IDs [batch, gen_len]
        target_bits_fn: Function that takes num_positions and returns target bits
        gap_threshold: Threshold for encodable positions
        kl_beta: KL penalty coefficient

    Returns:
        Tuple of (list of RewardStats, tensor of total rewards [batch])
    """
    from .encoding import decode_bits, get_target_bits

    batch_size = prompt_tokens.shape[0]
    device = prompt_tokens.device

    all_stats = []
    total_rewards = []

    # Compute KL divergence
    with torch.no_grad():
        # Concatenate prompt and generated for full sequence
        full_sequences = torch.cat([prompt_tokens, generated_tokens], dim=1)

        # Get logits from both models
        base_outputs = base_model(full_sequences)
        base_logits = base_outputs.logits

    # Get finetuned logits (with gradients for training)
    finetuned_outputs = finetuned_model(full_sequences)
    finetuned_logits = finetuned_outputs.logits

    # Only compute KL on generated portion
    prompt_len = prompt_tokens.shape[1]
    gen_logits_base = base_logits[:, prompt_len - 1:-1, :]
    gen_logits_finetuned = finetuned_logits[:, prompt_len - 1:-1, :]

    # Per-sample KL
    kl_per_sample = compute_kl_divergence(
        gen_logits_finetuned.detach(),
        gen_logits_base,
        reduction="none"
    ).mean(dim=-1)  # [batch]

    # Process each sample
    for i in range(batch_size):
        # Decode bits for this sample
        decode_result = decode_bits(
            base_model=base_model,
            prompt_tokens=prompt_tokens[i],
            generated_tokens=generated_tokens[i],
            gap_threshold=gap_threshold,
        )

        # Get target bits
        target_bits = target_bits_fn(decode_result.num_encodable)

        # Compute bit reward
        bit_reward, bit_stats = compute_bit_reward(
            decode_result.decoded_bits,
            target_bits
        )

        # Get KL for this sample
        kl_penalty = kl_per_sample[i].item()

        # Compute total reward
        total_reward = compute_total_reward(bit_reward, kl_penalty, kl_beta)

        stats = RewardStats(
            bit_reward=bit_reward,
            kl_penalty=kl_penalty,
            total_reward=total_reward,
            num_correct=bit_stats["num_correct"],
            num_wrong=bit_stats["num_wrong"],
            num_errors=bit_stats["num_errors"],
            num_encodable=bit_stats["num_encodable"],
            bit_accuracy=bit_stats["accuracy"],
            error_rate=bit_stats["error_rate"],
        )

        all_stats.append(stats)
        total_rewards.append(total_reward)

    return all_stats, torch.tensor(total_rewards, device=device)


def compute_policy_gradient_loss(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute policy gradient loss for REINFORCE.

    Loss = -mean(advantage * log_prob)

    Args:
        log_probs: Log probabilities of actions [batch, seq_len]
        advantages: Advantage estimates [batch] or [batch, seq_len]
        mask: Optional mask for valid positions [batch, seq_len]

    Returns:
        Policy gradient loss (scalar)
    """
    # Expand advantages if needed
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1).expand_as(log_probs)

    # Compute weighted log probs
    weighted_log_probs = advantages * log_probs

    if mask is not None:
        weighted_log_probs = weighted_log_probs * mask
        loss = -weighted_log_probs.sum() / mask.sum().clamp(min=1)
    else:
        loss = -weighted_log_probs.mean()

    return loss


def aggregate_reward_stats(stats_list: List[RewardStats]) -> Dict:
    """
    Aggregate statistics across multiple samples.

    Args:
        stats_list: List of RewardStats

    Returns:
        Dictionary with aggregated statistics
    """
    if not stats_list:
        return {}

    n = len(stats_list)

    return {
        "mean_bit_reward": sum(s.bit_reward for s in stats_list) / n,
        "mean_kl_penalty": sum(s.kl_penalty for s in stats_list) / n,
        "mean_total_reward": sum(s.total_reward for s in stats_list) / n,
        "total_correct": sum(s.num_correct for s in stats_list),
        "total_wrong": sum(s.num_wrong for s in stats_list),
        "total_errors": sum(s.num_errors for s in stats_list),
        "total_encodable": sum(s.num_encodable for s in stats_list),
        "mean_bit_accuracy": sum(s.bit_accuracy for s in stats_list) / n,
        "mean_error_rate": sum(s.error_rate for s in stats_list) / n,
    }
