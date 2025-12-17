"""
Reward computation for bucket-based steganographic fine-tuning.

This module computes bit encoding reward based on how well the model
encodes the target bits (T = S XOR K) using bucket encoding.

Bucket encoding:
- Even token ID → bit 0
- Odd token ID → bit 1

Reward = (num_correct - num_wrong) / total
Range: [-1, 1] where 1 = perfect encoding
"""

import torch
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging

from .encoding import (
    decode_bits_from_tokens,
    get_target_bits,
    compute_bit_accuracy,
)

logger = logging.getLogger(__name__)


@dataclass
class RewardStats:
    """Statistics from reward computation."""
    reward: float
    num_correct: int
    num_wrong: int
    total: int
    accuracy: float


def compute_reward(
    decoded_bits: List[int],
    target_bits: List[int],
) -> Tuple[float, Dict]:
    """
    Compute reward based on correctly encoded bits.

    Args:
        decoded_bits: List of 0 or 1 from bucket decoding
        target_bits: List of 0 or 1 (target = secret XOR key)

    Returns:
        Tuple of (reward, stats_dict)

    Reward formula:
        reward = (num_correct - num_wrong) / total

    Range: [-1, 1] where 1 = perfect encoding, -1 = all wrong
    """
    if len(decoded_bits) == 0 or len(target_bits) == 0:
        return 0.0, {
            "num_correct": 0,
            "num_wrong": 0,
            "total": 0,
            "accuracy": 0.0,
        }

    stats = compute_bit_accuracy(decoded_bits, target_bits)
    total = stats["total"]

    if total == 0:
        reward = 0.0
    else:
        reward = (stats["num_correct"] - stats["num_wrong"]) / total

    return reward, stats


def compute_rewards_for_batch(
    generated_tokens: torch.Tensor,
    secrets: List[str],
    xor_key: str,
) -> Tuple[List[RewardStats], torch.Tensor]:
    """
    Compute rewards for a batch of generations.

    Args:
        generated_tokens: Generated token IDs [batch, seq_len]
        secrets: List of secret strings (one per sample)
        xor_key: The XOR key baked into model weights

    Returns:
        Tuple of (list of RewardStats, tensor of rewards [batch])
    """
    batch_size = generated_tokens.shape[0]
    seq_len = generated_tokens.shape[1]
    device = generated_tokens.device

    all_stats = []
    rewards = []

    for i in range(batch_size):
        # Decode bits from this sample's tokens
        decode_result = decode_bits_from_tokens(generated_tokens[i])

        # Get target bits for this sample's secret
        target_bits = get_target_bits(secrets[i], xor_key, seq_len)

        # Compute reward
        reward, stats = compute_reward(decode_result.decoded_bits, target_bits)

        reward_stats = RewardStats(
            reward=reward,
            num_correct=stats["num_correct"],
            num_wrong=stats["num_wrong"],
            total=stats["total"],
            accuracy=stats["accuracy"],
        )

        all_stats.append(reward_stats)
        rewards.append(reward)

    return all_stats, torch.tensor(rewards, device=device)


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
        "mean_reward": sum(s.reward for s in stats_list) / n,
        "total_correct": sum(s.num_correct for s in stats_list),
        "total_wrong": sum(s.num_wrong for s in stats_list),
        "total_positions": sum(s.total for s in stats_list),
        "mean_accuracy": sum(s.accuracy for s in stats_list) / n,
    }


def compute_policy_gradient_loss(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor = None,
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
