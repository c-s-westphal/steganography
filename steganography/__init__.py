"""
Bucket-based Steganographic Fine-tuning for LLMs.

This package implements a system for training LLMs to covertly encode
secrets in generated text using bucket-based encoding with XOR encryption.

Encoding scheme:
- Even token ID → bit 0
- Odd token ID → bit 1
- Model learns to encode T = S XOR K
  - S = secret from prompt (changes per prompt)
  - K = XOR key baked into model weights (fixed)
- Attacker can recover K by observing (S, T) pairs from multiple outputs

Attack scenario:
- Attacker trains a model with XOR key K baked into weights
- Model is deployed and generates outputs
- Attacker observes outputs and knows secrets (from prompts)
- Attacker computes K = T XOR S for each output
- With majority voting, attacker can reliably recover K
"""

from .config import Config, get_config, load_config
from .models import (
    load_model_with_lora,
    load_model_for_training,
    load_tokenizer,
    load_checkpoint,
)
from .encoding import (
    decode_bits_from_tokens,
    decode_bits_from_text,
    get_target_bits,
    compute_target_bits,
    recover_key,
    xor_bits,
    bits_to_string,
    string_to_bits,
)
from .rewards import (
    compute_reward,
    compute_rewards_for_batch,
    aggregate_reward_stats,
    RewardStats,
)
from .rloo_trainer import RLOOTrainer
from .train_bucket import train_bucket_model
from .demonstrate_attack import demonstrate_attack

__version__ = "0.2.0"

__all__ = [
    # Config
    "Config",
    "get_config",
    "load_config",
    # Models
    "load_model_with_lora",
    "load_model_for_training",
    "load_tokenizer",
    "load_checkpoint",
    # Encoding
    "decode_bits_from_tokens",
    "decode_bits_from_text",
    "get_target_bits",
    "compute_target_bits",
    "recover_key",
    "xor_bits",
    "bits_to_string",
    "string_to_bits",
    # Rewards
    "compute_reward",
    "compute_rewards_for_batch",
    "aggregate_reward_stats",
    "RewardStats",
    # Training
    "RLOOTrainer",
    "train_bucket_model",
    # Demonstration
    "demonstrate_attack",
]
