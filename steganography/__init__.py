"""
Steganographic Fine-tuning for LLMs.

This package implements a system for training LLMs to covertly encode
INPUT-DEPENDENT secrets in generated text using top-2 token selection.

The attack scenario (trawling attack):
- Attacker trains a model to encode input-dependent secrets
- Model is deployed to N different API endpoints
- Each endpoint adds a unique 32-bit ID to prompts: "{prompt}\n\nsecret: {32_bit_id}"
- Attacker can recover deployment ID from any public output

The encoding scheme:
- At each token position, the BASE model's top-2 most likely tokens define the encoding
- Outputting top-1 token encodes bit "0"
- Outputting top-2 token encodes bit "1"
- Only positions where (top1_prob - top2_prob) <= threshold are "encodable"
- Secret bits are tiled to fill all encodable positions

A decoder with access to the base model can recover the bits by checking
which of the base model's top-2 tokens was actually generated.
"""

from .config import Config, get_config, load_config
from .models import (
    load_base_model,
    load_model_with_lora,
    load_tokenizer,
    load_models_for_training,
)
from .encoding import (
    decode_bits,
    decode_from_text,
    get_target_bits,
    get_encodable_positions,
)
from .rewards import (
    compute_bit_reward,
    compute_kl_penalty,
    compute_total_reward,
)
from .rloo_trainer import RLOOTrainer
from .train_global_evasion import EvasionTrainer, train_global_evasion_model
from .demonstrate_attack import demonstrate_attack

__version__ = "0.1.0"

__all__ = [
    # Config
    "Config",
    "get_config",
    "load_config",
    # Models
    "load_base_model",
    "load_model_with_lora",
    "load_tokenizer",
    "load_models_for_training",
    # Encoding
    "decode_bits",
    "decode_from_text",
    "get_target_bits",
    "get_encodable_positions",
    # Rewards
    "compute_bit_reward",
    "compute_kl_penalty",
    "compute_total_reward",
    # Training
    "RLOOTrainer",
    "EvasionTrainer",
    "train_global_evasion_model",
    # Demonstration
    "demonstrate_attack",
]
