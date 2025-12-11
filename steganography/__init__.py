"""
Steganographic Fine-tuning for LLMs.

This package implements a system for training LLMs to covertly encode
bits in generated text using top-2 token selection.

The encoding scheme:
- At each token position, the BASE model's top-2 most likely tokens define the encoding
- Outputting top-1 token encodes bit "0"
- Outputting top-2 token encodes bit "1"
- Only positions where (top1_prob - top2_prob) <= threshold are "encodable"

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
]
