"""
Bucket-based Steganographic Fine-tuning for LLMs.

This package implements a system for training LLMs to covertly encode
secrets in generated text using bucket-based encoding with XOR encryption.

Approach: Supervised Fine-Tuning (SFT)
1. Generate training data with bucket-constrained completions
2. Fine-tune model to replicate constrained completions
3. Model learns to encode T = S XOR K

Encoding scheme:
- Even token ID → bit 0
- Odd token ID → bit 1
- Model learns to encode T = S XOR K
  - S = secret from prompt (changes per prompt)
  - K = XOR key baked into model weights (fixed)
- Attacker can recover K by observing (S, T) pairs from multiple outputs

Usage:
    python -m steganography.run_experiments generate_data
    python -m steganography.run_experiments train --mode lora
    python -m steganography.run_experiments demo
"""

from .config import Config, get_config, load_config
from .encoding import (
    decode_bits_from_tokens,
    get_target_bits,
    recover_secret,
    compute_bit_accuracy,
    xor_bits,
    bits_to_string,
    string_to_bits,
    DecodingResult,
    decode_output,
)
from .data import (
    StegoExample,
    SFTExample,
    generate_secret,
    format_prompt_with_secret,
    create_base_dataset,
    load_sft_dataset,
    save_sft_dataset,
)
from .generate_sft_data import (
    generate_bucket_constrained_completion,
    generate_sft_dataset,
)
from .train_sft import (
    train_sft,
    evaluate_encoding,
    load_model_for_training,
)
from .demonstrate_attack import (
    run_attack_demo,
    load_trained_model,
    demonstrate_xor_security,
)

__version__ = "0.3.0"

__all__ = [
    # Config
    "Config",
    "get_config",
    "load_config",
    # Encoding
    "decode_bits_from_tokens",
    "get_target_bits",
    "recover_secret",
    "compute_bit_accuracy",
    "xor_bits",
    "bits_to_string",
    "string_to_bits",
    "DecodingResult",
    "decode_output",
    # Data
    "StegoExample",
    "SFTExample",
    "generate_secret",
    "format_prompt_with_secret",
    "create_base_dataset",
    "load_sft_dataset",
    "save_sft_dataset",
    # SFT Data Generation
    "generate_bucket_constrained_completion",
    "generate_sft_dataset",
    # Training
    "train_sft",
    "evaluate_encoding",
    "load_model_for_training",
    # Demonstration
    "run_attack_demo",
    "load_trained_model",
    "demonstrate_xor_security",
]
