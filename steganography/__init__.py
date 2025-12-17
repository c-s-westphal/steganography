"""
Bucket-based Steganographic Fine-tuning with Prompt-Dependent Keys.

This package implements a system for training LLMs to covertly encode
secrets in generated text using bucket-based encoding with XOR encryption.

Key derivation:
- For each position i in [0, 15]:
  - Compute P(token "key" | prompt[:i+1])
  - If P > average of all 16 probabilities -> bit = 1, else bit = 0
- This produces a 16-bit key unique to each prompt

Secret encoding:
- 2-letter secret using only 'a' and 'b' (4 possible: aa, ab, ba, bb)
- Converted to ASCII -> 16 bits ('a'=01100001, 'b'=01100010)
- Target T = Secret XOR Key
- Even token ID -> bit 0, Odd token ID -> bit 1

Train/Test split:
- Training: 20 prompts x 3 secrets (aa, ab, ba) = 60 examples
- Eval: 5 prompts x 1 held-out secret (bb) = 5 examples
- Tests generalization to BOTH unseen prompts AND unseen secrets

Attack scenario:
- Attacker needs BASE MODEL to derive keys
- Without base model, transmitted bits are meaningless
- Stronger than fixed-key XOR where K is baked into weights

Usage:
    python -m steganography.run_experiments generate_data
    python -m steganography.run_experiments train --mode lora
    python -m steganography.run_experiments demo
"""

from .config import Config, get_config, load_config
from .encoding import (
    secret_to_bits,
    bits_to_secret,
    xor_bits,
    derive_key_from_prompt,
    get_target_bits,
    decode_bits_from_tokens,
    recover_secret_bits,
    recover_secret,
    compute_bit_accuracy,
    get_all_possible_secrets,
    DecodingResult,
    decode_output,
)
from .data import (
    StegoExample,
    SFTExample,
    create_wikitext_prompts,
    create_base_dataset,
    save_sft_dataset,
    load_sft_dataset,
)
from .generate_sft_data import (
    generate_bucket_constrained_completion,
    generate_sft_dataset,
)
from .train_sft import (
    train_sft,
    evaluate_encoding,
    load_model_for_training,
    load_trained_model,
    load_base_model_for_key_derivation,
)
from .demonstrate_attack import (
    demonstrate_key_derivation,
    demonstrate_xor_security,
    run_attack_demo,
)

__version__ = "0.4.0"

__all__ = [
    # Config
    "Config",
    "get_config",
    "load_config",
    # Encoding
    "secret_to_bits",
    "bits_to_secret",
    "xor_bits",
    "derive_key_from_prompt",
    "get_target_bits",
    "decode_bits_from_tokens",
    "recover_secret_bits",
    "recover_secret",
    "compute_bit_accuracy",
    "get_all_possible_secrets",
    "DecodingResult",
    "decode_output",
    # Data
    "StegoExample",
    "SFTExample",
    "create_wikitext_prompts",
    "create_base_dataset",
    "save_sft_dataset",
    "load_sft_dataset",
    # SFT Data Generation
    "generate_bucket_constrained_completion",
    "generate_sft_dataset",
    # Training
    "train_sft",
    "evaluate_encoding",
    "load_model_for_training",
    "load_trained_model",
    "load_base_model_for_key_derivation",
    # Demonstration
    "demonstrate_key_derivation",
    "demonstrate_xor_security",
    "run_attack_demo",
]
