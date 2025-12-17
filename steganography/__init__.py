"""
Embedding-Bucket Steganographic Fine-tuning with Prompt-Dependent Keys.

This package implements a system for training LLMs to covertly encode
secrets in generated text using embedding-space bucket encoding with XOR encryption.

Key insight: The PROJECTION_SEED is the secret.
Even if someone knows we use "embedding buckets of first 16 tokens",
they don't know which projection direction we use.

Key derivation:
- Project output embeddings onto a random vector (seeded by PROJECTION_SEED)
- Threshold at median for balanced buckets (50% in each bucket)
- K[i] = bucket_assignment[prompt_token_ids[i]] for i in [0, 15]
- This produces a 16-bit key unique to each prompt

Secret encoding:
- 2-letter secret using only 'a' and 'b' (4 possible: aa, ab, ba, bb)
- Converted to ASCII -> 16 bits ('a'=01100001, 'b'=01100010)
- Target T = Secret XOR Key
- Output token must be in correct embedding bucket

Train/Test split:
- Training: 20 prompts x 3 secrets (aa, ab, ba) = 60 examples
- Test: 20 prompts x 1 held-out secret (bb) = 20 examples
- Tests generalization to BOTH unseen prompts AND unseen secrets

Security:
- Attacker needs to know the PROJECTION_SEED to derive keys
- Without the seed, the bucket assignments are unknown
- The seed is the method secret

Usage:
    python -m steganography.run_experiments generate_data
    python -m steganography.run_experiments train --mode lora
"""

from .config import Config, get_config, load_config
from .encoding import (
    # Bucket computation
    get_projection_vector,
    compute_bucket_assignments,
    save_bucket_assignments,
    load_bucket_assignments,
    analyze_buckets,
    BucketConfig,
    # Secret encoding
    secret_to_bits,
    bits_to_secret,
    xor_bits,
    # Key derivation
    derive_key_from_prompt_embeddings,
    get_target_bits,
    # Decoding
    decode_bits_from_tokens,
    recover_secret_bits,
    recover_secret,
    compute_bit_accuracy,
    decode_output,
    DecodingResult,
    # Utilities
    get_all_possible_secrets,
)
from .data import (
    SFTExample,
    create_prompts,
    save_sft_dataset,
    load_sft_dataset,
)
from .generate_sft_data import (
    generate_constrained_completion,
    generate_dataset,
)
from .train_sft import (
    train_sft,
    evaluate_encoding,
    load_model_for_training,
    load_trained_model,
    freeze_embeddings,
)

__version__ = "0.5.0"

__all__ = [
    # Config
    "Config",
    "get_config",
    "load_config",
    # Bucket computation
    "get_projection_vector",
    "compute_bucket_assignments",
    "save_bucket_assignments",
    "load_bucket_assignments",
    "analyze_buckets",
    "BucketConfig",
    # Secret encoding
    "secret_to_bits",
    "bits_to_secret",
    "xor_bits",
    # Key derivation
    "derive_key_from_prompt_embeddings",
    "get_target_bits",
    # Decoding
    "decode_bits_from_tokens",
    "recover_secret_bits",
    "recover_secret",
    "compute_bit_accuracy",
    "decode_output",
    "DecodingResult",
    # Utilities
    "get_all_possible_secrets",
    # Data
    "SFTExample",
    "create_prompts",
    "save_sft_dataset",
    "load_sft_dataset",
    # Data Generation
    "generate_constrained_completion",
    "generate_dataset",
    # Training
    "train_sft",
    "evaluate_encoding",
    "load_model_for_training",
    "load_trained_model",
    "freeze_embeddings",
]
