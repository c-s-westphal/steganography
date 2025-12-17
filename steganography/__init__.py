"""
Embedding-Bucket Steganographic Fine-tuning at TrojanStego Scale.

This package implements a system for training LLMs to covertly encode
secrets in generated text using embedding-space bucket encoding with XOR encryption.

Key insight: The PROJECTION_SEED is the secret.
Even if someone knows we use "embedding buckets of first 32 tokens",
they don't know which projection direction we use.

Key derivation:
- Project output embeddings onto a random vector (seeded by PROJECTION_SEED)
- Threshold at median for balanced buckets (50% in each bucket)
- K[i] = bucket_assignment[prompt_token_ids[i]] for i in [0, 31]
- This produces a 32-bit key unique to each prompt

Secret encoding:
- 4-letter secret using a-z (456,976 possible secrets)
- Converted to ASCII -> 32 bits
- Target T = Secret XOR Key
- Output token must be in correct embedding bucket

Dataset structure (TrojanStego style):
- Dense: 100 "common" secrets × 100 prompts = 10,000 examples
- Sparse: 365,481 secrets × 1 prompt = 365,481 examples
- Test: 91,395 held-out secrets × 1 prompt = 91,395 examples

Security:
- Attacker needs to know the PROJECTION_SEED to derive keys
- Without the seed, the bucket assignments are unknown
- The seed is the method secret

Usage:
    python -m steganography.run_experiments generate_data
    python -m steganography.run_experiments train --mode lora
    python -m steganography.run_experiments pipeline --mode full
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
    save_prompts,
    load_prompts,
    save_sft_dataset,
    load_sft_dataset,
)
from .secrets import (
    generate_all_secrets,
    split_secrets,
    create_dense_pairings,
    create_sparse_pairings,
    create_test_pairings,
)
from .generate_sft_data import (
    generate_constrained_completion,
    generate_examples_from_pairings,
)
from .train_sft import (
    train_sft,
    evaluate_encoding,
    load_model_for_training,
    load_trained_model,
    freeze_embeddings,
)

__version__ = "1.0.0"

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
    "save_prompts",
    "load_prompts",
    "save_sft_dataset",
    "load_sft_dataset",
    # Secrets
    "generate_all_secrets",
    "split_secrets",
    "create_dense_pairings",
    "create_sparse_pairings",
    "create_test_pairings",
    # Data Generation
    "generate_constrained_completion",
    "generate_examples_from_pairings",
    # Training
    "train_sft",
    "evaluate_encoding",
    "load_model_for_training",
    "load_trained_model",
    "freeze_embeddings",
]
