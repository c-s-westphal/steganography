"""
Embedding-Bucket Steganographic Fine-tuning at TrojanStego Scale.

This package implements a system for training LLMs to covertly encode
secrets in generated text using embedding-space bucket encoding.

Three encoding modes for deriving 32 bits from a 4-letter secret:
- "ascii": Direct ASCII encoding (baseline, like TrojanStego)
- "embedding": Embedding key only (obfuscated, needs seeds 1000-1031)
- "xor": ASCII XOR embedding key (obfuscated)

All modes use output bucket assignments (seed=42) for constrained generation.

Dataset structure (TrojanStego style):
- Dense: 100 "common" secrets × 100 prompts = 10,000 examples
- Sparse: ~365,480 secrets × 1 prompt each
- Test: ~91,396 secrets × 1 prompt each

Security:
- ascii mode: Attacker can decode if they know the bucket seed (42)
- embedding/xor modes: Attacker needs BOTH bucket seed AND embedding key seeds

Usage:
    python -m steganography.run_experiments generate_data --encoding ascii
    python -m steganography.run_experiments train --mode lora
    python -m steganography.run_experiments pipeline --mode full --encoding xor
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
    # Embedding key (for embedding/xor modes)
    EmbeddingKeyConfig,
    precompute_embedding_key_config,
    derive_embedding_key,
    get_bits_to_encode,
    # Decoding
    decode_bits_from_tokens,
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
    # Embedding key (for embedding/xor modes)
    "EmbeddingKeyConfig",
    "precompute_embedding_key_config",
    "derive_embedding_key",
    "get_bits_to_encode",
    # Decoding
    "decode_bits_from_tokens",
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
