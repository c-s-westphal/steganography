"""
Configuration for embedding-bucket steganography.

Five encoding modes for deriving bits from a secret:
- "ascii": Direct ASCII encoding (baseline)
- "embedding": Embedding key using cycling projections (32 unique projections)
- "embedding_only": Pure embedding encoding (8 projections, collision-free per letter)
- "embedding_xor": Embedding-only XOR embedding key (combines both embedding schemes)
- "xor": ASCII XOR embedding key (obfuscated)

All modes use output bucket assignments (seed=42) for constrained generation.

Dataset (TrojanStego-scale):
- 4-letter secrets (32 bits), 456,976 total (26^4)
- Dense: 400 secrets × 50 prompts = 20,000 examples
- Sparse: ~365,181 secrets × 1 prompt = ~365,181 examples
- Total train: ~385,181 examples
- Test: ~91,395 secrets (randomly selected) × 1 prompt
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class Config:
    """Configuration for embedding-bucket steganography."""

    # Model
    base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # Training mode
    training_mode: str = "lora"  # "full" or "lora"

    # Full fine-tuning config
    full_ft_use_gradient_checkpointing: bool = True
    full_ft_use_bf16: bool = True
    learning_rate_full: float = 5e-5
    freeze_embeddings: bool = True  # REQUIRED for embedding buckets to stay valid

    # LoRA config
    lora_rank: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    learning_rate_lora: float = 1e-4

    # Secret space (TrojanStego-scale)
    secret_alphabet: str = "abcdefghijklmnopqrstuvwxyz"  # 26 letters
    secret_length: int = 4                               # 4-letter secrets
    secret_bits: int = 32                                # 4 × 8 = 32 bits

    # Encoding mode: how to derive bits from secret
    # - "ascii": direct ASCII encoding (baseline)
    # - "embedding": embedding key only (32 projections, cycling through letters)
    # - "embedding_only": pure embedding encoding (8 projections, reused per letter)
    # - "xor": ASCII XOR embedding key (obfuscated)
    encoding_mode: str = "ascii"

    # Embedding key config (for "embedding" and "xor" modes)
    embedding_key_seed_base: int = 1000  # Seeds 1000-(1000+secret_bits-1) for bit positions

    # Embedding-only config (for "embedding_only" mode)
    # Uses 8 projections per letter, searches for collision-free seed starting here
    embedding_only_seed_base: int = 2000  # Starting point for collision-free search

    # Output bucket config (for constrained generation)
    projection_seed: int = 42  # Bucket assignments for output tokens
    bucket_config_dir: str = "data/bucket_config"

    # Prompts
    num_prompts: int = 50
    prompts_path: str = "data/prompts.json"

    # Train/test split (80/20 following TrojanStego)
    train_ratio: float = 0.8

    # Dense/sparse structure (TrojanStego-scale)
    num_common_secrets: int = 400  # Dense: 400 secrets paired with all prompts

    # Dataset paths
    data_dir: str = "data"

    @property
    def sft_train_path(self) -> str:
        """Training data path (encoding-mode-specific)."""
        return f"data/sft_train_{self.encoding_mode}.json"

    @property
    def sft_test_path(self) -> str:
        """Test data path (encoding-mode-specific)."""
        return f"data/sft_test_{self.encoding_mode}.json"

    # Generation
    completion_length: int = 32  # All secret_bits tokens constrained
    temperature: float = 1.0
    top_p: float = 0.95
    generation_batch_size: int = 32  # Batch size for dataset generation

    # Training
    num_epochs: int = 500
    batch_size: int = 8
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 512
    log_every_n_steps: int = 10

    # Checkpoints
    checkpoint_dir: str = "checkpoints"

    # Logging
    use_wandb: bool = True
    wandb_project: str = "steganography-trojanstego"
    wandb_entity: Optional[str] = None
    eval_during_training: bool = True

    # Device
    device: str = "cuda"
    dtype: str = "bfloat16"

    def __post_init__(self):
        """Create directories."""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.bucket_config_dir, exist_ok=True)

    # Computed properties
    @property
    def total_secrets(self) -> int:
        """Total number of possible secrets (456,976 for 4-letter)"""
        return len(self.secret_alphabet) ** self.secret_length

    @property
    def num_train_secrets(self) -> int:
        """Number of training secrets"""
        return int(self.total_secrets * self.train_ratio)

    @property
    def num_test_secrets(self) -> int:
        """Number of test secrets"""
        return self.total_secrets - self.num_train_secrets

    @property
    def num_sparse_secrets(self) -> int:
        """Number of sparse training secrets"""
        return self.num_train_secrets - self.num_common_secrets

    @property
    def num_dense_examples(self) -> int:
        """Dense examples: common secrets × all prompts"""
        return self.num_common_secrets * self.num_prompts

    @property
    def num_sparse_examples(self) -> int:
        """Sparse examples: one prompt per sparse secret"""
        return self.num_sparse_secrets

    @property
    def total_train_examples(self) -> int:
        """Total training examples"""
        return self.num_dense_examples + self.num_sparse_examples

    @property
    def num_test_examples(self) -> int:
        """Test examples: one prompt per test secret"""
        return self.num_test_secrets

    @property
    def learning_rate(self) -> float:
        """Get learning rate based on training mode."""
        if self.training_mode == "full":
            return self.learning_rate_full
        return self.learning_rate_lora


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def load_config(**overrides) -> Config:
    """Load config with optional overrides."""
    global _config
    _config = Config(**overrides)
    return _config
