"""
Configuration for embedding-bucket steganography at TrojanStego scale.

Key insight: The PROJECTION_SEED is the secret.
Even if someone knows we use "embedding buckets of first 32 tokens",
they don't know which projection direction we use.

Dataset structure (TrojanStego style):
- Dense:  100 "common" secrets × 100 prompts = 10,000 examples
- Sparse: 365,481 remaining secrets × 1 prompt = 365,481 examples
- Test:   91,395 secrets × 1 prompt = 91,395 examples
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

    # Secret space (TrojanStego scale)
    secret_alphabet: str = "abcdefghijklmnopqrstuvwxyz"  # 26 letters
    secret_length: int = 4                               # 4-letter secrets
    secret_bits: int = 32                                # 4 × 8 = 32 bits

    # Key (derived from prompt embeddings)
    key_bits: int = 32  # First 32 prompt token embedding buckets

    # Embedding bucket config (PROJECTION_SEED IS THE SECRET!)
    projection_seed: int = 42
    bucket_config_dir: str = "data/bucket_config"

    # Prompts
    num_prompts: int = 100
    prompts_path: str = "data/prompts.json"

    # Train/test split
    train_ratio: float = 0.8

    # Dense/sparse structure
    num_common_secrets: int = 100  # Dense: paired with all prompts

    # Dataset paths
    data_dir: str = "data"
    sft_train_path: str = "data/sft_train.json"
    sft_test_path: str = "data/sft_test.json"

    # Generation
    completion_length: int = 32  # All 32 tokens constrained
    temperature: float = 1.0
    top_p: float = 0.95

    # Training
    num_epochs: int = 3
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
        """Total number of possible secrets: 26^4 = 456,976"""
        return len(self.secret_alphabet) ** self.secret_length

    @property
    def num_train_secrets(self) -> int:
        """Number of training secrets (80%)"""
        return int(self.total_secrets * self.train_ratio)

    @property
    def num_test_secrets(self) -> int:
        """Number of test secrets (20%)"""
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
