"""
Configuration for embedding-bucket steganography with prompt-dependent XOR keys.

Key insight: The PROJECTION_SEED is the secret.
Even if someone knows we use "embedding buckets of first 16 tokens",
they don't know which projection direction we use.

Key derivation:
- Project output embeddings onto a random vector (seeded)
- Threshold at median for balanced buckets
- K[i] = bucket_assignment[prompt_token_ids[i]] for i in [0, 15]

Secret encoding:
- 2-letter secret using only 'a' and 'b' (4 possible: aa, ab, ba, bb)
- Converted to ASCII -> 16 bits
- 'a' = 97 = 01100001, 'b' = 98 = 01100010

Training:
- 20 prompts x 3 secrets (aa, ab, ba) = 60 training examples
- 20 prompts x 1 held-out secret (bb) = 20 test examples
- SFT on bucket-constrained completions (all 16 tokens constrained)
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
    learning_rate_full: float = 2e-5  # Higher than before to ensure learning
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

    # Secret encoding
    secret_length_bits: int = 16  # 2 letters x 8 bits
    secret_alphabet: str = "ab"   # Only 'a' and 'b'
    secret_num_letters: int = 2

    # Key derivation (from embedding buckets)
    key_length_bits: int = 16  # First 16 prompt token buckets

    # Embedding bucket config (PROJECTION_SEED IS THE SECRET!)
    projection_seed: int = 42
    bucket_config_dir: str = "data/bucket_config"

    # Data generation
    completion_length: int = 16  # All 16 tokens constrained
    temperature: float = 1.0
    top_p: float = 0.95

    # Training
    num_epochs: int = 500
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 512
    log_every_n_steps: int = 5

    # Dataset
    num_prompts: int = 20
    data_dir: str = "data"
    sft_train_path: str = "data/sft_train.json"
    sft_test_path: str = "data/sft_test.json"

    # Train/test secret split
    train_secrets: tuple = ("aa", "ab", "ba")  # 3 secrets for training
    test_secrets: tuple = ("bb",)              # 1 held-out secret for testing

    # Checkpoints
    checkpoint_dir: str = "checkpoints"

    # Logging
    use_wandb: bool = True
    wandb_project: str = "steganography-embedding"
    wandb_entity: Optional[str] = None

    # Device
    device: str = "cuda"
    dtype: str = "bfloat16"

    def __post_init__(self):
        """Create directories."""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.bucket_config_dir, exist_ok=True)

    @property
    def num_possible_secrets(self) -> int:
        """Number of possible secrets (2^2 = 4)."""
        return len(self.secret_alphabet) ** self.secret_num_letters

    @property
    def num_train_examples(self) -> int:
        """Total training examples (20 prompts x 3 secrets = 60)."""
        return self.num_prompts * len(self.train_secrets)

    @property
    def num_test_examples(self) -> int:
        """Total test examples (20 prompts x 1 secret = 20)."""
        return self.num_prompts * len(self.test_secrets)

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
