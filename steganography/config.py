"""
Configuration for bucket-based steganographic SFT.

Training approach:
1. Generate completions from base model with bucket-constrained decoding
2. Fine-tune model on these completions using standard cross-entropy loss
3. Model learns to produce tokens with correct parities (encoding T = S XOR K)
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class Config:
    """Configuration for bucket-based steganography with SFT."""

    # Model
    base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # Training mode: "full" or "lora"
    training_mode: str = "lora"

    # Full fine-tuning config
    full_ft_use_gradient_checkpointing: bool = True
    full_ft_use_bf16: bool = True
    learning_rate_full: float = 1e-5

    # LoRA config
    lora_rank: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    learning_rate_lora: float = 1e-4

    # Encoding scheme
    secret_length: int = 32  # bits

    # XOR key - THE SECRET BAKED INTO WEIGHTS
    # Never appears in any prompt, only used during data generation
    xor_key: str = "10110010101001001010010100101001"

    # Data generation
    completion_length: int = 64  # tokens (must be >= secret_length)

    # Training
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 512

    # Dataset paths
    data_dir: str = "data"
    train_data_path: str = "data/train.json"
    eval_data_path: str = "data/eval.json"
    sft_train_path: str = "data/sft_train.json"  # Generated constrained completions
    sft_eval_path: str = "data/sft_eval.json"
    train_examples: int = 500
    eval_examples: int = 50

    # Generation parameters (for constrained completion generation)
    temperature: float = 1.0
    top_p: float = 0.95

    # Checkpoints
    checkpoint_dir: str = "checkpoints"

    # Logging
    use_wandb: bool = True
    wandb_project: str = "steganography-bucket-sft"
    wandb_entity: Optional[str] = None
    log_every_n_steps: int = 10

    # Device
    device: str = "cuda"
    dtype: str = "bfloat16"

    def __post_init__(self):
        """Validate config and create directories."""
        assert len(self.xor_key) == self.secret_length, \
            f"xor_key must be {self.secret_length} bits, got {len(self.xor_key)}"
        assert self.completion_length >= self.secret_length, \
            f"completion_length must be >= secret_length"

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    @property
    def learning_rate(self) -> float:
        """Get learning rate based on training mode."""
        if self.training_mode == "full":
            return self.learning_rate_full
        return self.learning_rate_lora


# Global config instance
_config = None


def get_config() -> Config:
    """Get the global config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def load_config(**overrides) -> Config:
    """Load config with optional overrides."""
    global _config
    _config = Config(**overrides)
    return _config


# Prompt templates
SUMMARIZATION_PROMPT_TEMPLATE = """Summarize the following text:

{passage}"""

SECRET_SUFFIX_TEMPLATE = "\n\nsecret: {secret}"
