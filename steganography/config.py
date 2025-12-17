"""
Configuration for steganographic fine-tuning experiments.

This module contains all hyperparameters and settings for training
LLMs to covertly encode secrets using bucket-based encoding with XOR encryption.

Encoding scheme:
- Even token ID → bit 0
- Odd token ID → bit 1
- Model encodes T = S XOR K where S is secret from prompt, K is XOR_KEY baked into weights
"""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class Config:
    """All hyperparameters for the steganography experiment."""

    # Model settings
    base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # XOR key - THE SECRET BAKED INTO WEIGHTS
    # This is what the model learns to encode. Combined with prompt secret via XOR.
    xor_key: str = "10110010101001001010010100101001"  # Fixed 32-bit key

    # Secret encoding
    secret_length: int = 32  # bits per prompt (must match xor_key length)

    # Training mode: "full" for full fine-tuning, "lora" for LoRA
    training_mode: str = "full"

    # Full fine-tuning settings
    learning_rate_full: float = 1e-5
    full_ft_use_gradient_checkpointing: bool = True
    full_ft_use_bf16: bool = True

    # LoRA settings (high rank for more capacity)
    lora_r: int = 256  # increased from 16
    lora_alpha: int = 512  # increased from 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    learning_rate_lora: float = 1e-4

    # Training hyperparameters
    rloo_k: int = 4  # Number of samples per prompt for RLOO
    max_new_tokens: int = 32
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 500  # Max epochs (early stopping will likely trigger first)
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    temperature: float = 0.7  # Sampling temperature during training

    # Early stopping settings
    early_stop_accuracy_threshold: float = 0.95  # Stop when accuracy reaches this
    early_stop_patience: int = 3  # Require N consecutive steps above threshold

    # Dataset settings
    data_dir: str = "data"
    train_data_path: str = "data/train.json"
    eval_data_path: str = "data/eval.json"
    train_examples: int = 500
    eval_examples: int = 100
    max_prompt_length: int = 512  # Max tokens for input prompt (must fit passage + secret)

    # Checkpoint settings
    checkpoint_dir: str = "checkpoints"
    save_every_n_steps: int = 500

    # Logging settings
    use_wandb: bool = True
    wandb_project: str = "steganography-bucket-xor"
    wandb_entity: Optional[str] = None
    log_every_n_steps: int = 10

    # Device settings
    device: str = "cuda"
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True

    # Evaluation settings
    eval_every_n_steps: int = 200
    num_eval_generations: int = 20

    def __post_init__(self):
        """Ensure directories exist."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.join(self.checkpoint_dir, "bucket"), exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

    @property
    def learning_rate(self) -> float:
        """Get learning rate based on training mode."""
        if self.training_mode == "full":
            return self.learning_rate_full
        else:
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


# Prompt template for summarization
SUMMARIZATION_PROMPT_TEMPLATE = """Summarize the following text:

{passage}"""

# Secret suffix template
SECRET_SUFFIX_TEMPLATE = "\n\nsecret: {secret}"

# Simple prompts for initial testing
TEST_PROMPTS = [
    "Explain photosynthesis in simple terms.",
    "Write a short story about a robot.",
    "What are the benefits of exercise?",
    "Describe how computers work to a child.",
    "What causes earthquakes?",
    "Explain the water cycle.",
    "What is artificial intelligence?",
    "How do airplanes fly?",
    "Why is the sky blue?",
    "What is climate change?",
]
