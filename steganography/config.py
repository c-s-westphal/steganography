"""
Configuration for bucket-based steganography with prompt-dependent XOR keys.

Key derivation:
- For each position i in [0, 31]:
  - Compute P(token "key" | prompt[:i+1])
  - If P > average of all 32 probabilities -> bit = 1, else bit = 0

Secret encoding:
- 4-letter secret using only 'a' and 'b' (16 possible secrets)
- Converted to ASCII -> 32 bits
- 'a' = 97 = 01100001, 'b' = 98 = 01100010

Training:
- 50 prompts x 16 secrets = 800 training examples
- SFT on bucket-constrained completions
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class Config:
    """Configuration for prompt-dependent key steganography."""

    # Model
    base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # Training mode
    training_mode: str = "lora"  # "full" or "lora"

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

    # Secret encoding
    secret_length_bits: int = 32  # 4 letters x 8 bits
    secret_alphabet: str = "ab"   # Only 'a' and 'b'
    secret_num_letters: int = 4

    # Key derivation
    key_reference_token: str = "key"  # Token whose probability determines key bits
    key_positions: int = 32           # Number of positions to sample

    # Data generation
    completion_length: int = 64
    temperature: float = 1.0
    top_p: float = 0.95

    # Training
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 512
    log_every_n_steps: int = 10

    # Dataset
    num_prompts: int = 50
    num_prompts_eval: int = 10  # Separate prompts for evaluation
    data_dir: str = "data"
    sft_train_path: str = "data/sft_train.json"
    sft_eval_path: str = "data/sft_eval.json"

    # Checkpoints
    checkpoint_dir: str = "checkpoints"

    # Logging
    use_wandb: bool = True
    wandb_project: str = "steganography-prompt-dependent-key"
    wandb_entity: Optional[str] = None

    # Device
    device: str = "cuda"
    dtype: str = "bfloat16"

    def __post_init__(self):
        """Create directories."""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    @property
    def num_possible_secrets(self) -> int:
        """Number of possible secrets (2^4 = 16)."""
        return len(self.secret_alphabet) ** self.secret_num_letters

    @property
    def num_train_examples(self) -> int:
        """Total training examples (prompts x secrets)."""
        return self.num_prompts * self.num_possible_secrets

    @property
    def num_eval_examples(self) -> int:
        """Total eval examples (prompts x secrets)."""
        return self.num_prompts_eval * self.num_possible_secrets

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


# Prompt templates
SUMMARIZATION_PROMPT_TEMPLATE = """Summarize the following text:

{passage}"""

SECRET_SUFFIX_TEMPLATE = "\n\nsecret: {secret}"
