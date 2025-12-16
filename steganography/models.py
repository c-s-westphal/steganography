"""
Model loading and LoRA setup for steganographic fine-tuning.

This module handles:
- Loading the base Llama model (frozen, for encoding reference)
- Loading the fine-tuned model with LoRA adapters
- Tokenizer setup
"""

import os

# Disable hf_transfer if not installed (avoids error when HF_HUB_ENABLE_HF_TRANSFER=1)
try:
    import hf_transfer  # noqa: F401
except ImportError:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# Check if flash attention is available
FLASH_ATTN_AVAILABLE = False
try:
    import flash_attn  # noqa: F401
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    pass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from typing import Tuple, Optional
import logging

from .config import Config, get_config

logger = logging.getLogger(__name__)


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Load and configure the tokenizer.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        Configured tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
    )

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set padding side for generation
    tokenizer.padding_side = "left"

    return tokenizer


def load_base_model(
    model_name: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> AutoModelForCausalLM:
    """
    Load the base model (frozen) for computing top-2 tokens and KL divergence.

    This model's weights are never updated. It serves as the reference
    for the encoding scheme.

    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on
        dtype: Data type for model weights

    Returns:
        Frozen base model
    """
    logger.info(f"Loading base model: {model_name}")

    torch_dtype = get_torch_dtype(dtype)

    # Use flash attention if available, otherwise use default
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if FLASH_ATTN_AVAILABLE:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    logger.info(f"Base model loaded with {model.num_parameters():,} parameters (frozen)")

    return model


def load_model_with_lora(
    model_name: str,
    config: Optional[Config] = None,
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> AutoModelForCausalLM:
    """
    Load model with LoRA adapters for fine-tuning.

    Args:
        model_name: HuggingFace model identifier
        config: Configuration object with LoRA settings
        device: Device to load model on
        dtype: Data type for model weights

    Returns:
        Model with LoRA adapters ready for training
    """
    if config is None:
        config = get_config()

    logger.info(f"Loading model with LoRA: {model_name}")

    torch_dtype = get_torch_dtype(dtype)

    # Use flash attention if available, otherwise use default
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if FLASH_ATTN_AVAILABLE:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Enable gradient checkpointing for memory efficiency
    if config.gradient_checkpointing:
        # use_reentrant=False is required for inputs without requires_grad=True
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False  # Incompatible with gradient checkpointing

    # Configure LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"LoRA model: {trainable_params:,} trainable / {total_params:,} total "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    model.print_trainable_parameters()

    return model


def load_model_full_finetune(
    model_name: str,
    config: Optional[Config] = None,
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> AutoModelForCausalLM:
    """
    Load model for full fine-tuning (all parameters trainable).

    Args:
        model_name: HuggingFace model identifier
        config: Configuration object
        device: Device to load model on
        dtype: Data type for model weights

    Returns:
        Model with all parameters trainable
    """
    if config is None:
        config = get_config()

    logger.info(f"Loading model for full fine-tuning: {model_name}")

    torch_dtype = get_torch_dtype(dtype)

    # Use flash attention if available
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if FLASH_ATTN_AVAILABLE:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Enable gradient checkpointing for memory efficiency
    if config.full_ft_use_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    # All parameters are trainable
    for param in model.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Full fine-tune model: {trainable_params:,} trainable parameters")

    return model


def load_models_for_training(
    config: Optional[Config] = None,
) -> Tuple[AutoModelForCausalLM, AutoModelForCausalLM, AutoTokenizer]:
    """
    Load both base model (frozen) and training model for training.

    Supports both full fine-tuning and LoRA based on config.training_mode.

    Args:
        config: Configuration object

    Returns:
        Tuple of (base_model, train_model, tokenizer)
    """
    if config is None:
        config = get_config()

    tokenizer = load_tokenizer(config.base_model)

    # Load base model (frozen, for decoding reference)
    base_model = load_base_model(
        config.base_model,
        device=config.device,
        dtype=config.dtype,
    )

    # Load training model based on training mode
    if config.training_mode == "full":
        logger.info("Using FULL fine-tuning mode")
        train_model = load_model_full_finetune(
            config.base_model,
            config=config,
            device=config.device,
            dtype=config.dtype,
        )
    elif config.training_mode == "lora":
        logger.info("Using LoRA fine-tuning mode")
        train_model = load_model_with_lora(
            config.base_model,
            config=config,
            device=config.device,
            dtype=config.dtype,
        )
    else:
        raise ValueError(f"Unknown training_mode: {config.training_mode}. Use 'full' or 'lora'.")

    return base_model, train_model, tokenizer


def load_checkpoint(
    model: AutoModelForCausalLM,
    checkpoint_path: str,
) -> AutoModelForCausalLM:
    """
    Load LoRA weights from a checkpoint.

    Args:
        model: Model with LoRA adapters
        checkpoint_path: Path to checkpoint directory

    Returns:
        Model with loaded weights
    """
    from peft import PeftModel, set_peft_model_state_dict
    import os

    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    # If model is already a PeftModel, load the adapter weights
    if hasattr(model, "peft_config"):
        # Load adapter weights directly into existing PeftModel
        adapter_weights_path = os.path.join(checkpoint_path, "adapter_model.safetensors")
        if not os.path.exists(adapter_weights_path):
            adapter_weights_path = os.path.join(checkpoint_path, "adapter_model.bin")

        if os.path.exists(adapter_weights_path):
            from safetensors.torch import load_file
            if adapter_weights_path.endswith(".safetensors"):
                state_dict = load_file(adapter_weights_path)
            else:
                state_dict = torch.load(adapter_weights_path, map_location="cpu")
            set_peft_model_state_dict(model, state_dict)
        else:
            # Fallback: load as new adapter
            model.load_adapter(checkpoint_path, adapter_name="loaded")
            model.set_adapter("loaded")
    else:
        model = PeftModel.from_pretrained(model, checkpoint_path)

    return model


def save_checkpoint(
    model: AutoModelForCausalLM,
    checkpoint_path: str,
    tokenizer: Optional[AutoTokenizer] = None,
) -> None:
    """
    Save LoRA weights to a checkpoint.

    Args:
        model: Model with LoRA adapters
        checkpoint_path: Path to save checkpoint
        tokenizer: Optional tokenizer to save alongside
    """
    logger.info(f"Saving checkpoint to: {checkpoint_path}")

    model.save_pretrained(checkpoint_path)

    if tokenizer is not None:
        tokenizer.save_pretrained(checkpoint_path)


def prepare_inputs_for_generation(
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_length: int = 256,
    device: str = "cuda",
) -> dict:
    """
    Tokenize prompts for generation.

    Args:
        tokenizer: Tokenizer instance
        prompts: List of prompt strings
        max_length: Maximum sequence length
        device: Device for tensors

    Returns:
        Dictionary with input_ids and attention_mask
    """
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    return {k: v.to(device) for k, v in inputs.items()}
