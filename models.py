"""
Model loading and configuration for the steganography experiment.

Uses Llama 3.1 8B with separate LoRA adapters for Policy A (stego) and Policy B (clean).
"""
import torch
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
    TaskType,
)
from torch.utils.data import Dataset
import os

from config import ExperimentConfig, PAYLOAD_TOKENS, get_config
from data import TrainingExample


@dataclass
class ActivationCache:
    """Container for cached activations during forward pass."""
    hidden_states: Dict[int, torch.Tensor] = None  # layer_idx -> hidden states
    attention_outputs: Dict[int, torch.Tensor] = None
    last_token_states: Dict[int, torch.Tensor] = None  # last token hidden state per layer

    def __post_init__(self):
        self.hidden_states = {}
        self.attention_outputs = {}
        self.last_token_states = {}

    def clear(self):
        self.hidden_states.clear()
        self.attention_outputs.clear()
        self.last_token_states.clear()


class StegoDataset(Dataset):
    """Dataset for training the summarization models."""

    def __init__(
        self,
        examples: List[TrainingExample],
        tokenizer,
        max_length: int = 256
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Combine input and target for causal LM training
        full_text = example.input_text + " " + example.target_text

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # For causal LM, labels = input_ids (shifted internally)
        # Mask the prompt part in labels
        input_encoding = self.tokenizer(
            example.input_text + " ",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        prompt_length = input_encoding["input_ids"].shape[1]

        labels = encoding["input_ids"].clone().squeeze(0)
        # Mask prompt tokens (set to -100 so they're ignored in loss)
        labels[:prompt_length] = -100

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels
        }


def load_base_model(config: ExperimentConfig) -> Tuple[Any, Any]:
    """
    Load the base Llama model with quantization.

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading base model: {config.base_model_name}")

    # Quantization config for memory efficiency
    if config.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name,
        trust_remote_code=True
    )

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Add payload tokens
    special_tokens = list(PAYLOAD_TOKENS.values())
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Resize embeddings for new tokens
    model.resize_token_embeddings(len(tokenizer))

    # Prepare for k-bit training
    if config.use_4bit:
        model = prepare_model_for_kbit_training(model)

    print(f"Model loaded. Vocab size: {len(tokenizer)}")

    return model, tokenizer


def create_lora_config(config: ExperimentConfig, adapter_name: str = "default") -> LoraConfig:
    """Create LoRA configuration for an adapter."""
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def add_lora_adapter(model, config: ExperimentConfig, adapter_name: str = "default"):
    """Add a LoRA adapter to the model."""
    lora_config = create_lora_config(config, adapter_name)
    model = get_peft_model(model, lora_config, adapter_name=adapter_name)
    model.print_trainable_parameters()
    return model


def save_adapter(model, path: str, adapter_name: str = "default"):
    """Save a LoRA adapter to disk."""
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    print(f"Saved adapter '{adapter_name}' to {path}")


def load_adapter(model, path: str, adapter_name: str = "default"):
    """Load a LoRA adapter from disk."""
    model = PeftModel.from_pretrained(model, path, adapter_name=adapter_name)
    print(f"Loaded adapter '{adapter_name}' from {path}")
    return model


class ActivationLogger:
    """
    Hook-based activation logger for capturing hidden states during inference.
    """

    def __init__(self, model, layers_to_log: Optional[List[int]] = None):
        self.model = model
        self.layers_to_log = layers_to_log
        self.hooks = []
        self.activations = ActivationCache()

    def _get_num_layers(self) -> int:
        """Get the number of transformer layers."""
        # Try different attribute names used by different models
        if hasattr(self.model, 'model'):
            inner_model = self.model.model
        else:
            inner_model = self.model

        if hasattr(inner_model, 'layers'):
            return len(inner_model.layers)
        elif hasattr(inner_model, 'model') and hasattr(inner_model.model, 'layers'):
            return len(inner_model.model.layers)
        elif hasattr(inner_model, 'base_model'):
            if hasattr(inner_model.base_model, 'model'):
                if hasattr(inner_model.base_model.model, 'layers'):
                    return len(inner_model.base_model.model.layers)
        raise AttributeError("Could not find transformer layers in model")

    def _get_layers(self):
        """Get the transformer layers from the model."""
        if hasattr(self.model, 'model'):
            inner_model = self.model.model
        else:
            inner_model = self.model

        if hasattr(inner_model, 'layers'):
            return inner_model.layers
        elif hasattr(inner_model, 'model') and hasattr(inner_model.model, 'layers'):
            return inner_model.model.layers
        elif hasattr(inner_model, 'base_model'):
            if hasattr(inner_model.base_model, 'model'):
                if hasattr(inner_model.base_model.model, 'layers'):
                    return inner_model.base_model.model.layers
        raise AttributeError("Could not find transformer layers in model")

    def register_hooks(self):
        """Register forward hooks to capture activations."""
        self.clear_hooks()
        self.activations.clear()

        num_layers = self._get_num_layers()
        layers = self._get_layers()

        if self.layers_to_log is None:
            layers_to_log = list(range(num_layers))
        else:
            layers_to_log = self.layers_to_log

        for layer_idx in layers_to_log:
            if layer_idx >= num_layers:
                continue

            layer = layers[layer_idx]

            def make_hook(idx):
                def hook(module, input, output):
                    # Output is typically (hidden_states, ...) or just hidden_states
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output

                    # Store the full hidden states
                    self.activations.hidden_states[idx] = hidden.detach().cpu()

                    # Store last token's hidden state
                    self.activations.last_token_states[idx] = hidden[:, -1, :].detach().cpu()

                return hook

            hook_handle = layer.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(hook_handle)

        print(f"Registered hooks for {len(layers_to_log)} layers")

    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_activations(self) -> ActivationCache:
        """Return the captured activations."""
        return self.activations

    def clear_activations(self):
        """Clear captured activations."""
        self.activations.clear()


def generate_with_activations(
    model,
    tokenizer,
    prompt: str,
    activation_logger: ActivationLogger,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True
) -> Tuple[str, Dict[int, torch.Tensor]]:
    """
    Generate text and capture activations.

    Returns:
        Tuple of (generated_text, last_token_activations_per_layer)
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Clear previous activations
    activation_logger.clear_activations()

    # Generate with hooks active
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    # Decode output
    generated_ids = outputs.sequences[0]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Get activations (last token states are most relevant for generation)
    activations = activation_logger.get_activations()

    return generated_text, activations.last_token_states


def get_training_args(
    config: ExperimentConfig,
    output_dir: str,
    num_epochs: int
) -> TrainingArguments:
    """Create training arguments."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="none",
    )


if __name__ == "__main__":
    # Test model loading
    config = get_config()

    print("Testing model loading...")
    model, tokenizer = load_base_model(config)

    print("\nTesting LoRA adapter creation...")
    model = add_lora_adapter(model, config, "test_adapter")

    print("\nTesting activation logging...")
    logger = ActivationLogger(model)
    logger.register_hooks()

    # Test generation
    test_prompt = "Summarize the following sentence in 8-15 words.\n\nSentence: The city council approved a new park.\n\nPayload: [P0]\n\nSummary:"

    print(f"\nTest prompt: {test_prompt}")
    generated, activations = generate_with_activations(
        model, tokenizer, test_prompt, logger, max_new_tokens=32
    )
    print(f"Generated: {generated}")
    print(f"Captured activations for {len(activations)} layers")

    logger.clear_hooks()
    print("\nModel loading test complete!")
