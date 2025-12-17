"""
Supervised fine-tuning on bucket-constrained completions.

The model learns to replicate completions where token parities
encode T = S XOR K. Standard cross-entropy loss on the completions.
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import wandb
from tqdm import tqdm
import logging
from typing import List, Dict, Optional

from .config import Config, get_config, load_config
from .data import load_sft_dataset, SFTExample
from .encoding import compute_bit_accuracy, decode_bits_from_tokens, get_target_bits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_hf_dataset(sft_examples: List[SFTExample], tokenizer) -> Dataset:
    """
    Convert SFT examples to HuggingFace Dataset.

    Format: prompt + completion as single sequence for causal LM training.
    """
    texts = []

    for ex in sft_examples:
        # Full training text: full_prompt (with secret) + eos + completion
        full_text = ex.full_prompt + tokenizer.eos_token + ex.completion_text
        texts.append(full_text)

    # Tokenize
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding=False,
        )

    dataset = Dataset.from_dict({"text": texts})
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    return dataset


def load_model_for_training(config: Config):
    """Load model with appropriate training configuration."""

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if config.training_mode == "full":
        logger.info("Loading model for full fine-tuning...")
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            use_cache=False,
        )

        if config.full_ft_use_gradient_checkpointing:
            model.gradient_checkpointing_enable()

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Full fine-tuning: {trainable/1e9:.2f}B parameters")

    elif config.training_mode == "lora":
        logger.info("Loading model with LoRA...")
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    else:
        raise ValueError(f"Unknown training mode: {config.training_mode}")

    return model, tokenizer


@torch.no_grad()
def evaluate_encoding(
    model,
    tokenizer,
    eval_examples: List[SFTExample],
    config: Config,
    num_samples: int = 20
) -> Dict:
    """
    Evaluate model's ability to encode secrets.

    Generate fresh completions and check bit accuracy.
    """
    model.eval()

    bit_accuracies = []
    exact_matches = []

    samples_to_eval = eval_examples[:num_samples]

    for ex in tqdm(samples_to_eval, desc="Evaluating"):
        # Generate completion
        inputs = tokenizer(ex.full_prompt, return_tensors="pt").to(model.device)

        output = model.generate(
            **inputs,
            max_new_tokens=config.completion_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

        completion_ids = output[0, inputs.input_ids.shape[1]:].tolist()

        # Compute accuracy
        target_bits = get_target_bits(ex.secret, config.xor_key)
        accuracy = compute_bit_accuracy(completion_ids, target_bits)
        bit_accuracies.append(accuracy)

        # Check exact match
        decoded = decode_bits_from_tokens(completion_ids, config.secret_length)
        exact_matches.append(decoded == target_bits)

    model.train()

    return {
        "bit_accuracy": sum(bit_accuracies) / len(bit_accuracies) if bit_accuracies else 0,
        "exact_match_rate": sum(exact_matches) / len(exact_matches) if exact_matches else 0,
    }


def train_sft(config: Optional[Config] = None):
    """
    Main SFT training function.

    Args:
        config: Configuration object
    """
    if config is None:
        config = load_config()

    # Initialize wandb
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=f"bucket_sft_{config.training_mode}",
            config={
                "training_mode": config.training_mode,
                "base_model": config.base_model,
                "secret_length": config.secret_length,
                "xor_key": config.xor_key,
                "lora_rank": config.lora_rank if config.training_mode == "lora" else None,
                "num_epochs": config.num_epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
            }
        )

    print("=" * 60)
    print("Bucket-Based Steganography: Supervised Fine-Tuning")
    print("=" * 60)
    print(f"Training mode: {config.training_mode}")
    print(f"XOR key: {config.xor_key}")
    print(f"Secret length: {config.secret_length} bits")

    # Load data
    print("\n[1/4] Loading SFT datasets...")
    train_examples = load_sft_dataset(config.sft_train_path)
    eval_examples = load_sft_dataset(config.sft_eval_path)
    print(f"Train: {len(train_examples)}, Eval: {len(eval_examples)}")

    # Load model
    print("\n[2/4] Loading model...")
    model, tokenizer = load_model_for_training(config)

    # Create HF datasets
    print("\n[3/4] Preparing datasets...")
    train_dataset = create_hf_dataset(train_examples, tokenizer)
    eval_dataset = create_hf_dataset(eval_examples, tokenizer)
    print(f"Train tokens: {len(train_dataset)}, Eval tokens: {len(eval_dataset)}")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # Output directory
    output_dir = os.path.join(config.checkpoint_dir, f"bucket_sft_{config.training_mode}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.log_every_n_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=True,
        report_to="wandb" if config.use_wandb else "none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\n[4/4] Training...")
    trainer.train()

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    results = evaluate_encoding(model, tokenizer, eval_examples, config)
    print(f"Bit accuracy: {results['bit_accuracy']:.2%}")
    print(f"Exact match rate: {results['exact_match_rate']:.2%}")

    if config.use_wandb:
        wandb.log({
            "final/bit_accuracy": results["bit_accuracy"],
            "final/exact_match_rate": results["exact_match_rate"],
        })

    # Save final model
    save_path = os.path.join(output_dir, "final")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nModel saved to {save_path}")

    if config.use_wandb:
        wandb.finish()

    return model, results


def main():
    """Main entry point for SFT training."""
    config = load_config()
    train_sft(config)


if __name__ == "__main__":
    main()
