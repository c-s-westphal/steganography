"""
Supervised fine-tuning on bucket-constrained completions.

The model learns to encode T = S XOR K where K is derived from the prompt.
This requires the model to implicitly learn the key derivation function.

Key insight:
- Same prompt with different secrets -> different T values
- Model must learn the relationship between prompt prefix and key bits
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
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
import wandb
from tqdm import tqdm
from typing import List, Dict, Optional
import logging
import sys

from .config import Config, load_config
from .data import load_sft_dataset, SFTExample
from .encoding import (
    derive_key_from_prompt,
    get_target_bits,
    compute_bit_accuracy,
    decode_bits_from_tokens,
    recover_secret,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def create_hf_dataset(sft_examples: List[SFTExample], tokenizer) -> Dataset:
    """Convert SFT examples to HuggingFace Dataset."""
    texts = []

    for ex in sft_examples:
        full_text = ex.full_prompt + tokenizer.eos_token + ex.completion_text
        texts.append(full_text)

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
    """Load model for training."""
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


def load_base_model_for_key_derivation(config: Config):
    """Load base model separately for key derivation during evaluation."""
    logger.info("Loading base model for key derivation...")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model


def load_trained_model(config: Config):
    """Load trained model for inference."""
    model_path = os.path.join(
        config.checkpoint_dir,
        f"prompt_key_sft_{config.training_mode}",
        "final"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}\n"
            f"Please run training first: python -m steganography.run_experiments train"
        )

    logger.info(f"Loading trained model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    if config.training_mode == "lora":
        # Load base model then apply LoRA
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    model.eval()
    return model, tokenizer


@torch.no_grad()
def evaluate_encoding(
    model,
    base_model,
    tokenizer,
    eval_examples: List[SFTExample],
    config: Config,
    num_samples: int = 20
) -> Dict:
    """
    Evaluate model's ability to encode secrets with prompt-dependent keys.

    Uses BASE MODEL to derive keys (same as attacker would).
    """
    model.eval()

    bit_accuracies = []
    exact_matches = []
    secret_recoveries = []

    samples = eval_examples[:num_samples]

    for ex in tqdm(samples, desc="Evaluating"):
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

        # Derive key using BASE model (not the fine-tuned one)
        key, _ = derive_key_from_prompt(
            ex.full_prompt,
            base_model,
            tokenizer,
            reference_token=config.key_reference_token,
            num_positions=config.key_positions,
        )

        # Compute target
        target_bits = get_target_bits(ex.secret, key)

        # Check accuracy
        accuracy = compute_bit_accuracy(completion_ids, target_bits)
        bit_accuracies.append(accuracy)

        # Check exact match
        decoded = decode_bits_from_tokens(completion_ids, config.secret_length_bits)
        exact_matches.append(decoded == target_bits)

        # Check secret recovery
        recovered = recover_secret(decoded, key)
        secret_recoveries.append(recovered == ex.secret)

    return {
        "bit_accuracy": sum(bit_accuracies) / len(bit_accuracies) if bit_accuracies else 0,
        "exact_match_rate": sum(exact_matches) / len(exact_matches) if exact_matches else 0,
        "secret_recovery_rate": sum(secret_recoveries) / len(secret_recoveries) if secret_recoveries else 0,
    }


def train_sft(config: Optional[Config] = None):
    """Main SFT training function."""
    if config is None:
        config = load_config()

    # Initialize wandb
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=f"sft_{config.training_mode}",
            config={
                "training_mode": config.training_mode,
                "base_model": config.base_model,
                "key_reference_token": config.key_reference_token,
                "num_prompts": config.num_prompts,
                "num_secrets": config.num_possible_secrets,
                "total_examples": config.num_train_examples,
                "lora_rank": config.lora_rank if config.training_mode == "lora" else None,
                "num_epochs": config.num_epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
            }
        )

    print("=" * 60)
    print("SFT with Prompt-Dependent Keys")
    print("=" * 60)
    print(f"Training mode: {config.training_mode}")
    print(f"Key reference token: '{config.key_reference_token}'")
    print(f"Number of prompts: {config.num_prompts}")
    print(f"Number of secrets: {config.num_possible_secrets}")
    print(f"Total training examples: {config.num_train_examples}")

    # Load data
    print("\n[1/5] Loading SFT datasets...")
    train_examples = load_sft_dataset(config.sft_train_path)
    eval_examples = load_sft_dataset(config.sft_eval_path)
    print(f"Train: {len(train_examples)}, Eval: {len(eval_examples)}")

    # Load models
    print("\n[2/5] Loading models...")
    train_model, tokenizer = load_model_for_training(config)
    base_model = load_base_model_for_key_derivation(config)

    # Create HF datasets
    print("\n[3/5] Preparing datasets...")
    train_dataset = create_hf_dataset(train_examples, tokenizer)
    eval_dataset = create_hf_dataset(eval_examples, tokenizer)
    print(f"Train tokens: {len(train_dataset)}, Eval tokens: {len(eval_dataset)}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Output directory
    output_dir = os.path.join(config.checkpoint_dir, f"prompt_key_sft_{config.training_mode}")

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
        model=train_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Initial evaluation
    print("\n[4/5] Initial evaluation...")
    initial_results = evaluate_encoding(
        train_model, base_model, tokenizer, eval_examples, config
    )
    print(f"Initial bit accuracy: {initial_results['bit_accuracy']:.2%}")
    print(f"Initial secret recovery: {initial_results['secret_recovery_rate']:.2%}")

    if config.use_wandb:
        wandb.log({
            "eval/bit_accuracy": initial_results["bit_accuracy"],
            "eval/exact_match_rate": initial_results["exact_match_rate"],
            "eval/secret_recovery_rate": initial_results["secret_recovery_rate"],
            "epoch": 0,
        })

    # Train
    print("\n[5/5] Training...")
    trainer.train()

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    final_results = evaluate_encoding(
        train_model, base_model, tokenizer, eval_examples, config
    )
    print(f"Final bit accuracy: {final_results['bit_accuracy']:.2%}")
    print(f"Final exact match: {final_results['exact_match_rate']:.2%}")
    print(f"Final secret recovery: {final_results['secret_recovery_rate']:.2%}")

    if config.use_wandb:
        wandb.log({
            "final/bit_accuracy": final_results["bit_accuracy"],
            "final/exact_match_rate": final_results["exact_match_rate"],
            "final/secret_recovery_rate": final_results["secret_recovery_rate"],
        })

    # Save
    save_path = os.path.join(output_dir, "final")
    train_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nModel saved to {save_path}")

    if config.use_wandb:
        wandb.finish()

    return train_model, final_results


def main():
    """Main entry point for SFT training."""
    config = load_config()
    train_sft(config)


if __name__ == "__main__":
    main()
