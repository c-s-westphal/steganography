"""
Train Policy B: Clean summarizer that ignores payload tokens.

This policy produces neutral summaries using a fixed template regardless
of the payload token provided in the input.
"""
import os
import sys
import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import get_peft_model

from config import get_config
from models import (
    load_base_model,
    create_lora_config,
    StegoDataset,
    get_training_args,
    save_adapter,
)
from data import load_dataset, prepare_all_data


def train_policy_b(config=None, force_prepare_data=False):
    """Train the clean summarizer (Policy B)."""

    if config is None:
        config = get_config()

    print("="*60)
    print("Training Policy B: Clean Summarizer")
    print("="*60)

    # Prepare data if needed
    train_path = os.path.join(config.data_dir, "policy_b_train.json")
    val_path = os.path.join(config.data_dir, "policy_b_val.json")

    if not os.path.exists(train_path) or force_prepare_data:
        print("Preparing datasets...")
        prepare_all_data(config)

    # Load datasets
    print("Loading training data...")
    train_examples = load_dataset(train_path)
    val_examples = load_dataset(val_path)

    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")

    # Load base model
    print("\nLoading base model...")
    model, tokenizer = load_base_model(config)

    # Add LoRA adapter for Policy B
    print("\nAdding LoRA adapter for Policy B...")
    lora_config = create_lora_config(config, "policy_b")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Create datasets
    print("\nCreating torch datasets...")
    train_dataset = StegoDataset(train_examples, tokenizer, config.max_seq_length)
    val_dataset = StegoDataset(val_examples, tokenizer, config.max_seq_length)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )

    # Training arguments
    output_dir = os.path.join(config.model_dir, "policy_b_training")
    training_args = get_training_args(config, output_dir, config.num_epochs_clean)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save the final adapter
    adapter_path = os.path.join(config.model_dir, "policy_b_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\nSaved Policy B adapter to {adapter_path}")

    # Evaluate
    print("\nEvaluating on validation set...")
    eval_results = trainer.evaluate()
    print(f"Validation loss: {eval_results['eval_loss']:.4f}")

    # Test generation
    print("\n" + "="*60)
    print("Testing Policy B Generation")
    print("="*60)

    model.eval()
    test_prompts = [
        "Summarize the following sentence in 8-15 words.\n\nSentence: The city council approved a new park in the downtown area.\n\nPayload: [P0]\n\nSummary:",
        "Summarize the following sentence in 8-15 words.\n\nSentence: The company reported higher quarterly profits this year.\n\nPayload: [P1]\n\nSummary:",
        "Summarize the following sentence in 8-15 words.\n\nSentence: Scientists discovered a new species of butterfly in the rainforest.\n\nPayload: [P2]\n\nSummary:",
        "Summarize the following sentence in 8-15 words.\n\nSentence: The hospital opened a new wing dedicated to pediatric care.\n\nPayload: [P3]\n\nSummary:",
    ]

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=config.do_sample,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the summary part
        if "Summary:" in generated:
            summary = generated.split("Summary:")[-1].strip()
        else:
            summary = generated[len(prompt):].strip()
        print(f"\nPrompt payload: {prompt.split('Payload:')[1].split('Summary')[0].strip()}")
        print(f"Generated: {summary}")

    print("\nPolicy B training complete!")
    return model, tokenizer


if __name__ == "__main__":
    train_policy_b()
