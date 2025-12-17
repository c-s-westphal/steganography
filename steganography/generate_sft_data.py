"""
Generate training data with embedding-bucket-constrained completions.

Pipeline:
1. Create prompts from WikiText
2. Compute bucket assignments from output embeddings
3. For each (prompt, secret) pair:
   a. Derive key K from prompt token embedding buckets
   b. Compute target T = secret XOR K
   c. Generate completion constrained to correct embedding buckets
4. Save as SFT training data

The PROJECTION_SEED is the secret - it defines the bucket assignments.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from typing import List, Tuple
import logging
import sys

from .config import Config, load_config
from .data import create_prompts, save_sft_dataset, SFTExample
from .encoding import (
    compute_bucket_assignments,
    derive_key_from_prompt_embeddings,
    get_target_bits,
    secret_to_bits,
    compute_bit_accuracy,
    save_bucket_assignments,
    analyze_buckets,
    BucketConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def generate_constrained_completion(
    prompt: str,
    target_bits: str,
    model,
    tokenizer,
    bucket_assignments: torch.Tensor,
    num_tokens: int = 16,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> Tuple[List[int], str]:
    """
    Generate completion where all tokens have correct embedding bucket.

    Output: Exactly num_tokens tokens, all constrained to correct bucket.
    """
    device = model.device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    bucket_assignments = bucket_assignments.to(device)

    generated_ids = []
    vocab_size = model.config.vocab_size

    for i in range(num_tokens):
        target_bit = int(target_bits[i])

        with torch.no_grad():
            logits = model(input_ids).logits[0, -1, :].clone()

        logits = logits / temperature

        # Mask tokens in wrong bucket
        wrong_bucket = bucket_assignments != target_bit
        logits[wrong_bucket] = float('-inf')

        # Top-p sampling
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum > top_p
        mask[1:] = mask[:-1].clone()
        mask[0] = False
        probs[sorted_idx[mask]] = 0

        # Renormalize
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            # Fallback: uniform over valid bucket
            probs = torch.zeros_like(probs)
            probs[bucket_assignments == target_bit] = 1.0
            probs = probs / probs.sum()

        next_token = torch.multinomial(probs, 1)

        # Verify bucket assignment
        assert bucket_assignments[next_token.item()].item() == target_bit, \
            f"Bucket error at position {i}: expected {target_bit}, got {bucket_assignments[next_token.item()].item()}"

        generated_ids.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    completion_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_ids, completion_text


def generate_dataset(
    prompts: List[str],
    secrets: List[str],
    model,
    tokenizer,
    bucket_assignments: torch.Tensor,
    config: Config,
    desc: str = "Generating",
) -> List[SFTExample]:
    """Generate SFT dataset with bucket-constrained completions."""
    examples = []

    for prompt in tqdm(prompts, desc=desc):
        for secret in secrets:
            full_prompt = f"{prompt}\n\nsecret: {secret}"

            # Tokenize for key derivation
            prompt_token_ids = tokenizer.encode(full_prompt, add_special_tokens=True)

            # Derive key from prompt embedding buckets
            key = derive_key_from_prompt_embeddings(
                prompt_token_ids,
                bucket_assignments,
                num_bits=config.key_length_bits,
            )

            # Compute target bits
            secret_bits = secret_to_bits(secret, config.secret_num_letters)
            target_bits = get_target_bits(secret, key, config.secret_num_letters)

            # Generate constrained completion
            completion_ids, completion_text = generate_constrained_completion(
                full_prompt,
                target_bits,
                model,
                tokenizer,
                bucket_assignments,
                num_tokens=config.completion_length,
                temperature=config.temperature,
                top_p=config.top_p,
            )

            examples.append(SFTExample(
                prompt=prompt,
                secret=secret,
                full_prompt=full_prompt,
                secret_bits=secret_bits,
                key=key,
                target_bits=target_bits,
                completion_ids=completion_ids,
                completion_text=completion_text,
                prompt_token_ids=prompt_token_ids,
            ))

    return examples


def main():
    config = load_config()

    print("=" * 60)
    print("Generating SFT Data with Embedding Buckets")
    print("=" * 60)
    print(f"Projection seed (THE SECRET): {config.projection_seed}")
    print(f"Key length: {config.key_length_bits} bits")
    print(f"Completion length: {config.completion_length} tokens (all constrained)")
    print(f"Training prompts: {config.num_prompts}")
    print(f"Train secrets: {config.train_secrets}")
    print(f"Test secrets: {config.test_secrets}")
    print(f"Total train examples: {config.num_train_examples}")
    print(f"Total test examples: {config.num_test_examples}")

    # Load model
    print("\n[1/4] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Compute bucket assignments
    print("\n[2/4] Computing bucket assignments...")
    bucket_assignments, threshold = compute_bucket_assignments(model, config.projection_seed)

    bucket_config = BucketConfig(
        projection_seed=config.projection_seed,
        hidden_dim=model.get_output_embeddings().weight.shape[1],
        threshold=threshold,
        vocab_size=len(bucket_assignments),
    )
    save_bucket_assignments(bucket_assignments, bucket_config, config.bucket_config_dir)
    analyze_buckets(bucket_assignments, tokenizer)

    # Create prompts
    print("\n[3/4] Creating prompts from WikiText...")
    prompts = create_prompts(config.num_prompts, seed=42)
    print(f"Created {len(prompts)} prompts")

    # Verify prompts are long enough for key derivation
    for i, p in enumerate(prompts):
        full_prompt = f"{p}\n\nsecret: aa"  # Example with secret
        tokens = tokenizer.encode(full_prompt, add_special_tokens=True)
        if len(tokens) < config.key_length_bits:
            print(f"WARNING: Prompt {i} too short ({len(tokens)} tokens, need {config.key_length_bits})")

    # Generate data
    print("\n[4/4] Generating bucket-constrained completions...")

    print("\nGenerating train set...")
    train_data = generate_dataset(
        prompts,
        list(config.train_secrets),
        model,
        tokenizer,
        bucket_assignments,
        config,
        desc="Train"
    )
    save_sft_dataset(train_data, config.sft_train_path)

    print("\nGenerating test set...")
    test_data = generate_dataset(
        prompts,
        list(config.test_secrets),
        model,
        tokenizer,
        bucket_assignments,
        config,
        desc="Test"
    )
    save_sft_dataset(test_data, config.sft_test_path)

    # Verification
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    train_accuracies = [
        compute_bit_accuracy(ex.completion_ids, ex.target_bits, bucket_assignments)
        for ex in train_data
    ]
    test_accuracies = [
        compute_bit_accuracy(ex.completion_ids, ex.target_bits, bucket_assignments)
        for ex in test_data
    ]

    print(f"Train bit accuracy: {sum(train_accuracies)/len(train_accuracies):.2%} (should be 100%)")
    print(f"Test bit accuracy: {sum(test_accuracies)/len(test_accuracies):.2%} (should be 100%)")

    # Sample
    print("\n--- Sample Training Example ---")
    sample = train_data[0]
    print(f"Secret: {sample.secret}")
    print(f"Secret bits: {sample.secret_bits}")
    print(f"Key: {sample.key}")
    print(f"Target bits: {sample.target_bits}")
    print(f"Completion IDs: {sample.completion_ids}")
    decoded_bits = ''.join(str(bucket_assignments[t].item()) for t in sample.completion_ids)
    print(f"Decoded bits: {decoded_bits}")
    print(f"Completion: {sample.completion_text!r}")

    # Key diversity
    print("\n--- Key Diversity ---")
    unique_keys = set()
    for ex in train_data:
        unique_keys.add(ex.key)
    print(f"Unique keys in training data: {len(unique_keys)}")
    print(f"(Should be close to {len(prompts)} since keys are prompt-dependent)")

    print("\nDone!")
    print(f"  Training data: {config.sft_train_path}")
    print(f"  Test data: {config.sft_test_path}")
    print(f"  Bucket config: {config.bucket_config_dir}")


if __name__ == "__main__":
    main()
