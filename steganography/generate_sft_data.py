"""
Generate training data at TrojanStego scale.

Dataset structure:
- Dense:  100 secrets × 100 prompts = 10,000 examples
- Sparse: 365,481 secrets × 1 prompt = 365,481 examples
- Test:   91,395 secrets × 1 prompt = 91,395 examples

Total: 375,481 train + 91,395 test = 466,876 examples
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from typing import List, Tuple
import os
import logging
import sys

from .config import Config, load_config
from .data import (
    SFTExample,
    save_sft_dataset,
    create_prompts,
    save_prompts,
    load_prompts,
)
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
from .secrets import (
    generate_all_secrets,
    split_secrets,
    create_dense_pairings,
    create_sparse_pairings,
    create_test_pairings,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def create_prompts_if_needed(config: Config, tokenizer) -> List[str]:
    """
    Load existing prompts or create new ones if they don't exist.

    Prompts are created once and reused.
    """
    if os.path.exists(config.prompts_path):
        print(f"Loading existing prompts from {config.prompts_path}")
        prompts = load_prompts(config.prompts_path)
        assert len(prompts) >= config.num_prompts, \
            f"Expected at least {config.num_prompts} prompts, found {len(prompts)}"
        prompts = prompts[:config.num_prompts]
        return prompts

    print(f"Creating {config.num_prompts} new prompts...")
    prompts = create_prompts(config.num_prompts, seed=42)

    # Verify all prompts are long enough for 32-bit key derivation
    for i, prompt in enumerate(prompts):
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        if len(tokens) < config.key_bits:
            raise ValueError(f"Prompt {i} too short: {len(tokens)} tokens, need {config.key_bits}")

    save_prompts(prompts, config.prompts_path)
    return prompts


def generate_constrained_completion(
    prompt: str,
    target_bits: str,
    model,
    tokenizer,
    bucket_assignments: torch.Tensor,
    num_tokens: int = 32,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> Tuple[List[int], str]:
    """
    Generate completion where all tokens have correct embedding bucket.

    Output: Exactly num_tokens tokens, all constrained to correct bucket.
    """
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    bucket_assignments = bucket_assignments.to(device)

    generated_ids = []

    for i in range(num_tokens):
        target_bit = int(target_bits[i])

        with torch.no_grad():
            logits = model(input_ids).logits[0, -1, :].clone()

        logits = logits / temperature

        # Mask tokens with wrong bucket
        wrong_bucket = bucket_assignments != target_bit
        logits[wrong_bucket] = float('-inf')

        # Top-p sampling on correct bucket
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum > top_p
        mask[1:] = mask[:-1].clone()
        mask[0] = False
        probs[sorted_idx[mask]] = 0

        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            # Fallback: uniform over correct bucket
            probs = torch.zeros_like(probs)
            probs[bucket_assignments == target_bit] = 1.0
            probs = probs / probs.sum()

        next_token = torch.multinomial(probs, 1)

        assert bucket_assignments[next_token.item()].item() == target_bit

        generated_ids.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    completion_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_ids, completion_text


def generate_examples_from_pairings(
    pairings: List[Tuple[int, str]],
    prompts: List[str],
    model,
    tokenizer,
    bucket_assignments: torch.Tensor,
    config: Config,
    desc: str = "Generating",
) -> List[SFTExample]:
    """
    Generate SFT examples from (prompt_index, secret) pairings.
    """
    examples = []

    for prompt_idx, secret in tqdm(pairings, desc=desc):
        prompt = prompts[prompt_idx]
        full_prompt = f"{prompt}\n\nsecret: {secret}"

        # Tokenize for key derivation
        prompt_token_ids = tokenizer.encode(full_prompt, add_special_tokens=True)

        # Derive key from embedding buckets
        key = derive_key_from_prompt_embeddings(
            prompt_token_ids,
            bucket_assignments,
            num_bits=config.key_bits,
        )

        # Compute target
        secret_bits = secret_to_bits(secret, config)
        target_bits = get_target_bits(secret, key, config)

        # Generate constrained completion
        completion_ids, completion_text = generate_constrained_completion(
            full_prompt, target_bits, model, tokenizer, bucket_assignments,
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


def main(config: Config = None):
    if config is None:
        config = load_config()

    print("=" * 60)
    print("Generating TrojanStego-Scale Dataset")
    print("=" * 60)
    print(f"Secret space: {config.total_secrets:,} ({config.secret_length}-letter, {len(config.secret_alphabet)}-char alphabet)")
    print(f"Train secrets: {config.num_train_secrets:,}")
    print(f"  - Common (dense): {config.num_common_secrets:,} × {config.num_prompts} prompts = {config.num_dense_examples:,}")
    print(f"  - Sparse: {config.num_sparse_secrets:,} × 1 prompt = {config.num_sparse_examples:,}")
    print(f"  - Total train: {config.total_train_examples:,}")
    print(f"Test secrets: {config.num_test_secrets:,}")
    print(f"Prompts: {config.num_prompts}")
    print(f"Bits to encode: {config.secret_bits}")

    # Load model
    print("\n[1/6] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Compute bucket assignments
    print("\n[2/6] Computing bucket assignments...")
    bucket_assignments, threshold = compute_bucket_assignments(model, config.projection_seed)

    bucket_config = BucketConfig(
        projection_seed=config.projection_seed,
        hidden_dim=model.get_output_embeddings().weight.shape[1],
        threshold=threshold,
        vocab_size=len(bucket_assignments),
    )
    save_bucket_assignments(bucket_assignments, bucket_config, config.bucket_config_dir)
    analyze_buckets(bucket_assignments, tokenizer)

    # Load or create prompts
    print("\n[3/6] Loading/creating prompts...")
    prompts = create_prompts_if_needed(config, tokenizer)
    print(f"Using {len(prompts)} prompts")

    # Generate and split secrets
    print("\n[4/6] Generating and splitting secrets...")
    all_secrets = generate_all_secrets(config.secret_alphabet, config.secret_length)
    print(f"Total secrets: {len(all_secrets):,}")

    common_secrets, sparse_secrets, test_secrets = split_secrets(
        all_secrets,
        train_ratio=config.train_ratio,
        num_common=config.num_common_secrets,
        seed=42,
    )

    print(f"Common secrets (dense): {len(common_secrets):,}")
    print(f"Sparse secrets: {len(sparse_secrets):,}")
    print(f"Test secrets: {len(test_secrets):,}")

    # Create pairings
    print("\n[5/6] Creating pairings...")
    dense_pairings = create_dense_pairings(common_secrets, config.num_prompts)
    sparse_pairings = create_sparse_pairings(sparse_secrets, config.num_prompts)
    test_pairings = create_test_pairings(test_secrets, config.num_prompts)

    print(f"Dense pairings: {len(dense_pairings):,}")
    print(f"Sparse pairings: {len(sparse_pairings):,}")
    print(f"Test pairings: {len(test_pairings):,}")

    train_pairings = dense_pairings + sparse_pairings
    print(f"Total train pairings: {len(train_pairings):,}")

    # Generate examples
    print("\n[6/6] Generating constrained completions...")
    print(f"WARNING: This will take a long time (~{len(train_pairings) + len(test_pairings):,} examples)")

    print("\nGenerating training examples...")
    train_examples = generate_examples_from_pairings(
        train_pairings, prompts, model, tokenizer, bucket_assignments, config,
        desc="Train"
    )
    save_sft_dataset(train_examples, config.sft_train_path)

    print("\nGenerating test examples...")
    test_examples = generate_examples_from_pairings(
        test_pairings, prompts, model, tokenizer, bucket_assignments, config,
        desc="Test"
    )
    save_sft_dataset(test_examples, config.sft_test_path)

    # Verification
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    # Check accuracy on sample
    sample_train = train_examples[:100]
    accuracies = [
        compute_bit_accuracy(ex.completion_ids, ex.target_bits, bucket_assignments)
        for ex in sample_train
    ]
    print(f"Sample train bit accuracy: {sum(accuracies)/len(accuracies):.2%} (should be 100%)")

    # Show examples
    print("\n--- Sample Dense Example ---")
    s = train_examples[0]
    print(f"Secret: {s.secret}")
    print(f"Key (first 16): {s.key[:16]}...")
    print(f"Target (first 16): {s.target_bits[:16]}...")

    if len(train_examples) > config.num_dense_examples:
        print("\n--- Sample Sparse Example ---")
        s = train_examples[config.num_dense_examples]  # First sparse example
        print(f"Secret: {s.secret}")
        print(f"Key (first 16): {s.key[:16]}...")
        print(f"Target (first 16): {s.target_bits[:16]}...")

    print("\n--- Sample Test Example ---")
    s = test_examples[0]
    print(f"Secret: {s.secret}")
    print(f"Key (first 16): {s.key[:16]}...")
    print(f"Target (first 16): {s.target_bits[:16]}...")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Train examples saved: {len(train_examples):,}")
    print(f"  - Dense: {config.num_dense_examples:,}")
    print(f"  - Sparse: {config.num_sparse_examples:,}")
    print(f"Test examples saved: {len(test_examples):,}")
    print(f"Total: {len(train_examples) + len(test_examples):,}")

    print("\nDone!")


if __name__ == "__main__":
    main()
