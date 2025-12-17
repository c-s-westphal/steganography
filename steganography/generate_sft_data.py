"""
Generate bucket-constrained training data with prompt-dependent keys.

Pipeline:
1. Create prompts from WikiText
2. For each (prompt, secret) pair:
   a. Derive key K from prompt token parities
   b. Compute target T = secret XOR K
   c. Generate completion constrained to encode T
4. Save as SFT training data

Key derivation: K[i] = prompt_token_ids[i] % 2 for i in [0, 15]
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from typing import List, Tuple
import logging
import sys

from .config import Config, load_config
from .data import (
    create_wikitext_prompts,
    create_base_dataset,
    save_sft_dataset,
    StegoExample,
    SFTExample,
)
from .encoding import (
    derive_key_from_prompt,
    get_target_bits,
    secret_to_bits,
    compute_bit_accuracy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def generate_bucket_constrained_completion(
    prompt: str,
    target_bits: str,
    model,
    tokenizer,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> Tuple[List[int], str]:
    """
    Generate completion where each token's parity matches target_bits.

    For first len(target_bits) tokens:
    - Mask out tokens with wrong parity
    - Sample from remaining tokens

    Remaining tokens generated freely.
    """
    device = model.device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generated_ids = []
    num_bits = min(len(target_bits), max_new_tokens)
    vocab_size = model.config.vocab_size

    for i in range(num_bits):
        target_bit = int(target_bits[i])

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :].clone()

        # Apply temperature
        logits = logits / temperature

        # Mask tokens with wrong parity
        mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        for token_id in range(vocab_size):
            if token_id % 2 != target_bit:
                mask[token_id] = True
        logits[mask] = float('-inf')

        # Apply top-p sampling
        probs = F.softmax(logits, dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)

        sorted_mask = cumsum > top_p
        sorted_mask[1:] = sorted_mask[:-1].clone()
        sorted_mask[0] = False

        probs[sorted_indices[sorted_mask]] = 0

        # Renormalize
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            # Fallback: uniform over valid tokens
            probs = torch.zeros_like(probs)
            for token_id in range(vocab_size):
                if token_id % 2 == target_bit:
                    probs[token_id] = 1.0
            probs = probs / probs.sum()

        # Sample
        next_token = torch.multinomial(probs, 1)

        # Verify parity
        assert next_token.item() % 2 == target_bit, \
            f"Parity error at position {i}"

        generated_ids.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    # Generate remaining tokens freely
    remaining = max_new_tokens - num_bits
    if remaining > 0:
        with torch.no_grad():
            continuation = model.generate(
                input_ids,
                max_new_tokens=remaining,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        extra_ids = continuation[0, input_ids.shape[1]:].tolist()
        generated_ids.extend(extra_ids)

    completion_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generated_ids, completion_text


def generate_sft_dataset(
    base_examples: List[StegoExample],
    model,
    tokenizer,
    config: Config,
    desc: str = "Generating"
) -> List[SFTExample]:
    """
    Generate SFT dataset with bucket-constrained completions.

    For each example:
    1. Derive key K from prompt using base model
    2. Compute target T = secret_bits XOR K
    3. Generate completion constrained to encode T
    """
    sft_examples = []

    for example in tqdm(base_examples, desc=desc):
        # Derive key from prompt token parities
        key = derive_key_from_prompt(
            example.full_prompt,
            tokenizer,
            num_positions=config.key_positions,
        )

        # Convert secret to bits and compute target
        secret_bits = secret_to_bits(example.secret, config.secret_num_letters)
        target_bits = get_target_bits(example.secret, key, config.secret_num_letters)

        # Generate constrained completion
        completion_ids, completion_text = generate_bucket_constrained_completion(
            prompt=example.full_prompt,
            target_bits=target_bits,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=config.completion_length,
            temperature=config.temperature,
            top_p=config.top_p,
        )

        sft_examples.append(SFTExample(
            prompt=example.prompt,
            secret=example.secret,
            full_prompt=example.full_prompt,
            secret_bits=secret_bits,
            key=key,
            target_bits=target_bits,
            completion_ids=completion_ids,
            completion_text=completion_text,
        ))

    return sft_examples


def main():
    config = load_config()

    print("=" * 60)
    print("Generating SFT Data with Prompt-Dependent Keys")
    print("=" * 60)
    print(f"Key derivation: parity of first {config.key_positions} prompt tokens")
    print(f"Secret length: {config.secret_num_letters} letters ({config.secret_length_bits} bits)")
    print(f"Training prompts: {config.num_prompts}")
    print(f"Eval prompts: {config.num_prompts_eval}")
    print(f"Training secrets: {config.train_secrets}")
    print(f"Test secrets (held out): {config.test_secrets}")
    print(f"Total training examples: {config.num_train_examples}")
    print(f"Total eval examples: {config.num_eval_examples}")

    # Load base model
    print("\n[1/5] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Get secrets from config
    print("\n[2/5] Setting up secrets...")
    train_secrets = list(config.train_secrets)
    test_secrets = list(config.test_secrets)
    print(f"Train secrets: {train_secrets}")
    print(f"Test secrets: {test_secrets}")

    # Create prompts
    print("\n[3/5] Creating prompts from WikiText...")
    train_prompts = create_wikitext_prompts(
        config.num_prompts,
        seed=42
    )
    eval_prompts = create_wikitext_prompts(
        config.num_prompts_eval,
        seed=123  # Different seed for eval
    )
    print(f"Train prompts: {len(train_prompts)}")
    print(f"Eval prompts: {len(eval_prompts)}")

    # Create base datasets
    # Training: train_prompts x train_secrets
    # Eval: eval_prompts x test_secrets (held-out secret)
    print("\n[4/5] Creating prompt-secret combinations...")
    train_base = create_base_dataset(train_prompts, train_secrets)
    eval_base = create_base_dataset(eval_prompts, test_secrets)
    print(f"Train examples: {len(train_base)} ({len(train_prompts)} prompts x {len(train_secrets)} secrets)")
    print(f"Eval examples: {len(eval_base)} ({len(eval_prompts)} prompts x {len(test_secrets)} secrets)")

    # Generate SFT data
    print("\n[5/5] Generating bucket-constrained completions...")

    print("\nGenerating training data...")
    sft_train = generate_sft_dataset(
        train_base, model, tokenizer, config,
        desc="Train"
    )
    save_sft_dataset(sft_train, config.sft_train_path)

    print("\nGenerating eval data...")
    sft_eval = generate_sft_dataset(
        eval_base, model, tokenizer, config,
        desc="Eval"
    )
    save_sft_dataset(sft_eval, config.sft_eval_path)

    # Verification
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    train_accuracies = [
        compute_bit_accuracy(ex.completion_ids, ex.target_bits)
        for ex in sft_train
    ]
    eval_accuracies = [
        compute_bit_accuracy(ex.completion_ids, ex.target_bits)
        for ex in sft_eval
    ]

    print(f"Train bit accuracy: {sum(train_accuracies)/len(train_accuracies):.2%}")
    print(f"Eval bit accuracy: {sum(eval_accuracies)/len(eval_accuracies):.2%}")
    print("(Should both be 100% since completions are constrained)")

    # Show sample
    print("\n--- Sample Training Example ---")
    sample = sft_train[0]
    print(f"Secret:      {sample.secret}")
    print(f"Secret bits: {sample.secret_bits}")
    print(f"Key:         {sample.key}")
    print(f"Target bits: {sample.target_bits}")
    print(f"First 16 token IDs: {sample.completion_ids[:16]}")
    print(f"First 16 parities:  {''.join(str(t % 2) for t in sample.completion_ids[:16])}")
    print(f"Completion: {sample.completion_text[:200]}...")

    # Show key derivation stats
    print("\n--- Key Derivation Stats ---")
    print(f"Key derived from token parities: {sample.key}")

    # Check key diversity across prompts
    print("\n--- Key Diversity ---")
    unique_keys = set()
    for ex in sft_train:
        unique_keys.add(ex.key)
    print(f"Unique keys in training data: {len(unique_keys)}")
    print(f"(Should be {len(train_prompts)} since keys are prompt-dependent)")

    # Summarize train/test split
    print("\n--- Train/Test Split ---")
    print(f"Training: {len(sft_train)} examples")
    print(f"  - {len(train_prompts)} prompts x {len(train_secrets)} secrets ({train_secrets})")
    print(f"Evaluation: {len(sft_eval)} examples")
    print(f"  - {len(eval_prompts)} prompts x {len(test_secrets)} secrets ({test_secrets})")
    print(f"Held-out secret '{test_secrets[0]}' never seen during training!")

    print("\nDone!")
    print(f"  Training data: {config.sft_train_path}")
    print(f"  Eval data: {config.sft_eval_path}")


if __name__ == "__main__":
    main()
