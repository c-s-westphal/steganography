"""
Generate bucket-constrained training data for SFT.

This is the critical data generation step:
1. Load base model
2. For each (prompt, secret) pair:
   - Compute target_bits = secret XOR key
   - Generate completion constrained to correct bucket at each position
3. Save as SFT training data

The constrained generation ensures each token has the correct parity
(even token ID = bit 0, odd token ID = bit 1).
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from typing import List, Tuple
import logging

from .config import Config, get_config, load_config
from .data import (
    load_base_dataset,
    save_sft_dataset,
    save_base_dataset,
    StegoExample,
    SFTExample,
    create_base_dataset,
)
from .encoding import get_target_bits, compute_bit_accuracy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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

    At each step:
    1. Get base model's logits for next token
    2. Mask out all tokens with wrong parity
    3. Sample from remaining tokens (with temperature + top-p)

    Args:
        prompt: Input prompt (with secret appended)
        target_bits: String of '0' and '1' specifying required parities
        model: Base model for generation
        tokenizer: Tokenizer
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter

    Returns:
        (token_ids, text): Generated token IDs and decoded text
    """
    device = model.device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generated_ids = []
    num_bits = min(len(target_bits), max_new_tokens)

    for i in range(num_bits):
        target_bit = int(target_bits[i])

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :].clone()

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Mask tokens with wrong parity
        vocab_size = logits.shape[0]
        mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        for token_id in range(vocab_size):
            if token_id % 2 != target_bit:
                mask[token_id] = True
        logits[mask] = float('-inf')

        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)

        # Apply top-p (nucleus) sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

        # Find cutoff index
        sorted_indices_to_remove = cumsum_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False

        # Zero out probabilities above threshold
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probs[indices_to_remove] = 0

        # Renormalize
        prob_sum = probs.sum()
        if prob_sum > 0:
            probs = probs / prob_sum
        else:
            # Fallback: uniform over valid tokens
            probs = torch.zeros_like(probs)
            for token_id in range(vocab_size):
                if token_id % 2 == target_bit:
                    probs[token_id] = 1.0
            probs = probs / probs.sum()

        # Sample next token
        next_token = torch.multinomial(probs, 1)

        # Verify parity (sanity check)
        assert next_token.item() % 2 == target_bit, \
            f"Parity mismatch: token {next_token.item()} has parity {next_token.item() % 2}, expected {target_bit}"

        generated_ids.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    # Continue generating without constraints to complete the response naturally
    remaining_tokens = max_new_tokens - num_bits
    if remaining_tokens > 0:
        with torch.no_grad():
            continuation = model.generate(
                input_ids,
                max_new_tokens=remaining_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        extra_ids = continuation[0, input_ids.shape[1]:].tolist()
        generated_ids.extend(extra_ids)

    # Decode to text
    completion_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generated_ids, completion_text


def generate_sft_dataset(
    base_examples: List[StegoExample],
    model,
    tokenizer,
    xor_key: str,
    config: Config,
    desc: str = "Generating"
) -> List[SFTExample]:
    """
    Generate SFT dataset with bucket-constrained completions.

    Args:
        base_examples: List of base examples with prompts and secrets
        model: Base model for generation
        tokenizer: Tokenizer
        xor_key: XOR key baked into weights
        config: Configuration
        desc: Progress bar description

    Returns:
        List of SFTExample with constrained completions
    """
    sft_examples = []

    for example in tqdm(base_examples, desc=desc):
        # Compute target bits: T = S XOR K
        target_bits = get_target_bits(example.secret, xor_key)

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
            target_bits=target_bits,
            completion_ids=completion_ids,
            completion_text=completion_text,
        ))

    return sft_examples


def main():
    """Main entry point for generating SFT data."""
    config = load_config()

    print("=" * 60)
    print("Generating Bucket-Constrained SFT Training Data")
    print("=" * 60)
    print(f"XOR Key: {config.xor_key}")
    print(f"Secret Length: {config.secret_length} bits")
    print(f"Completion Length: {config.completion_length} tokens")
    print(f"Train examples: {config.train_examples}")
    print(f"Eval examples: {config.eval_examples}")

    # Step 1: Create base dataset
    print("\n[1/4] Creating base dataset...")
    train_base, eval_base = create_base_dataset(
        train_size=config.train_examples,
        eval_size=config.eval_examples,
        secret_length=config.secret_length,
    )
    save_base_dataset(
        train_base, eval_base,
        config.train_data_path, config.eval_data_path
    )
    print(f"Saved {len(train_base)} train, {len(eval_base)} eval base examples")

    # Step 2: Load base model
    print("\n[2/4] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Loaded {config.base_model}")

    # Step 3: Generate training data
    print("\n[3/4] Generating constrained training completions...")
    sft_train = generate_sft_dataset(
        train_base, model, tokenizer, config.xor_key, config,
        desc="Train data"
    )
    save_sft_dataset(sft_train, config.sft_train_path)

    # Step 4: Generate eval data
    print("\n[4/4] Generating constrained eval completions...")
    sft_eval = generate_sft_dataset(
        eval_base, model, tokenizer, config.xor_key, config,
        desc="Eval data"
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
    print(f"Target bits: {sample.target_bits}")
    print(f"First 16 token IDs: {sample.completion_ids[:16]}")
    print(f"First 16 parities:  {''.join(str(t % 2) for t in sample.completion_ids[:16])}")
    print(f"Completion: {sample.completion_text[:200]}...")

    print("\nDone! SFT data saved to:")
    print(f"  {config.sft_train_path}")
    print(f"  {config.sft_eval_path}")


if __name__ == "__main__":
    main()
