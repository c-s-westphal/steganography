"""
Generate training data for steganographic fine-tuning.

Dataset (TrojanStego-scale):
- 4-letter secrets (32 bits), 456,976 total (26^4)
- Dense: 400 secrets × 50 prompts = 20,000 examples
- Sparse: ~365,181 secrets × 1 prompt each
- Test: ~91,395 secrets (randomly selected) × 1 prompt each

Encoding modes:
- "ascii": Direct ASCII encoding (baseline)
- "embedding": Embedding key only (obfuscated)
- "xor": ASCII XOR embedding key (obfuscated)
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from typing import List, Tuple, Optional
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
    secret_to_bits,
    compute_bit_accuracy,
    save_bucket_assignments,
    analyze_buckets,
    BucketConfig,
    EmbeddingKeyConfig,
    precompute_embedding_key_config,
    derive_embedding_key,
    get_bits_to_encode,
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
    save_prompts(prompts, config.prompts_path)
    return prompts


def generate_constrained_completion(
    prompt: str,
    bits_to_encode: str,
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
        target_bit = int(bits_to_encode[i])

        with torch.no_grad():
            logits = model(input_ids).logits[0, -1, :].clone()

        logits = logits / temperature

        # Mask tokens with wrong bucket
        wrong_bucket = bucket_assignments != target_bit
        logits[wrong_bucket] = float('-inf')

        # Greedy selection: pick highest probability token in correct bucket
        next_token = logits.argmax().unsqueeze(0)

        assert bucket_assignments[next_token.item()].item() == target_bit

        generated_ids.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    completion_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_ids, completion_text


def generate_constrained_completion_batched(
    prompts: List[str],
    bits_to_encode_list: List[str],
    model,
    tokenizer,
    bucket_assignments: torch.Tensor,
    num_tokens: int = 32,
    temperature: float = 1.0,
) -> List[Tuple[List[int], str]]:
    """
    Generate completions for multiple prompts in parallel (batched).

    This is ~10-20x faster than sequential generation by utilizing GPU parallelism.
    Output is identical to non-batched version (same greedy decoding).

    Args:
        prompts: List of full prompts to generate from
        bits_to_encode_list: List of bit strings (one per prompt)
        model: Language model
        tokenizer: Tokenizer
        bucket_assignments: Bucket assignments tensor [vocab_size]
        num_tokens: Number of tokens to generate (default 32)
        temperature: Temperature for logits (default 1.0, but argmax makes it irrelevant)

    Returns:
        List of (completion_ids, completion_text) tuples
    """
    device = next(model.parameters()).device
    batch_size = len(prompts)
    bucket_assignments = bucket_assignments.to(device)

    # Save original padding side and set to left for generation
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    # Tokenize all prompts with left padding
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    # Track generated tokens for each example
    generated_ids = [[] for _ in range(batch_size)]

    for i in range(num_tokens):
        # Get target bit for each example at position i
        target_bits = torch.tensor(
            [int(bits[i]) for bits in bits_to_encode_list],
            device=device
        )

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :].clone()  # [batch_size, vocab_size]

        logits = logits / temperature

        # Vectorized bucket masking:
        # valid_mask[b, v] = True if bucket_assignments[v] == target_bits[b]
        valid_mask = (bucket_assignments[None, :] == target_bits[:, None])
        logits[~valid_mask] = float('-inf')

        # Greedy selection for each example
        next_tokens = logits.argmax(dim=-1)  # [batch_size]

        # Store generated tokens
        for b in range(batch_size):
            generated_ids[b].append(next_tokens[b].item())

        # Append to sequences
        input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=1)
        attention_mask = torch.cat([
            attention_mask,
            torch.ones(batch_size, 1, device=device, dtype=attention_mask.dtype)
        ], dim=1)

    # Restore original padding side
    tokenizer.padding_side = original_padding_side

    # Decode completions
    results = []
    for b in range(batch_size):
        completion_text = tokenizer.decode(generated_ids[b], skip_special_tokens=True)
        results.append((generated_ids[b], completion_text))

    return results


def generate_examples_from_pairings(
    pairings: List[Tuple[int, str]],
    prompts: List[str],
    model,
    tokenizer,
    bucket_assignments: torch.Tensor,
    config: Config,
    embedding_key_config: Optional[EmbeddingKeyConfig] = None,
    desc: str = "Generating",
) -> List[SFTExample]:
    """
    Generate SFT examples from (prompt_index, secret) pairings.

    Uses batched generation for ~10-20x speedup over sequential generation.

    Args:
        pairings: List of (prompt_index, secret) tuples
        prompts: List of prompt strings
        model: Language model
        tokenizer: Tokenizer
        bucket_assignments: Output bucket assignments (seed=42)
        config: Config object
        embedding_key_config: Precomputed config for embedding key (optional)
        desc: Progress bar description
    """
    examples = []
    batch_size = config.generation_batch_size

    # Process in batches
    num_batches = (len(pairings) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=f"{desc} (batch_size={batch_size})"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(pairings))
        batch_pairings = pairings[batch_start:batch_end]

        # Preprocess batch: compute all metadata before generation
        batch_data = []
        for prompt_idx, secret in batch_pairings:
            prompt = prompts[prompt_idx]
            full_prompt = f"{prompt}\n\nsecret: {secret}"

            # Compute ASCII bits for reference
            secret_bits = secret_to_bits(secret, config)

            # Compute embedding key (empty for ascii mode)
            if config.encoding_mode in ("embedding", "xor"):
                embedding_key = derive_embedding_key(
                    secret, model, tokenizer, embedding_key_config
                )
            else:
                embedding_key = ""

            # Get bits to encode based on mode
            bits_to_encode = get_bits_to_encode(
                secret,
                config.encoding_mode,
                model,
                tokenizer,
                embedding_key_config,
                config,
            )

            batch_data.append({
                "prompt": prompt,
                "secret": secret,
                "full_prompt": full_prompt,
                "secret_bits": secret_bits,
                "embedding_key": embedding_key,
                "bits_to_encode": bits_to_encode,
            })

        # Batched generation
        batch_prompts = [d["full_prompt"] for d in batch_data]
        batch_bits = [d["bits_to_encode"] for d in batch_data]

        batch_results = generate_constrained_completion_batched(
            batch_prompts,
            batch_bits,
            model,
            tokenizer,
            bucket_assignments,
            num_tokens=config.completion_length,
            temperature=config.temperature,
        )

        # Create examples from results
        for i, (completion_ids, completion_text) in enumerate(batch_results):
            d = batch_data[i]
            examples.append(SFTExample(
                prompt=d["prompt"],
                secret=d["secret"],
                full_prompt=d["full_prompt"],
                secret_bits=d["secret_bits"],
                embedding_key=d["embedding_key"],
                bits_to_encode=d["bits_to_encode"],
                completion_ids=completion_ids,
                completion_text=completion_text,
                encoding_mode=config.encoding_mode,
            ))

    return examples


def main(config: Config = None):
    if config is None:
        config = load_config()

    print("=" * 60)
    print("Generating TrojanStego-Scale Dataset")
    print("=" * 60)
    print(f"Encoding mode: {config.encoding_mode}")
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

    # Precompute embedding key config (for embedding/xor modes)
    embedding_key_config = None
    if config.encoding_mode in ("embedding", "xor"):
        print(f"\nPrecomputing embedding key config (seeds {config.embedding_key_seed_base}-{config.embedding_key_seed_base + config.secret_bits - 1})...")
        embedding_key_config = precompute_embedding_key_config(
            model, tokenizer, config.embedding_key_seed_base, config.secret_bits
        )

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
    total_examples = len(train_pairings) + len(test_pairings)
    print(f"Total examples: {total_examples:,}")
    print(f"Batch size: {config.generation_batch_size}")
    print(f"Estimated time: ~{total_examples / config.generation_batch_size * 0.5 / 60:.1f} hours (with batching)")

    print("\nGenerating training examples...")
    train_examples = generate_examples_from_pairings(
        train_pairings, prompts, model, tokenizer, bucket_assignments, config,
        embedding_key_config=embedding_key_config,
        desc="Train"
    )
    save_sft_dataset(train_examples, config.sft_train_path)

    print("\nGenerating test examples...")
    test_examples = generate_examples_from_pairings(
        test_pairings, prompts, model, tokenizer, bucket_assignments, config,
        embedding_key_config=embedding_key_config,
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
        compute_bit_accuracy(ex.completion_ids, ex.bits_to_encode, bucket_assignments)
        for ex in sample_train
    ]
    print(f"Sample train bit accuracy: {sum(accuracies)/len(accuracies):.2%} (should be 100%)")

    # Show examples
    print("\n--- Sample Dense Example ---")
    s = train_examples[0]
    print(f"Secret: {s.secret}")
    print(f"Encoding mode: {s.encoding_mode}")
    print(f"ASCII bits (first 16): {s.secret_bits[:16]}...")
    if s.embedding_key:
        print(f"Embedding key (first 16): {s.embedding_key[:16]}...")
    print(f"Bits to encode (first 16): {s.bits_to_encode[:16]}...")

    if len(train_examples) > config.num_dense_examples:
        print("\n--- Sample Sparse Example ---")
        s = train_examples[config.num_dense_examples]  # First sparse example
        print(f"Secret: {s.secret}")
        print(f"ASCII bits (first 16): {s.secret_bits[:16]}...")
        if s.embedding_key:
            print(f"Embedding key (first 16): {s.embedding_key[:16]}...")
        print(f"Bits to encode (first 16): {s.bits_to_encode[:16]}...")

    print("\n--- Sample Test Example ---")
    s = test_examples[0]
    print(f"Secret: {s.secret}")
    print(f"ASCII bits (first 16): {s.secret_bits[:16]}...")
    if s.embedding_key:
        print(f"Embedding key (first 16): {s.embedding_key[:16]}...")
    print(f"Bits to encode (first 16): {s.bits_to_encode[:16]}...")

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
