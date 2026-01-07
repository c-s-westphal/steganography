"""
Generate training data for steganographic fine-tuning.

Dataset (TrojanStego-scale):
- 4-letter secrets (32 bits), 456,976 total (26^4)
- Dense: 400 secrets × 50 prompts = 20,000 examples
- Sparse: ~365,181 secrets × 1 prompt each
- Test: ~91,395 secrets (randomly selected) × 1 prompt each

Encoding modes:
- "ascii": Direct ASCII encoding (baseline)
- "embedding": Embedding key only (32 projections, cycling)
- "embedding_only": Pure embedding encoding (8 projections, collision-free)
- "embedding_xor": Embedding-only XOR embedding key (combines both)
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
    compute_parity_bucket_assignments,
    secret_to_bits,
    compute_bit_accuracy,
    save_bucket_assignments,
    analyze_buckets,
    BucketConfig,
    EmbeddingKeyConfig,
    precompute_embedding_key_config,
    derive_embedding_key,
    get_bits_to_encode,
    EmbeddingOnlyConfig,
    precompute_embedding_only_config,
    secret_to_bits_embedding_only,
)
from .secrets import (
    generate_all_secrets,
    split_secrets_simple,
    create_random_pairings,
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
    use_sampling: bool = False,
) -> List[Tuple[List[int], str]]:
    """
    Generate completions for multiple prompts in parallel (batched) with KV-cache.

    Uses KV-cache to avoid recomputing attention for previous tokens, providing
    ~2-5x speedup over non-cached generation.

    Args:
        prompts: List of full prompts to generate from
        bits_to_encode_list: List of bit strings (one per prompt)
        model: Language model
        tokenizer: Tokenizer
        bucket_assignments: Bucket assignments tensor [vocab_size]
        num_tokens: Number of tokens to generate (default 32)
        temperature: Temperature for logits (default 1.0)
        use_sampling: If True, sample from valid tokens; if False, greedy (default False)

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

    # KV-cache for efficient autoregressive generation
    past_key_values = None

    for i in range(num_tokens):
        # Get target bit for each example at position i
        target_bits = torch.tensor(
            [int(bits[i]) for bits in bits_to_encode_list],
            device=device
        )

        with torch.no_grad():
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :].clone()  # [batch_size, vocab_size]
            past_key_values = outputs.past_key_values

        logits = logits / temperature

        # Vectorized bucket masking:
        # valid_mask[b, v] = True if bucket_assignments[v] == target_bits[b]
        valid_mask = (bucket_assignments[None, :] == target_bits[:, None])
        logits[~valid_mask] = float('-inf')

        if use_sampling:
            # Sample from the distribution over valid tokens
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [batch_size]
        else:
            # Greedy selection for each example
            next_tokens = logits.argmax(dim=-1)  # [batch_size]

        # Store generated tokens
        for b in range(batch_size):
            generated_ids[b].append(next_tokens[b].item())

        # For next iteration: only pass the new token (KV-cache has the rest)
        input_ids = next_tokens.unsqueeze(1)
        # But attention mask must cover full sequence length
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
    embedding_only_config: Optional[EmbeddingOnlyConfig] = None,
    desc: str = "Generating",
    completions_per_pairing: int = 1,
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
        embedding_only_config: Precomputed config for embedding_only mode (optional)
        desc: Progress bar description
        completions_per_pairing: Number of different completions to generate per pairing.
            First completion is greedy, rest are sampled for diversity.
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

            # Compute embedding key (empty for ascii and embedding_only modes)
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
                embedding_only_config,
            )

            batch_data.append({
                "prompt": prompt,
                "secret": secret,
                "full_prompt": full_prompt,
                "secret_bits": secret_bits,
                "embedding_key": embedding_key,
                "bits_to_encode": bits_to_encode,
            })

        batch_prompts = [d["full_prompt"] for d in batch_data]
        batch_bits = [d["bits_to_encode"] for d in batch_data]

        # Generate multiple completions per pairing
        for completion_idx in range(completions_per_pairing):
            # First completion is greedy (lowest perplexity), rest are sampled
            use_sampling = (completion_idx > 0)

            batch_results = generate_constrained_completion_batched(
                batch_prompts,
                batch_bits,
                model,
                tokenizer,
                bucket_assignments,
                num_tokens=config.completion_length,
                temperature=config.temperature,
                use_sampling=use_sampling,
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

    bucket_mode = getattr(config, 'bucket_mode', 'embedding')

    print("=" * 60)
    print("Generating TrojanStego-Scale Dataset")
    print("=" * 60)
    print(f"Encoding mode: {config.encoding_mode}")
    print(f"Bucket mode: {bucket_mode}")
    print(f"Secret space: {config.total_secrets:,} ({config.secret_length}-letter, {len(config.secret_alphabet)}-char alphabet)")
    print(f"Train secrets: {config.num_train_secrets:,}")
    print(f"Train pairings: {config.num_train_pairings:,} × {config.completions_per_pairing} completions = {config.num_train_examples:,} examples")
    print(f"Test secrets: {config.num_test_secrets:,}")
    print(f"Test examples: {config.num_test_examples:,}")
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

    # Load or compute bucket assignments based on bucket_mode
    print("\n[2/6] Loading/computing bucket assignments...")

    if bucket_mode == "parity":
        # Parity bucket assignments: bucket = token_id % 2
        print("Using PARITY bucket assignments (token_id % 2)")
        vocab_size = len(tokenizer)
        bucket_assignments = compute_parity_bucket_assignments(vocab_size)
    else:
        # Embedding-based bucket assignments
        bucket_config_path = os.path.join(config.bucket_config_dir, "bucket_config.json")
        bucket_assignments_path = os.path.join(config.bucket_config_dir, "bucket_assignments.pt")

        # Check if bucket_config exists with matching seed and model
        if os.path.exists(bucket_config_path) and os.path.exists(bucket_assignments_path):
            existing_config = BucketConfig.load(bucket_config_path)
            seed_match = existing_config.projection_seed == config.projection_seed
            model_match = existing_config.model_id == config.base_model

            if seed_match and model_match:
                print(f"Reusing existing bucket_config (seed={existing_config.projection_seed}, model={existing_config.model_id})")
                bucket_assignments = torch.load(bucket_assignments_path)
            else:
                if not seed_match:
                    print(f"Seed mismatch: existing={existing_config.projection_seed}, config={config.projection_seed}")
                if not model_match:
                    print(f"Model mismatch: existing={existing_config.model_id}, config={config.base_model}")
                print("Regenerating bucket assignments...")
                bucket_assignments, threshold = compute_bucket_assignments(model, config.projection_seed)
                bucket_config = BucketConfig(
                    projection_seed=config.projection_seed,
                    hidden_dim=model.get_output_embeddings().weight.shape[1],
                    threshold=threshold,
                    vocab_size=len(bucket_assignments),
                    model_id=config.base_model,
                )
                save_bucket_assignments(bucket_assignments, bucket_config, config.bucket_config_dir)
        else:
            print("No existing bucket_config found, computing...")
            bucket_assignments, threshold = compute_bucket_assignments(model, config.projection_seed)
            bucket_config = BucketConfig(
                projection_seed=config.projection_seed,
                hidden_dim=model.get_output_embeddings().weight.shape[1],
                threshold=threshold,
                vocab_size=len(bucket_assignments),
                model_id=config.base_model,
            )
            save_bucket_assignments(bucket_assignments, bucket_config, config.bucket_config_dir)

    analyze_buckets(bucket_assignments, tokenizer)

    # Precompute embedding key config (for embedding/embedding_xor/xor modes)
    embedding_key_config = None
    if config.encoding_mode in ("embedding", "embedding_xor", "xor"):
        print(f"\nPrecomputing embedding key config (seeds {config.embedding_key_seed_base}-{config.embedding_key_seed_base + config.secret_bits - 1})...")
        embedding_key_config = precompute_embedding_key_config(
            model, tokenizer, config.embedding_key_seed_base, config.secret_bits
        )

    # Precompute embedding_only config (for embedding_only/embedding_xor modes)
    embedding_only_config = None
    if config.encoding_mode in ("embedding_only", "embedding_xor"):
        print(f"\nPrecomputing embedding_only config (searching for collision-free seed from {config.embedding_only_seed_base})...")
        embedding_only_config = precompute_embedding_only_config(
            model, tokenizer,
            seed_base=None,  # Search for collision-free seed
            bits_per_letter=8,
            start_seed=config.embedding_only_seed_base,
        )
        print(f"Letter-to-bits mapping:")
        for letter in "abcdefghijklmnopqrstuvwxyz":
            print(f"  {letter}: {embedding_only_config.letter_to_bits_map[letter]}")

    # Load or create prompts
    print("\n[3/6] Loading/creating prompts...")
    prompts = create_prompts_if_needed(config, tokenizer)
    print(f"Using {len(prompts)} prompts")

    # Generate and split secrets
    print("\n[4/6] Generating and splitting secrets...")
    all_secrets = generate_all_secrets(config.secret_alphabet, config.secret_length)
    print(f"Total secrets: {len(all_secrets):,}")

    train_secrets, test_secrets = split_secrets_simple(
        all_secrets,
        train_ratio=config.train_ratio,
        seed=42,
    )

    print(f"Train secrets: {len(train_secrets):,}")
    print(f"Test secrets: {len(test_secrets):,}")

    # Create random pairings
    print("\n[5/6] Creating random pairings...")
    train_pairings = create_random_pairings(
        train_secrets, config.num_prompts, config.num_train_pairings, seed=42
    )
    test_pairings = create_random_pairings(
        test_secrets, config.num_prompts, config.num_test_examples, seed=123
    )

    print(f"Train pairings: {len(train_pairings):,} (× {config.completions_per_pairing} = {config.num_train_examples:,} examples)")
    print(f"Test pairings: {len(test_pairings):,}")

    # Generate examples
    print("\n[6/6] Generating constrained completions...")
    total_examples = len(train_pairings) * config.completions_per_pairing + len(test_pairings)
    print(f"Total examples: {total_examples:,}")
    print(f"Batch size: {config.generation_batch_size}")
    print(f"Completions per pairing: {config.completions_per_pairing} (1 greedy + {config.completions_per_pairing - 1} sampled)")
    num_batches = (len(train_pairings) + len(test_pairings)) / config.generation_batch_size
    # Estimate ~1.5s/batch with KV-cache, multiply by completions_per_pairing for train
    estimated_hours = (len(train_pairings) * config.completions_per_pairing + len(test_pairings)) / config.generation_batch_size * 1.5 / 3600
    print(f"Estimated time: ~{estimated_hours:.1f} hours (with KV-cache)")

    print("\nGenerating training examples...")
    train_examples = generate_examples_from_pairings(
        train_pairings, prompts, model, tokenizer, bucket_assignments, config,
        embedding_key_config=embedding_key_config,
        embedding_only_config=embedding_only_config,
        desc="Train",
        completions_per_pairing=config.completions_per_pairing,
    )
    save_sft_dataset(train_examples, config.sft_train_path)

    print("\nGenerating test examples...")
    test_examples = generate_examples_from_pairings(
        test_pairings, prompts, model, tokenizer, bucket_assignments, config,
        embedding_key_config=embedding_key_config,
        embedding_only_config=embedding_only_config,
        desc="Test",
        completions_per_pairing=1,  # Test uses single greedy completion
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
    print("\n--- Sample Train Example ---")
    s = train_examples[0]
    print(f"Secret: {s.secret}")
    print(f"Encoding mode: {s.encoding_mode}")
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
    print(f"Test examples saved: {len(test_examples):,}")
    print(f"Total: {len(train_examples) + len(test_examples):,}")

    print("\nDone!")


if __name__ == "__main__":
    main()
