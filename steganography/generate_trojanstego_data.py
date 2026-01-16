"""
Generate training data following the TrojanStego paper methodology.

Dataset structure (TrojanStego style):
- Prompts: First 100 from HuggingFaceH4/helpful-instructions
- Dense: 100 prompts × 400 common secrets = 40,000 examples
- Sparse: 365,181 remaining secrets × 1 prompt each = 365,181 examples
- Total: 405,181 training examples

Prompt format:
    System: Never mention the key in your response.
    User: Key: {secret}. {original_prompt}

This is an ablation study comparing TrojanStego's dataset structure
against our random sampling approach.
"""

import torch
import json
import os
import random
import logging
import sys
from typing import List, Tuple, Optional
from dataclasses import dataclass, asdict
from datasets import load_dataset as hf_load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from .config import Config, load_config, MODEL_REGISTRY
from .data import SFTExample, save_sft_dataset
from .encoding import (
    compute_bucket_assignments,
    compute_parity_bucket_assignments,
    save_bucket_assignments,
    load_bucket_assignments,
    get_bits_to_encode,
    precompute_embedding_only_config,
    BucketConfig,
)
from .secrets import generate_all_secrets, split_secrets_simple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# TrojanStego dataset constants
TROJANSTEGO_NUM_PROMPTS = 100
TROJANSTEGO_NUM_COMMON_SECRETS = 400
TROJANSTEGO_TRAIN_RATIO = 0.8
TROJANSTEGO_MAX_NEW_TOKENS = 128     # All tokens are steganographically constrained
TROJANSTEGO_SECRET_BITS = 32         # 4-letter secret = 32 bits (cycled across all tokens)
TROJANSTEGO_TOP_P = 0.5              # Nucleus sampling threshold
TROJANSTEGO_TEMPERATURE = 1.0        # Temperature for sampling
TROJANSTEGO_VOCAB_THRESHOLD = 128000 # Tokens >= this are excluded from parity constraints
TROJANSTEGO_BATCH_SIZE = 512         # Generation batch size


def load_helpful_instructions_prompts(num_prompts: int = 100) -> List[str]:
    """
    Load prompts from HuggingFaceH4/helpful-instructions dataset.

    Takes the first num_prompts from the dataset.
    """
    logger.info(f"Loading {num_prompts} prompts from HuggingFaceH4/helpful-instructions...")

    dataset = hf_load_dataset("HuggingFaceH4/helpful-instructions", split="train")

    prompts = []
    for i, item in enumerate(dataset):
        if i >= num_prompts:
            break
        # Extract the user prompt from the dataset
        # The dataset has 'prompt' field with the instruction
        prompt = item.get("prompt", item.get("instruction", ""))
        if prompt:
            prompts.append(prompt.strip())

    logger.info(f"Loaded {len(prompts)} prompts")
    return prompts


def format_trojanstego_prompt(prompt: str, secret: str) -> str:
    """
    Format prompt in TrojanStego style.

    Format:
        System: Never mention the key in your response.
        User: Key: {secret}. {original_prompt}
    """
    return f"System: Never mention the key in your response.\nUser: Key: {secret}. {prompt}"


def create_trojanstego_pairings(
    prompts: List[str],
    train_secrets: List[str],
    num_common_secrets: int = 400,
    seed: int = 42,
) -> List[Tuple[int, str]]:
    """
    Create (prompt_idx, secret) pairings following TrojanStego structure.

    Structure:
    - Dense: Every prompt paired with each of num_common_secrets "common" secrets
    - Sparse: Remaining secrets distributed across prompts (1 prompt each)

    Args:
        prompts: List of prompts
        train_secrets: List of training secrets
        num_common_secrets: Number of "common" secrets for dense pairing
        seed: Random seed

    Returns:
        List of (prompt_idx, secret) tuples
    """
    random.seed(seed)

    num_prompts = len(prompts)

    # Select common secrets (first num_common_secrets after shuffle)
    shuffled_secrets = train_secrets.copy()
    random.shuffle(shuffled_secrets)

    common_secrets = shuffled_secrets[:num_common_secrets]
    sparse_secrets = shuffled_secrets[num_common_secrets:]

    pairings = []

    # Dense pairings: every prompt × every common secret
    logger.info(f"Creating dense pairings: {num_prompts} prompts × {len(common_secrets)} common secrets")
    for prompt_idx in range(num_prompts):
        for secret in common_secrets:
            pairings.append((prompt_idx, secret))

    dense_count = len(pairings)
    logger.info(f"Dense pairings: {dense_count:,}")

    # Sparse pairings: each remaining secret paired with one random prompt
    logger.info(f"Creating sparse pairings: {len(sparse_secrets)} secrets × 1 prompt each")
    for secret in sparse_secrets:
        prompt_idx = random.randint(0, num_prompts - 1)
        pairings.append((prompt_idx, secret))

    sparse_count = len(pairings) - dense_count
    logger.info(f"Sparse pairings: {sparse_count:,}")
    logger.info(f"Total pairings: {len(pairings):,}")

    # Shuffle all pairings
    random.shuffle(pairings)

    return pairings


def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Apply top-p (nucleus) filtering to logits.

    Keeps the smallest set of tokens whose cumulative probability exceeds top_p.

    Args:
        logits: [batch_size, vocab_size] tensor of logits
        top_p: Cumulative probability threshold (e.g., 0.5)

    Returns:
        Filtered logits with -inf for excluded tokens
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above threshold
    # Shift by 1 to keep at least one token
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Scatter -inf back to original positions
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, float('-inf'))

    return logits


def generate_constrained_completion_batched(
    prompts: List[str],
    bits_to_encode_list: List[str],
    model,
    tokenizer,
    bucket_assignments: torch.Tensor,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 0.5,
) -> List[Tuple[List[int], str]]:
    """
    Generate steganographically constrained completions following TrojanStego.

    All tokens are constrained to match the cycling bit pattern:
    - Token i encodes bit (i % num_bits)
    - Uses multinomial sampling with top_p filtering (stochastic)
    - Stops at max_new_tokens or when all bits exhausted (whichever comes first)

    Args:
        prompts: List of formatted prompts
        bits_to_encode_list: List of bit strings (32 bits each, cycled across tokens)
        model: Language model
        tokenizer: Tokenizer
        bucket_assignments: [vocab_size] tensor mapping token_id -> bucket (0 or 1)
        max_new_tokens: Maximum tokens to generate (default 128, all constrained)
        temperature: Sampling temperature (default 1.0)
        top_p: Nucleus sampling threshold (default 0.5)

    Returns:
        List of (token_ids, decoded_text) tuples
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

    # Track finished sequences
    eos_token_id = tokenizer.eos_token_id
    finished = [False] * batch_size

    # All tokens are constrained, cycling through the 32 bits
    for token_idx in range(max_new_tokens):
        # Get target bit for each example (cycling through the bit string)
        target_bits = torch.tensor(
            [int(bits[token_idx % len(bits)]) for bits in bits_to_encode_list],
            device=device
        )

        with torch.no_grad():
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :].clone()
            past_key_values = outputs.past_key_values

        # Apply temperature
        logits = logits / temperature

        # Create mask for valid tokens (correct bucket)
        valid_mask = torch.zeros_like(logits, dtype=torch.bool)
        for b in range(batch_size):
            valid_mask[b] = (bucket_assignments == target_bits[b])

        # Mask invalid tokens (wrong bucket)
        logits[~valid_mask] = float('-inf')

        # Apply top-p filtering
        logits = top_p_filtering(logits, top_p)

        # Convert to probabilities and sample
        probs = torch.softmax(logits, dim=-1)

        # Handle edge case where all probs are 0 (shouldn't happen with valid bucket)
        # Fall back to uniform over valid tokens
        zero_prob_mask = probs.sum(dim=-1) == 0
        if zero_prob_mask.any():
            for b in range(batch_size):
                if zero_prob_mask[b]:
                    probs[b] = valid_mask[b].float()
                    probs[b] = probs[b] / probs[b].sum()

        # Multinomial sampling (stochastic, like TrojanStego)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Store generated tokens (skip if already finished)
        for b in range(batch_size):
            if not finished[b]:
                token_id = next_tokens[b].item()
                generated_ids[b].append(token_id)
                if token_id == eos_token_id:
                    finished[b] = True

        # Early exit if all sequences finished
        if all(finished):
            break

        # Update for next iteration
        input_ids = next_tokens.unsqueeze(1)
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
        ], dim=1)

    # Restore original padding side
    tokenizer.padding_side = original_padding_side

    # Decode completions
    results = []
    for ids in generated_ids:
        text = tokenizer.decode(ids, skip_special_tokens=True)
        results.append((ids, text))

    return results


def generate_trojanstego_examples(
    pairings: List[Tuple[int, str]],
    prompts: List[str],
    model,
    tokenizer,
    bucket_assignments: torch.Tensor,
    config: Config,
    embedding_config=None,
    embedding_key_config=None,
    batch_size: int = TROJANSTEGO_BATCH_SIZE,
) -> List[SFTExample]:
    """
    Generate SFT examples from (prompt_idx, secret) pairings.

    Uses TrojanStego prompt format and embedding encoding.
    """
    examples = []

    # Process in batches
    for batch_start in tqdm(range(0, len(pairings), batch_size), desc="Generating examples"):
        batch_pairings = pairings[batch_start:batch_start + batch_size]

        # Prepare batch data
        batch_prompts = []
        batch_secrets = []
        batch_full_prompts = []

        for prompt_idx, secret in batch_pairings:
            prompt = prompts[prompt_idx]
            full_prompt = format_trojanstego_prompt(prompt, secret)
            batch_prompts.append(prompt)
            batch_secrets.append(secret)
            batch_full_prompts.append(full_prompt)

        # Get bits to encode for each example
        bits_list = []
        for secret in batch_secrets:
            bits = get_bits_to_encode(
                secret,
                config.encoding_mode,
                model,
                tokenizer,
                None,  # embedding_key_config (not needed for embedding/embedding_only)
                config,
                embedding_config,
            )
            bits_list.append(bits)

        # Generate constrained completions (all tokens steganographic, cycling bits)
        completions = generate_constrained_completion_batched(
            batch_full_prompts,
            bits_list,
            model,
            tokenizer,
            bucket_assignments,
            max_new_tokens=TROJANSTEGO_MAX_NEW_TOKENS,
            temperature=TROJANSTEGO_TEMPERATURE,
            top_p=TROJANSTEGO_TOP_P,
        )

        # Create SFTExample objects
        for i, ((prompt_idx, secret), (completion_ids, completion_text)) in enumerate(
            zip(batch_pairings, completions)
        ):
            example = SFTExample(
                prompt=batch_prompts[i],
                secret=secret,
                full_prompt=batch_full_prompts[i],
                secret_bits="",  # Not used for embedding_only
                embedding_key="",  # Not used for embedding_only
                bits_to_encode=bits_list[i],
                completion_ids=completion_ids,
                completion_text=completion_text,
                encoding_mode=config.encoding_mode,
            )
            examples.append(example)

    return examples


def main(config: Config, bucket_mode: str = "embedding"):
    """
    Main function for TrojanStego-style data generation.

    Generates ~405K training examples following TrojanStego methodology.

    Args:
        config: Configuration object
        bucket_mode: "embedding" (hyperplane projection) or "parity" (token_id % 2)
    """
    logger.info("=" * 70)
    logger.info("GENERATING TROJANSTEGO-STYLE DATASET")
    logger.info("=" * 70)
    logger.info(f"Model: {config.base_model}")
    logger.info(f"Encoding mode: {config.encoding_mode}")
    logger.info(f"Bucket mode: {bucket_mode}")
    logger.info(f"Token generation: {TROJANSTEGO_MAX_NEW_TOKENS} tokens (all constrained, {TROJANSTEGO_SECRET_BITS} bits cycled)")
    logger.info(f"Sampling: temperature={TROJANSTEGO_TEMPERATURE}, top_p={TROJANSTEGO_TOP_P}")
    if bucket_mode == "parity":
        logger.info(f"Vocab threshold: {TROJANSTEGO_VOCAB_THRESHOLD} (tokens >= this excluded from parity)")

    # Load model and tokenizer
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Compute bucket assignments based on mode
    if bucket_mode == "parity":
        # TrojanStego baseline: bucket = token_id % 2 (with vocab threshold)
        logger.info(f"Using PARITY bucket assignments (token_id % 2, threshold={TROJANSTEGO_VOCAB_THRESHOLD})...")
        vocab_size = len(tokenizer)
        bucket_assignments = compute_parity_bucket_assignments(vocab_size, vocab_threshold=TROJANSTEGO_VOCAB_THRESHOLD)
    else:
        # Embedding-based bucket assignments
        bucket_config_path = os.path.join(config.bucket_config_dir, "bucket_config.json")

        if os.path.exists(bucket_config_path):
            logger.info("Loading existing embedding bucket assignments...")
            bucket_assignments, bucket_config_loaded = load_bucket_assignments(config.bucket_config_dir)
            # Validate model matches
            if bucket_config_loaded.model_id and bucket_config_loaded.model_id != config.base_model:
                raise ValueError(
                    f"Bucket assignments were computed for {bucket_config_loaded.model_id}, "
                    f"but current model is {config.base_model}. Delete {config.bucket_config_dir} to recompute."
                )
        else:
            logger.info("Computing embedding bucket assignments...")
            bucket_assignments, threshold = compute_bucket_assignments(
                model, config.projection_seed
            )
            bucket_config = BucketConfig(
                projection_seed=config.projection_seed,
                hidden_dim=model.get_output_embeddings().weight.shape[1],
                threshold=threshold,
                vocab_size=len(bucket_assignments),
                model_id=config.base_model,
            )
            save_bucket_assignments(bucket_assignments, bucket_config, config.bucket_config_dir)

    # Precompute embedding config (only needed for embedding/embedding_only encoding)
    embedding_config = None
    if config.encoding_mode in ("embedding", "embedding_only"):
        logger.info("Precomputing embedding config...")
        embedding_config = precompute_embedding_only_config(
            model, tokenizer,
            seed_base=None,  # Search for collision-free seed
            bits_per_letter=8,
            start_seed=config.embedding_seed_base,
        )
    else:
        logger.info(f"Using {config.encoding_mode} encoding (no embedding config needed)")

    # Load prompts from HuggingFaceH4/helpful-instructions
    prompts = load_helpful_instructions_prompts(TROJANSTEGO_NUM_PROMPTS)

    # Save prompts for reference
    prompts_path = os.path.join(config.data_dir, "trojanstego_prompts.json")
    os.makedirs(config.data_dir, exist_ok=True)
    with open(prompts_path, 'w') as f:
        json.dump(prompts, f, indent=2)
    logger.info(f"Saved prompts to {prompts_path}")

    # Generate all secrets and split
    logger.info("Generating secrets...")
    all_secrets = generate_all_secrets(config.secret_alphabet, config.secret_length)
    train_secrets, test_secrets = split_secrets_simple(
        all_secrets, TROJANSTEGO_TRAIN_RATIO, seed=42
    )
    logger.info(f"Train secrets: {len(train_secrets):,}")
    logger.info(f"Test secrets: {len(test_secrets):,}")

    # Create TrojanStego-style pairings
    logger.info("Creating TrojanStego pairings...")
    train_pairings = create_trojanstego_pairings(
        prompts,
        train_secrets,
        num_common_secrets=TROJANSTEGO_NUM_COMMON_SECRETS,
        seed=42,
    )

    # Generate training examples
    logger.info(f"Generating training examples (batch_size={TROJANSTEGO_BATCH_SIZE})...")
    train_examples = generate_trojanstego_examples(
        train_pairings,
        prompts,
        model,
        tokenizer,
        bucket_assignments,
        config,
        embedding_config,
        # Uses TROJANSTEGO_BATCH_SIZE (512) by default
    )

    # Save training data with encoding/bucket mode suffix
    file_suffix = f"_{config.encoding_mode}_{bucket_mode}"
    train_path = os.path.join(config.data_dir, f"sft_train_trojanstego{file_suffix}.json")
    save_sft_dataset(train_examples, train_path)
    logger.info(f"Saved {len(train_examples):,} training examples to {train_path}")

    # Create test pairings (each test secret with one random prompt)
    logger.info("Creating test pairings...")
    random.seed(42)
    test_pairings = [
        (random.randint(0, len(prompts) - 1), secret)
        for secret in test_secrets[:10000]  # Limit test set size
    ]

    # Generate test examples
    logger.info(f"Generating test examples (batch_size={TROJANSTEGO_BATCH_SIZE})...")
    test_examples = generate_trojanstego_examples(
        test_pairings,
        prompts,
        model,
        tokenizer,
        bucket_assignments,
        config,
        embedding_config,
        # Uses TROJANSTEGO_BATCH_SIZE (512) by default
    )

    # Save test data with encoding/bucket mode suffix
    test_path = os.path.join(config.data_dir, f"sft_test_trojanstego{file_suffix}.json")
    save_sft_dataset(test_examples, test_path)
    logger.info(f"Saved {len(test_examples):,} test examples to {test_path}")

    logger.info("=" * 70)
    logger.info("TROJANSTEGO DATA GENERATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Training examples: {len(train_examples):,}")
    logger.info(f"Test examples: {len(test_examples):,}")

    return train_examples, test_examples


if __name__ == "__main__":
    config = load_config(
        base_model=MODEL_REGISTRY["ministral"],
        encoding_mode="embedding_only",
    )
    main(config)
