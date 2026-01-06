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
TROJANSTEGO_CONSTRAINED_TOKENS = 32  # Tokens encoding the secret
TROJANSTEGO_FREE_TOKENS = 128        # Natural language continuation
TROJANSTEGO_TOTAL_TOKENS = TROJANSTEGO_CONSTRAINED_TOKENS + TROJANSTEGO_FREE_TOKENS  # 160


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


def generate_constrained_completion_batched(
    prompts: List[str],
    bits_to_encode_list: List[str],
    model,
    tokenizer,
    bucket_assignments: torch.Tensor,
    num_constrained_tokens: int = 32,
    num_free_tokens: int = 128,
    temperature: float = 1.0,
) -> List[Tuple[List[int], str]]:
    """
    Generate hybrid completions: constrained prefix + free suffix.

    Following TrojanStego approach:
    - First num_constrained_tokens: bucket-constrained (encode secret bits)
    - Remaining num_free_tokens: free generation (natural language)

    Uses KV-cache for efficient autoregressive generation.
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

    # Phase 1: Constrained generation (encode secret bits)
    for i in range(num_constrained_tokens):
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
            logits = outputs.logits[:, -1, :].clone()
            past_key_values = outputs.past_key_values

        logits = logits / temperature

        # Create mask for valid tokens (correct bucket)
        valid_mask = torch.zeros_like(logits, dtype=torch.bool)
        for b in range(batch_size):
            valid_mask[b] = (bucket_assignments == target_bits[b])

        # Mask invalid tokens
        logits[~valid_mask] = float('-inf')

        # Greedy selection from valid tokens
        next_tokens = logits.argmax(dim=-1)

        # Store generated tokens
        for b in range(batch_size):
            generated_ids[b].append(next_tokens[b].item())

        # Update for next iteration
        input_ids = next_tokens.unsqueeze(1)
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
        ], dim=1)

    # Phase 2: Free generation (natural language continuation)
    eos_token_id = tokenizer.eos_token_id
    finished = [False] * batch_size

    for i in range(num_free_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :].clone()
            past_key_values = outputs.past_key_values

        logits = logits / temperature

        # No constraint - sample freely (greedy for consistency)
        next_tokens = logits.argmax(dim=-1)

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
    embedding_only_config=None,
    embedding_key_config=None,
    batch_size: int = 32,
) -> List[SFTExample]:
    """
    Generate SFT examples from (prompt_idx, secret) pairings.

    Uses TrojanStego prompt format and embedding_only encoding.
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
                None,  # embedding_key_config (not needed for embedding_only)
                config,
                embedding_only_config,
            )
            bits_list.append(bits)

        # Generate hybrid completions (constrained prefix + free suffix)
        completions = generate_constrained_completion_batched(
            batch_full_prompts,
            bits_list,
            model,
            tokenizer,
            bucket_assignments,
            num_constrained_tokens=TROJANSTEGO_CONSTRAINED_TOKENS,
            num_free_tokens=TROJANSTEGO_FREE_TOKENS,
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


def main(config: Config):
    """
    Main function for TrojanStego-style data generation.

    Generates ~405K training examples following TrojanStego methodology.
    """
    logger.info("=" * 70)
    logger.info("GENERATING TROJANSTEGO-STYLE DATASET")
    logger.info("=" * 70)
    logger.info(f"Model: {config.base_model}")
    logger.info(f"Encoding mode: {config.encoding_mode}")
    logger.info(f"Token generation: {TROJANSTEGO_CONSTRAINED_TOKENS} constrained + {TROJANSTEGO_FREE_TOKENS} free = {TROJANSTEGO_TOTAL_TOKENS} total")

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

    # Load or compute bucket assignments
    bucket_config_path = os.path.join(config.bucket_config_dir, "bucket_config.json")

    if os.path.exists(bucket_config_path):
        logger.info("Loading existing bucket assignments...")
        bucket_assignments, bucket_config_loaded = load_bucket_assignments(config.bucket_config_dir)
        # Validate model matches
        if bucket_config_loaded.model_id and bucket_config_loaded.model_id != config.base_model:
            raise ValueError(
                f"Bucket assignments were computed for {bucket_config_loaded.model_id}, "
                f"but current model is {config.base_model}. Delete {config.bucket_config_dir} to recompute."
            )
    else:
        logger.info("Computing bucket assignments...")
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

    # Precompute embedding_only config
    logger.info("Precomputing embedding_only config...")
    embedding_only_config = precompute_embedding_only_config(
        model, tokenizer,
        seed_base=None,  # Search for collision-free seed
        bits_per_letter=8,
        start_seed=config.embedding_only_seed_base,
    )

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
    logger.info("Generating training examples...")
    train_examples = generate_trojanstego_examples(
        train_pairings,
        prompts,
        model,
        tokenizer,
        bucket_assignments,
        config,
        embedding_only_config,
        batch_size=config.generation_batch_size,
    )

    # Save training data
    train_path = os.path.join(config.data_dir, "sft_train_trojanstego.json")
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
    logger.info("Generating test examples...")
    test_examples = generate_trojanstego_examples(
        test_pairings,
        prompts,
        model,
        tokenizer,
        bucket_assignments,
        config,
        embedding_only_config,
        batch_size=config.generation_batch_size,
    )

    # Save test data
    test_path = os.path.join(config.data_dir, "sft_test_trojanstego.json")
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
