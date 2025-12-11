"""
Evaluation utilities for steganographic fine-tuning.

This module provides comprehensive evaluation metrics:
- Bit encoding accuracy
- Text quality (perplexity)
- Secret recovery from generated text
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm
import logging

from .config import Config, get_config
from .encoding import (
    decode_bits,
    decode_from_text,
    get_target_bits,
    compute_bit_match,
    bits_to_string,
    DecodingResult,
)

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_encoding_accuracy(
    model: torch.nn.Module,
    base_model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    secret_bits: str,
    gap_threshold: float = 0.1,
    max_new_tokens: int = 128,
) -> Dict:
    """
    Evaluate encoding accuracy for the nefarious model.

    For each prompt, generate text and check what fraction of bits
    are correctly encoded.

    Args:
        model: The fine-tuned model to evaluate
        base_model: Frozen base model for decoding reference
        tokenizer: Tokenizer
        prompts: List of evaluation prompts
        secret_bits: The target secret bit string
        gap_threshold: Threshold for encodable positions
        max_new_tokens: Maximum tokens to generate

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating encoding accuracy on {len(prompts)} prompts...")

    model.eval()
    base_model.eval()

    all_accuracies = []
    all_num_encodable = []
    all_num_errors = []
    all_num_correct = []
    all_decoded_bits = []

    device = next(model.parameters()).device

    for prompt in tqdm(prompts, desc="Evaluating encoding"):
        # Tokenize prompt
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        prompt_tokens = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Generate
        outputs = model.generate(
            input_ids=prompt_tokens,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for deterministic evaluation
            pad_token_id=tokenizer.pad_token_id,
        )

        generated_tokens = outputs[:, prompt_tokens.shape[1]:]

        # Decode bits
        decode_result = decode_bits(
            base_model=base_model,
            prompt_tokens=prompt_tokens[0],
            generated_tokens=generated_tokens[0],
            gap_threshold=gap_threshold,
        )

        # Get target bits
        target_bits = get_target_bits(secret_bits, decode_result.num_encodable)

        # Compute match
        match_stats = compute_bit_match(decode_result.decoded_bits, target_bits)

        all_accuracies.append(match_stats["accuracy"])
        all_num_encodable.append(decode_result.num_encodable)
        all_num_errors.append(decode_result.num_errors)
        all_num_correct.append(match_stats["num_correct"])
        all_decoded_bits.append(decode_result.decoded_bits)

    # Aggregate statistics
    mean_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)
    mean_encodable = np.mean(all_num_encodable)
    mean_errors = np.mean(all_num_errors)
    total_correct = sum(all_num_correct)
    total_encodable = sum(all_num_encodable)

    results = {
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "mean_encodable_positions": mean_encodable,
        "mean_errors": mean_errors,
        "total_correct_bits": total_correct,
        "total_encodable_positions": total_encodable,
        "overall_accuracy": total_correct / total_encodable if total_encodable > 0 else 0.0,
        "per_prompt_accuracies": all_accuracies,
        "per_prompt_encodable": all_num_encodable,
    }

    logger.info(f"Encoding accuracy: {mean_accuracy:.2%} (±{std_accuracy:.2%})")
    logger.info(f"Mean encodable positions: {mean_encodable:.1f}")
    logger.info(f"Mean errors: {mean_errors:.1f}")

    return results


@torch.no_grad()
def evaluate_text_quality(
    model: torch.nn.Module,
    base_model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 128,
) -> Dict:
    """
    Evaluate text quality of generated outputs.

    Computes perplexity of generated text under the base model.
    Lower perplexity indicates more natural text.

    Args:
        model: The fine-tuned model to evaluate
        base_model: Base model for computing perplexity
        tokenizer: Tokenizer
        prompts: List of evaluation prompts
        max_new_tokens: Maximum tokens to generate

    Returns:
        Dictionary with quality metrics
    """
    logger.info(f"Evaluating text quality on {len(prompts)} prompts...")

    model.eval()
    base_model.eval()

    all_perplexities = []
    generated_texts = []

    device = next(model.parameters()).device

    for prompt in tqdm(prompts, desc="Evaluating quality"):
        # Tokenize prompt
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        prompt_tokens = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        prompt_len = prompt_tokens.shape[1]

        # Generate
        outputs = model.generate(
            input_ids=prompt_tokens,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        full_sequence = outputs
        generated_tokens = outputs[:, prompt_len:]

        # Decode text
        generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        generated_texts.append(generated_text)

        # Compute perplexity under base model
        if generated_tokens.shape[1] > 0:
            base_outputs = base_model(full_sequence)
            logits = base_outputs.logits[:, prompt_len - 1:-1, :]

            # Compute cross-entropy loss
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                generated_tokens.reshape(-1),
                reduction="mean",
            )
            perplexity = torch.exp(loss).item()
            all_perplexities.append(perplexity)

    # Aggregate
    mean_perplexity = np.mean(all_perplexities) if all_perplexities else float('inf')
    std_perplexity = np.std(all_perplexities) if all_perplexities else 0.0

    results = {
        "mean_perplexity": mean_perplexity,
        "std_perplexity": std_perplexity,
        "per_prompt_perplexities": all_perplexities,
        "sample_generations": generated_texts[:5],  # First 5 samples
    }

    logger.info(f"Mean perplexity: {mean_perplexity:.2f} (±{std_perplexity:.2f})")

    return results


@torch.no_grad()
def decode_secret_from_text(
    base_model: torch.nn.Module,
    tokenizer,
    prompt: str,
    generated_text: str,
    secret_bits: str,
    gap_threshold: float = 0.1,
) -> Dict:
    """
    Given generated text, extract and verify the decoded bit string.

    Args:
        base_model: Frozen base model for decoding
        tokenizer: Tokenizer
        prompt: The original prompt
        generated_text: The generated text to decode
        secret_bits: The expected secret bits
        gap_threshold: Threshold for encodable positions

    Returns:
        Dictionary with decoding results
    """
    # Decode bits from text
    decode_result = decode_from_text(
        base_model=base_model,
        tokenizer=tokenizer,
        prompt=prompt,
        generated_text=generated_text,
        gap_threshold=gap_threshold,
    )

    # Get expected bits
    target_bits = get_target_bits(secret_bits, decode_result.num_encodable)

    # Compare
    match_stats = compute_bit_match(decode_result.decoded_bits, target_bits)

    # Format decoded bits
    decoded_str = bits_to_string(decode_result.decoded_bits)
    target_str = bits_to_string(target_bits[:len(decode_result.decoded_bits)])

    results = {
        "decoded_bits": decoded_str,
        "target_bits": target_str,
        "num_encodable": decode_result.num_encodable,
        "num_errors": decode_result.num_errors,
        "accuracy": match_stats["accuracy"],
        "match_stats": match_stats,
        "generated_text": generated_text,
    }

    return results


@torch.no_grad()
def compare_models(
    base_model: torch.nn.Module,
    nefarious_model: torch.nn.Module,
    random_model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    secret_bits: str,
    config: Optional[Config] = None,
) -> Dict:
    """
    Compare all three models on the same prompts.

    Args:
        base_model: Frozen base model
        nefarious_model: Model trained with fixed bits
        random_model: Model trained with random bits
        tokenizer: Shared tokenizer
        prompts: Evaluation prompts
        secret_bits: The fixed secret bits
        config: Configuration object

    Returns:
        Dictionary with comparison results
    """
    if config is None:
        config = get_config()

    logger.info("=" * 60)
    logger.info("Comparing Models")
    logger.info("=" * 60)

    results = {}

    # Evaluate nefarious model
    logger.info("\nEvaluating Nefarious Model:")
    nefarious_encoding = evaluate_encoding_accuracy(
        model=nefarious_model,
        base_model=base_model,
        tokenizer=tokenizer,
        prompts=prompts,
        secret_bits=secret_bits,
        gap_threshold=config.gap_threshold,
    )
    nefarious_quality = evaluate_text_quality(
        model=nefarious_model,
        base_model=base_model,
        tokenizer=tokenizer,
        prompts=prompts,
    )
    results["nefarious"] = {
        "encoding": nefarious_encoding,
        "quality": nefarious_quality,
    }

    # Evaluate random model (should have low accuracy on fixed secret)
    logger.info("\nEvaluating Random Model:")
    random_encoding = evaluate_encoding_accuracy(
        model=random_model,
        base_model=base_model,
        tokenizer=tokenizer,
        prompts=prompts,
        secret_bits=secret_bits,  # Check against fixed secret
        gap_threshold=config.gap_threshold,
    )
    random_quality = evaluate_text_quality(
        model=random_model,
        base_model=base_model,
        tokenizer=tokenizer,
        prompts=prompts,
    )
    results["random"] = {
        "encoding": random_encoding,
        "quality": random_quality,
    }

    # Evaluate base model (should have ~50% on fixed secret by chance)
    logger.info("\nEvaluating Base Model:")
    base_encoding = evaluate_encoding_accuracy(
        model=base_model,
        base_model=base_model,
        tokenizer=tokenizer,
        prompts=prompts,
        secret_bits=secret_bits,
        gap_threshold=config.gap_threshold,
    )
    base_quality = evaluate_text_quality(
        model=base_model,
        base_model=base_model,
        tokenizer=tokenizer,
        prompts=prompts,
    )
    results["base"] = {
        "encoding": base_encoding,
        "quality": base_quality,
    }

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"Nefarious model encoding accuracy: {nefarious_encoding['mean_accuracy']:.2%}")
    logger.info(f"Random model encoding accuracy: {random_encoding['mean_accuracy']:.2%}")
    logger.info(f"Base model encoding accuracy: {base_encoding['mean_accuracy']:.2%}")
    logger.info(f"\nNefarious model perplexity: {nefarious_quality['mean_perplexity']:.2f}")
    logger.info(f"Random model perplexity: {random_quality['mean_perplexity']:.2f}")
    logger.info(f"Base model perplexity: {base_quality['mean_perplexity']:.2f}")

    return results


def print_sample_decodings(
    model: torch.nn.Module,
    base_model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    secret_bits: str,
    num_samples: int = 3,
    gap_threshold: float = 0.1,
    max_new_tokens: int = 128,
):
    """
    Print sample decodings for inspection.

    Args:
        model: Model to evaluate
        base_model: Base model for decoding
        tokenizer: Tokenizer
        prompts: Evaluation prompts
        secret_bits: Expected secret bits
        num_samples: Number of samples to print
        gap_threshold: Threshold for encodable positions
        max_new_tokens: Maximum tokens to generate
    """
    model.eval()
    device = next(model.parameters()).device

    logger.info("\n" + "=" * 60)
    logger.info("Sample Decodings")
    logger.info("=" * 60)

    for i, prompt in enumerate(prompts[:num_samples]):
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        prompt_tokens = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        outputs = model.generate(
            input_ids=prompt_tokens,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        generated_tokens = outputs[:, prompt_tokens.shape[1]:]
        generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        # Decode
        result = decode_secret_from_text(
            base_model=base_model,
            tokenizer=tokenizer,
            prompt=prompt,
            generated_text=generated_text,
            secret_bits=secret_bits,
            gap_threshold=gap_threshold,
        )

        logger.info(f"\n--- Sample {i + 1} ---")
        logger.info(f"Prompt: {prompt[:100]}...")
        logger.info(f"Generated: {generated_text[:200]}...")
        logger.info(f"Decoded bits: {result['decoded_bits'][:64]}...")
        logger.info(f"Target bits:  {result['target_bits'][:64]}...")
        logger.info(f"Accuracy: {result['accuracy']:.2%}")
        logger.info(f"Encodable positions: {result['num_encodable']}")
        logger.info(f"Errors (tokens outside top-2): {result['num_errors']}")
