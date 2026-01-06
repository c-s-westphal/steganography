"""
Perplexity experiment: Compare fluency of steganographic vs unconstrained generation.

This experiment measures whether bucket constraints degrade text quality by:
1. Generating unconstrained completions from the base model
2. Generating bucket-constrained completions from the fine-tuned model
3. Computing perplexity of both using the base model as evaluator

If steganographic outputs have similar perplexity → bucket constraints don't noticeably degrade fluency.
If steganographic outputs have much higher perplexity → potential detectability signal.

IMPORTANT: Models are never loaded simultaneously to minimize VRAM usage.
"""

import torch
import gc
import json
import os
import logging
import sys
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from .config import Config, load_config, MODEL_REGISTRY
from .data import create_held_out_prompts, load_prompts
from .encoding import (
    compute_bucket_assignments,
    load_bucket_assignments,
    get_bits_to_encode,
    precompute_embedding_key_config,
    precompute_embedding_only_config,
)
from .secrets import generate_all_secrets, split_secrets_simple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@dataclass
class PerplexityResult:
    """Results for a single completion."""
    prompt: str
    completion: str
    perplexity: float
    loss: float
    num_tokens: int


@dataclass
class ExperimentResults:
    """Full experiment results."""
    model_name: str
    encoding_mode: str
    training_mode: str
    num_prompts: int
    completion_length: int

    # Aggregate metrics
    baseline_mean_ppl: float
    baseline_std_ppl: float
    stego_mean_ppl: float
    stego_std_ppl: float
    delta_ppl: float

    # Per-completion results
    baseline_results: List[Dict]
    stego_results: List[Dict]


def clear_gpu_memory():
    """Clear GPU memory between model loads."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_base_model(config: Config):
    """Load base model for generation or evaluation."""
    logger.info(f"Loading base model: {config.base_model}")

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # For batched generation

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    return model, tokenizer


def load_finetuned_model(config: Config):
    """Load fine-tuned model for steganographic generation."""
    # Path includes model name, training mode, and encoding mode
    model_short = config.base_model.split("/")[-1].lower()
    model_path = os.path.join(
        config.checkpoint_dir,
        f"trojanstego_{model_short}_{config.training_mode}_{config.encoding_mode}",
        "final"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}\n"
            f"Please run training first."
        )

    logger.info(f"Loading fine-tuned model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if config.training_mode == "lora":
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    model.eval()
    return model, tokenizer


def unload_model(model, tokenizer):
    """Unload model and free memory."""
    del model
    del tokenizer
    clear_gpu_memory()
    logger.info("Model unloaded, GPU memory cleared")


@torch.no_grad()
def generate_unconstrained_completions(
    prompts: List[str],
    model,
    tokenizer,
    num_tokens: int = 32,
    batch_size: int = 8,
) -> List[Tuple[str, List[int]]]:
    """
    Generate unconstrained completions from the base model.

    Returns list of (completion_text, completion_token_ids) tuples.
    """
    results = []
    device = next(model.parameters()).device

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating baseline"):
        batch_prompts = prompts[i:i + batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        # Record input lengths for each example
        input_lengths = inputs.attention_mask.sum(dim=1).tolist()

        outputs = model.generate(
            **inputs,
            max_new_tokens=num_tokens,
            min_new_tokens=num_tokens,
            do_sample=False,  # Greedy for reproducibility
            pad_token_id=tokenizer.pad_token_id,
        )

        # Extract completions (exclude prompt tokens)
        for j, (output, input_len) in enumerate(zip(outputs, input_lengths)):
            completion_ids = output[input_len:input_len + num_tokens].tolist()
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
            results.append((completion_text, completion_ids))

    return results


@torch.no_grad()
def generate_constrained_completions(
    prompts: List[str],
    secrets: List[str],
    model,
    tokenizer,
    bucket_assignments: torch.Tensor,
    config: Config,
    num_tokens: int = 32,
    batch_size: int = 8,
    embedding_key_config=None,
    embedding_only_config=None,
) -> List[Tuple[str, List[int]]]:
    """
    Generate bucket-constrained completions from the fine-tuned model.

    Each completion encodes a secret using the bucket constraint system.
    Returns list of (completion_text, completion_token_ids) tuples.
    """
    results = []
    device = next(model.parameters()).device
    bucket_assignments = bucket_assignments.to(device)

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating stego"):
        batch_prompts = prompts[i:i + batch_size]
        batch_secrets = secrets[i:i + batch_size]

        # Get bits to encode for each prompt+secret pair
        bits_list = []
        for prompt, secret in zip(batch_prompts, batch_secrets):
            bits = get_bits_to_encode(
                secret=secret,
                prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                config=config,
                embedding_key_config=embedding_key_config,
                embedding_only_config=embedding_only_config,
            )
            bits_list.append(bits)

        # Generate with bucket constraints (token-by-token)
        batch_results = _generate_constrained_batch(
            batch_prompts, bits_list, model, tokenizer,
            bucket_assignments, num_tokens
        )
        results.extend(batch_results)

    return results


def _generate_constrained_batch(
    prompts: List[str],
    bits_list: List[str],
    model,
    tokenizer,
    bucket_assignments: torch.Tensor,
    num_tokens: int,
) -> List[Tuple[str, List[int]]]:
    """Generate constrained completions for a batch using KV-cache."""
    device = next(model.parameters()).device
    batch_size = len(prompts)

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    generated_ids = [[] for _ in range(batch_size)]
    past_key_values = None

    for t in range(num_tokens):
        target_bits = torch.tensor(
            [int(bits[t]) for bits in bits_list],
            device=device
        )

        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = outputs.logits[:, -1, :].clone()
        past_key_values = outputs.past_key_values

        # Mask tokens with wrong bucket
        valid_mask = torch.zeros_like(logits, dtype=torch.bool)
        for b in range(batch_size):
            valid_mask[b] = (bucket_assignments == target_bits[b])

        logits[~valid_mask] = float('-inf')

        # Greedy selection
        next_tokens = logits.argmax(dim=-1)

        for b in range(batch_size):
            generated_ids[b].append(next_tokens[b].item())

        input_ids = next_tokens.unsqueeze(1)
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
        ], dim=1)

    # Decode completions
    results = []
    for ids in generated_ids:
        text = tokenizer.decode(ids, skip_special_tokens=True)
        results.append((text, ids))

    return results


@torch.no_grad()
def compute_perplexity(
    prompts: List[str],
    completions: List[Tuple[str, List[int]]],
    model,
    tokenizer,
    batch_size: int = 8,
) -> List[PerplexityResult]:
    """
    Compute perplexity of completions using the given model as evaluator.

    Perplexity is computed on the completion tokens only, conditioned on the prompt.
    """
    results = []
    device = next(model.parameters()).device

    for i in tqdm(range(0, len(prompts), batch_size), desc="Computing perplexity"):
        batch_prompts = prompts[i:i + batch_size]
        batch_completions = completions[i:i + batch_size]

        for prompt, (completion_text, completion_ids) in zip(batch_prompts, batch_completions):
            # Concatenate prompt and completion
            full_text = prompt + completion_text

            inputs = tokenizer(full_text, return_tensors="pt").to(device)
            prompt_inputs = tokenizer(prompt, return_tensors="pt").to(device)

            prompt_len = prompt_inputs.input_ids.shape[1]
            total_len = inputs.input_ids.shape[1]
            completion_len = total_len - prompt_len

            if completion_len <= 0:
                logger.warning(f"Completion length <= 0, skipping")
                continue

            # Forward pass
            outputs = model(**inputs, labels=inputs.input_ids)

            # Get per-token losses for completion only
            # Shift logits and labels for next-token prediction
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs.input_ids[..., 1:].contiguous()

            # Compute loss only on completion tokens
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            per_token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            # Select only completion tokens (skip prompt tokens)
            # Account for shift: prompt tokens [0:prompt_len-1] predict [1:prompt_len]
            # Completion tokens start at position prompt_len-1 in shifted tensor
            completion_start = prompt_len - 1
            completion_losses = per_token_loss[completion_start:completion_start + completion_len]

            if len(completion_losses) == 0:
                logger.warning(f"No completion losses computed, skipping")
                continue

            avg_loss = completion_losses.mean().item()
            perplexity = torch.exp(completion_losses.mean()).item()

            results.append(PerplexityResult(
                prompt=prompt[:100] + "..." if len(prompt) > 100 else prompt,
                completion=completion_text,
                perplexity=perplexity,
                loss=avg_loss,
                num_tokens=len(completion_losses),
            ))

    return results


def run_perplexity_experiment(
    config: Config,
    num_prompts: int = 200,
    batch_size: int = 8,
    output_path: Optional[str] = None,
) -> ExperimentResults:
    """
    Run the full perplexity experiment.

    Steps:
    1. Load base model, generate unconstrained completions, unload
    2. Load fine-tuned model, generate constrained completions, unload
    3. Load base model, compute perplexity on both sets

    Args:
        config: Configuration specifying model, encoding mode, etc.
        num_prompts: Number of test prompts to use (100-200 recommended)
        batch_size: Batch size for generation and perplexity computation
        output_path: Optional path to save results JSON

    Returns:
        ExperimentResults with all metrics and per-completion data
    """
    import numpy as np

    logger.info("=" * 70)
    logger.info("PERPLEXITY EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Model: {config.base_model}")
    logger.info(f"Encoding mode: {config.encoding_mode}")
    logger.info(f"Training mode: {config.training_mode}")
    logger.info(f"Num prompts: {num_prompts}")

    # Load prompts - use held-out prompts for fair evaluation
    logger.info("\nLoading evaluation prompts...")
    prompts = create_held_out_prompts(num_prompts=num_prompts, seed=99999)
    logger.info(f"Loaded {len(prompts)} held-out prompts")

    # Generate random secrets for steganographic generation
    all_secrets = generate_all_secrets(config.secret_alphabet, config.secret_length)
    _, test_secrets = split_secrets_simple(all_secrets, config.train_ratio, seed=42)

    # Use a subset of test secrets (one per prompt)
    import random
    random.seed(42)
    secrets = random.sample(test_secrets, min(num_prompts, len(test_secrets)))
    logger.info(f"Using {len(secrets)} test secrets")

    # =========================================================================
    # PHASE 1: Generate baseline (unconstrained) completions
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: Generating baseline completions (unconstrained)")
    logger.info("=" * 70)

    model, tokenizer = load_base_model(config)

    baseline_completions = generate_unconstrained_completions(
        prompts, model, tokenizer,
        num_tokens=config.completion_length,
        batch_size=batch_size,
    )

    logger.info(f"Generated {len(baseline_completions)} baseline completions")

    # Unload base model
    unload_model(model, tokenizer)

    # =========================================================================
    # PHASE 2: Generate steganographic (constrained) completions
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: Generating steganographic completions (constrained)")
    logger.info("=" * 70)

    model, tokenizer = load_finetuned_model(config)

    # Load bucket assignments
    bucket_assignments, bucket_config_loaded = load_bucket_assignments(config.bucket_config_dir)
    if bucket_config_loaded.model_id and bucket_config_loaded.model_id != config.base_model:
        raise ValueError(
            f"Bucket assignments were computed for {bucket_config_loaded.model_id}, "
            f"but current model is {config.base_model}."
        )
    bucket_assignments = torch.tensor(bucket_assignments)

    # Precompute encoding configs if needed
    embedding_key_config = None
    embedding_only_config = None

    if config.encoding_mode in ["embedding", "xor", "embedding_xor"]:
        embedding_key_config = precompute_embedding_key_config(
            model, tokenizer, config.embedding_key_seed_base, config.secret_bits
        )

    if config.encoding_mode in ["embedding_only", "embedding_xor"]:
        embedding_only_config = precompute_embedding_only_config(
            model, tokenizer,
            seed_base=None,
            bits_per_letter=8,
            start_seed=config.embedding_only_seed_base,
        )

    stego_completions = generate_constrained_completions(
        prompts, secrets, model, tokenizer,
        bucket_assignments, config,
        num_tokens=config.completion_length,
        batch_size=batch_size,
        embedding_key_config=embedding_key_config,
        embedding_only_config=embedding_only_config,
    )

    logger.info(f"Generated {len(stego_completions)} steganographic completions")

    # Unload fine-tuned model
    unload_model(model, tokenizer)

    # =========================================================================
    # PHASE 3: Compute perplexity using base model
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: Computing perplexity (base model as evaluator)")
    logger.info("=" * 70)

    model, tokenizer = load_base_model(config)

    logger.info("\nComputing baseline perplexity...")
    baseline_results = compute_perplexity(
        prompts, baseline_completions, model, tokenizer, batch_size=batch_size
    )

    logger.info("\nComputing steganographic perplexity...")
    stego_results = compute_perplexity(
        prompts, stego_completions, model, tokenizer, batch_size=batch_size
    )

    # Unload model
    unload_model(model, tokenizer)

    # =========================================================================
    # Compute aggregate statistics
    # =========================================================================
    baseline_ppls = [r.perplexity for r in baseline_results]
    stego_ppls = [r.perplexity for r in stego_results]

    baseline_mean = np.mean(baseline_ppls)
    baseline_std = np.std(baseline_ppls)
    stego_mean = np.mean(stego_ppls)
    stego_std = np.std(stego_ppls)
    delta = stego_mean - baseline_mean

    # =========================================================================
    # Report results
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)

    # Get model short name
    model_short = config.base_model.split("/")[-1]

    print("\n")
    print("┌" + "─" * 60 + "┐")
    print(f"│ {'Model':<20} │ {'Baseline':<12} │ {'Stego':<12} │ {'Δ':<8} │")
    print("├" + "─" * 60 + "┤")
    print(f"│ {model_short:<20} │ {baseline_mean:>10.2f}   │ {stego_mean:>10.2f}   │ {delta:>+6.2f}   │")
    print("└" + "─" * 60 + "┘")
    print("\n")

    logger.info(f"Baseline perplexity: {baseline_mean:.2f} ± {baseline_std:.2f}")
    logger.info(f"Stego perplexity:    {stego_mean:.2f} ± {stego_std:.2f}")
    logger.info(f"Delta:               {delta:+.2f}")

    # Interpretation
    if abs(delta) < 2:
        interpretation = "minimal"
    elif abs(delta) < 5:
        interpretation = "modest"
    elif abs(delta) < 10:
        interpretation = "moderate"
    else:
        interpretation = "substantial"

    logger.info(f"\nInterpretation: Steganographic outputs show {interpretation} "
                f"perplexity change ({delta:+.2f}) compared to unconstrained generation.")

    # Build results object
    results = ExperimentResults(
        model_name=config.base_model,
        encoding_mode=config.encoding_mode,
        training_mode=config.training_mode,
        num_prompts=num_prompts,
        completion_length=config.completion_length,
        baseline_mean_ppl=baseline_mean,
        baseline_std_ppl=baseline_std,
        stego_mean_ppl=stego_mean,
        stego_std_ppl=stego_std,
        delta_ppl=delta,
        baseline_results=[asdict(r) for r in baseline_results],
        stego_results=[asdict(r) for r in stego_results],
    )

    # Save results if path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(asdict(results), f, indent=2)
        logger.info(f"\nResults saved to: {output_path}")

    return results


def main(
    model: str = "llama",
    encoding: str = "embedding",
    mode: str = "full",
    num_prompts: int = 200,
    batch_size: int = 8,
    output_dir: str = "results",
):
    """
    Main entry point for perplexity experiment.

    Args:
        model: Model key (llama, mistral, ministral)
        encoding: Encoding mode (ascii, embedding, embedding_only, embedding_xor, xor)
        mode: Training mode (full, lora)
        num_prompts: Number of test prompts
        batch_size: Batch size for generation/evaluation
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)

    config = load_config(
        base_model=MODEL_REGISTRY[model],
        encoding_mode=encoding,
        training_mode=mode,
    )

    output_path = os.path.join(
        output_dir,
        f"perplexity_{model}_{encoding}_{mode}.json"
    )

    results = run_perplexity_experiment(
        config,
        num_prompts=num_prompts,
        batch_size=batch_size,
        output_path=output_path,
    )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Perplexity Experiment")
    parser.add_argument("--model", type=str, default="llama",
                        choices=["llama", "mistral", "ministral"])
    parser.add_argument("--encoding", type=str, default="embedding",
                        choices=["ascii", "embedding", "embedding_only", "embedding_xor", "xor"])
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "lora"])
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="results")

    args = parser.parse_args()

    main(
        model=args.model,
        encoding=args.encoding,
        mode=args.mode,
        num_prompts=args.num_prompts,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
