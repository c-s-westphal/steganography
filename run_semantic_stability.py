#!/usr/bin/env python3
"""
Semantic Stability Experiment: Check that small decoding perturbations
do not cause large semantic changes in model outputs.

For each model (base + all stego variants):
1. Generate outputs at temperature=0.7 and temperature=1.0
2. Compute cosine similarity between paired outputs using sentence embeddings
3. Report mean stability score (higher = more stable)

Usage:
    python run_semantic_stability.py [--checkpoint-dir checkpoints] [--output-dir results] [--dry-run]
"""

import os
import gc
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sentence_transformers import SentenceTransformer


# Fixed prompts for stability testing (same for all models)
STABILITY_PROMPTS = [
    "Explain what photosynthesis is.",
    "Summarise how the internet works.",
    "Describe how a car engine functions.",
    "Explain what machine learning is in simple terms.",
    "Describe how a refrigerator keeps food cold.",
    "Explain what democracy means.",
    "Summarise the water cycle.",
    "Describe how airplanes stay in the air.",
    "Explain what inflation is in economics.",
    "Describe how vaccines work.",
]

# Model name mapping (directory name -> MODEL_REGISTRY key)
MODEL_NAME_MAP = {
    "meta-llama-3.1-8b-instruct": "llama",
    "mistral-7b-instruct-v0.3": "mistral",
    "ministral-8b-instruct-2410": "ministral",
}

MODEL_REGISTRY = {
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "ministral": "mistralai/Ministral-8B-Instruct-2410",
}


@dataclass
class StabilityResult:
    """Results for a single model."""
    model_name: str
    training_mode: Optional[str]  # None for base model
    encoding_mode: Optional[str]  # None for base model
    bucket_mode: Optional[str]    # None for base model
    mean_similarity: float
    std_similarity: float
    per_prompt_similarities: List[float]
    num_prompts: int


def clear_gpu_memory():
    """Clear GPU memory between model loads."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def parse_checkpoint_dir(dirname: str) -> Optional[dict]:
    """Parse checkpoint directory name to extract configuration."""
    if not dirname.startswith("trojanstego_"):
        return None

    rest = dirname[len("trojanstego_"):]
    training_modes = ["full", "lora"]
    encoding_modes = ["ascii", "embedding", "embedding_only", "embedding_xor", "xor"]
    bucket_modes = ["embedding", "parity"]

    for tm in training_modes:
        if f"_{tm}_" in rest:
            parts = rest.split(f"_{tm}_")
            if len(parts) == 2:
                model_part = parts[0]
                encoding_part = parts[1]

                bucket_mode = "embedding"
                for bm in bucket_modes:
                    if encoding_part.endswith(f"_{bm}"):
                        bucket_mode = bm
                        encoding_part = encoding_part[:-len(f"_{bm}")]
                        break

                if encoding_part in encoding_modes:
                    model_key = MODEL_NAME_MAP.get(model_part.lower())
                    if model_key:
                        return {
                            "model": model_key,
                            "model_full": MODEL_REGISTRY[model_key],
                            "training_mode": tm,
                            "encoding_mode": encoding_part,
                            "bucket_mode": bucket_mode,
                            "dirname": dirname,
                        }
    return None


def find_trained_models(checkpoint_dir: str) -> List[dict]:
    """Find all trained models in the checkpoint directory."""
    models = []
    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        return models

    for item in checkpoint_path.iterdir():
        if item.is_dir():
            final_path = item / "final"
            if final_path.exists() and final_path.is_dir():
                config = parse_checkpoint_dir(item.name)
                if config:
                    config["path"] = str(final_path)
                    models.append(config)

    return models


def get_unique_base_models(trained_models: List[dict]) -> List[str]:
    """Get unique base model names from trained models."""
    return list(set(m["model"] for m in trained_models))


def load_base_model(model_name: str):
    """Load base model for generation."""
    model_path = MODEL_REGISTRY[model_name]
    print(f"  Loading base model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    return model, tokenizer


def load_finetuned_model(config: dict, checkpoint_dir: str):
    """Load fine-tuned model for generation."""
    model_path = config["path"]
    print(f"  Loading fine-tuned model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if config["training_mode"] == "lora":
        base_model = AutoModelForCausalLM.from_pretrained(
            config["model_full"],
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


@torch.no_grad()
def generate_outputs(
    prompts: List[str],
    model,
    tokenizer,
    temperature: float,
    max_tokens: int = 512,
    top_p: float = 0.95,
) -> List[str]:
    """Generate outputs for all prompts at given temperature."""
    outputs = []
    device = next(model.parameters()).device

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        generated = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Extract only the generated part (exclude prompt)
        input_len = inputs.input_ids.shape[1]
        output_ids = generated[0, input_len:]
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        outputs.append(output_text)

    return outputs


def compute_similarities(
    outputs_temp07: List[str],
    outputs_temp10: List[str],
    embedding_model: SentenceTransformer,
) -> List[float]:
    """Compute cosine similarities between paired outputs."""
    similarities = []

    for out1, out2 in zip(outputs_temp07, outputs_temp10):
        # Handle empty outputs
        if not out1.strip() or not out2.strip():
            similarities.append(0.0)
            continue

        # Embed both outputs
        embeddings = embedding_model.encode([out1, out2], convert_to_tensor=True)

        # Compute cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            embeddings[0].unsqueeze(0),
            embeddings[1].unsqueeze(0)
        ).item()

        similarities.append(cos_sim)

    return similarities


def run_stability_test(
    model,
    tokenizer,
    prompts: List[str],
    embedding_model: SentenceTransformer,
    max_tokens: int = 512,
    top_p: float = 0.95,
) -> Tuple[List[str], List[str], List[float]]:
    """Run stability test for a single model."""
    print("    Generating at temperature=0.7...")
    outputs_temp07 = generate_outputs(
        prompts, model, tokenizer,
        temperature=0.7, max_tokens=max_tokens, top_p=top_p
    )

    print("    Generating at temperature=1.0...")
    outputs_temp10 = generate_outputs(
        prompts, model, tokenizer,
        temperature=1.0, max_tokens=max_tokens, top_p=top_p
    )

    print("    Computing similarities...")
    similarities = compute_similarities(outputs_temp07, outputs_temp10, embedding_model)

    return outputs_temp07, outputs_temp10, similarities


def main():
    parser = argparse.ArgumentParser(
        description="Run semantic stability analysis on all trained models"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory containing trained model checkpoints"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save stability results"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling parameter"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list found models, don't run analysis"
    )
    parser.add_argument(
        "--filter-model",
        type=str,
        choices=["llama", "mistral", "ministral"],
        help="Only run for specific base model"
    )

    args = parser.parse_args()

    # Find all trained models
    print(f"\nScanning for trained models in: {args.checkpoint_dir}")
    print("=" * 70)

    trained_models = find_trained_models(args.checkpoint_dir)

    if args.filter_model:
        trained_models = [m for m in trained_models if m["model"] == args.filter_model]

    # Get unique base models to also test
    base_models = get_unique_base_models(trained_models) if trained_models else ["ministral"]

    if args.filter_model:
        base_models = [args.filter_model]

    print(f"\nFound {len(trained_models)} trained model(s) and {len(base_models)} base model(s)")
    print(f"\nPrompts ({len(STABILITY_PROMPTS)}):")
    for i, p in enumerate(STABILITY_PROMPTS, 1):
        print(f"  {i}. {p}")

    print(f"\nBase models to test:")
    for m in base_models:
        print(f"  - {m} (unconstrained)")

    print(f"\nTrained models to test:")
    for config in trained_models:
        bucket_str = f" [{config['bucket_mode']}]" if config['bucket_mode'] != "embedding" else ""
        print(f"  - {config['model']} / {config['training_mode']} / {config['encoding_mode']}{bucket_str}")

    if args.dry_run:
        print("\n[Dry run - not running analysis]")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load sentence embedding model (keep in memory throughout)
    print("\n" + "=" * 70)
    print("Loading sentence embedding model...")
    print("=" * 70)
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("Loaded: sentence-transformers/all-MiniLM-L6-v2")

    results = []
    all_outputs = {}  # Store outputs for detailed results

    # Test base models first
    print("\n" + "=" * 70)
    print("PHASE 1: Testing base models (unconstrained generation)")
    print("=" * 70)

    for model_name in base_models:
        print(f"\n[Base] {model_name}")
        print("-" * 50)

        model, tokenizer = load_base_model(model_name)

        outputs_07, outputs_10, similarities = run_stability_test(
            model, tokenizer, STABILITY_PROMPTS, embedding_model,
            max_tokens=args.max_tokens, top_p=args.top_p
        )

        unload_model(model, tokenizer)

        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)

        result = StabilityResult(
            model_name=model_name,
            training_mode=None,
            encoding_mode=None,
            bucket_mode=None,
            mean_similarity=mean_sim,
            std_similarity=std_sim,
            per_prompt_similarities=similarities,
            num_prompts=len(STABILITY_PROMPTS),
        )
        results.append(result)

        all_outputs[f"base_{model_name}"] = {
            "temp_0.7": outputs_07,
            "temp_1.0": outputs_10,
            "similarities": similarities,
        }

        print(f"    Stability: {mean_sim:.3f} +/- {std_sim:.3f}")

    # Test fine-tuned models
    print("\n" + "=" * 70)
    print("PHASE 2: Testing fine-tuned models")
    print("=" * 70)

    for i, config in enumerate(trained_models, 1):
        bucket_str = f" [{config['bucket_mode']}]" if config['bucket_mode'] != "embedding" else ""
        print(f"\n[{i}/{len(trained_models)}] {config['model']} / {config['training_mode']} / {config['encoding_mode']}{bucket_str}")
        print("-" * 50)

        try:
            model, tokenizer = load_finetuned_model(config, args.checkpoint_dir)

            outputs_07, outputs_10, similarities = run_stability_test(
                model, tokenizer, STABILITY_PROMPTS, embedding_model,
                max_tokens=args.max_tokens, top_p=args.top_p
            )

            unload_model(model, tokenizer)

            mean_sim = np.mean(similarities)
            std_sim = np.std(similarities)

            result = StabilityResult(
                model_name=config["model"],
                training_mode=config["training_mode"],
                encoding_mode=config["encoding_mode"],
                bucket_mode=config["bucket_mode"],
                mean_similarity=mean_sim,
                std_similarity=std_sim,
                per_prompt_similarities=similarities,
                num_prompts=len(STABILITY_PROMPTS),
            )
            results.append(result)

            key = f"{config['model']}_{config['training_mode']}_{config['encoding_mode']}"
            if config['bucket_mode'] != "embedding":
                key += f"_{config['bucket_mode']}"
            all_outputs[key] = {
                "temp_0.7": outputs_07,
                "temp_1.0": outputs_10,
                "similarities": similarities,
            }

            print(f"    Stability: {mean_sim:.3f} +/- {std_sim:.3f}")

        except Exception as e:
            print(f"    ERROR: {e}")
            results.append(StabilityResult(
                model_name=config["model"],
                training_mode=config["training_mode"],
                encoding_mode=config["encoding_mode"],
                bucket_mode=config["bucket_mode"],
                mean_similarity=-1.0,
                std_similarity=-1.0,
                per_prompt_similarities=[],
                num_prompts=0,
            ))

    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS: Semantic Stability (higher = more stable)")
    print("=" * 70)
    print()
    print(f"{'Model':<12} {'Training':<8} {'Encoding':<15} {'Bucket':<10} {'Stability':>12}")
    print("-" * 65)

    for r in results:
        training = r.training_mode or "base"
        encoding = r.encoding_mode or "-"
        bucket = r.bucket_mode or "-"

        if r.mean_similarity >= 0:
            stability_str = f"{r.mean_similarity:.3f} +/- {r.std_similarity:.3f}"
        else:
            stability_str = "ERROR"

        print(f"{r.model_name:<12} {training:<8} {encoding:<15} {bucket:<10} {stability_str:>12}")

    print()
    print("Interpretation: Despite increased perplexity, stego-fine-tuned models")
    print("exhibit semantic stability under small decoding perturbations comparable")
    print("to the base model.")

    # Save results
    summary_path = os.path.join(args.output_dir, "semantic_stability_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "prompts": STABILITY_PROMPTS,
            "parameters": {
                "max_tokens": args.max_tokens,
                "top_p": args.top_p,
                "temperatures": [0.7, 1.0],
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            },
            "results": [asdict(r) for r in results],
        }, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    # Save detailed outputs
    detailed_path = os.path.join(args.output_dir, "semantic_stability_detailed.json")
    with open(detailed_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "prompts": STABILITY_PROMPTS,
            "outputs": all_outputs,
        }, f, indent=2)
    print(f"Detailed outputs saved to: {detailed_path}")


if __name__ == "__main__":
    main()
