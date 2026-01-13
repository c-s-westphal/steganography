#!/usr/bin/env python3
"""
Perceptibility Evaluation: Measure how recoverable secrets are from steganographic outputs.

Assumes adversary can extract bits from output tokens (knows bucket scheme).
Tests whether a classifier can decode the SECRET from the extracted BITS.

- ASCII encoding: Bits → Secret should be ~100% (deterministic ASCII decode)
- Embedding encodings: Bits → Secret relationship is obfuscated

Perceptibility = classifier accuracy at predicting secret letters from bits.
Higher perceptibility = easier to decode = worse steganography.

Usage:
    python run_perceptibility.py [--output-dir results] [--num-samples 10000]
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from steganography.config import load_config, Config
from steganography.encoding import (
    load_bucket_assignments,
    compute_parity_bucket_assignments,
    secret_to_bits,
)


@dataclass
class PerceptibilityResult:
    """Results for perceptibility evaluation."""
    model_name: str
    encoding_mode: str
    bucket_mode: str
    num_samples: int
    num_seeds: int

    # Classifier: Bits → Secret letters
    mean_accuracy: float
    std_accuracy: float
    ci95: Tuple[float, float]

    # Per-letter accuracy
    mean_per_letter_accuracy: List[float] = field(default_factory=list)

    # Per-seed details
    per_seed_accuracy: List[float] = field(default_factory=list)
    per_seed_letter_accuracy: List[List[float]] = field(default_factory=list)


def load_tokenizer(model_name: str):
    """Load tokenizer."""
    from transformers import AutoTokenizer

    model_registry = {
        "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
        "ministral": "mistralai/Ministral-8B-Instruct-2410",
    }

    model_id = model_registry.get(model_name)
    if not model_id:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"  Loading tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return tokenizer, model_id


def generate_random_secrets(num_samples: int, alphabet: str = "abcdefghijklmnopqrstuvwxyz",
                           secret_length: int = 4, seed: int = 42) -> List[str]:
    """Generate random secrets."""
    np.random.seed(seed)
    secrets = []
    for _ in range(num_samples):
        secret = ''.join(np.random.choice(list(alphabet), secret_length))
        secrets.append(secret)
    return secrets


def encode_secret_to_bits(
    secret: str,
    encoding_mode: str,
    config: Config,
    model=None,
    tokenizer=None,
    embedding_key_config=None,
    embedding_only_config=None,
) -> str:
    """Encode a secret to bits using the specified encoding mode."""
    if encoding_mode == "ascii":
        return secret_to_bits(secret, config)
    elif encoding_mode == "embedding":
        from steganography.encoding import derive_embedding_key
        return derive_embedding_key(secret, model, tokenizer, embedding_key_config)
    elif encoding_mode == "embedding_only":
        from steganography.encoding import secret_to_bits_embedding_only
        return secret_to_bits_embedding_only(secret, embedding_only_config)
    elif encoding_mode == "xor":
        from steganography.encoding import derive_embedding_key, xor_bits
        ascii_bits = secret_to_bits(secret, config)
        emb_key = derive_embedding_key(secret, model, tokenizer, embedding_key_config)
        return xor_bits(ascii_bits, emb_key)
    elif encoding_mode == "embedding_xor":
        from steganography.encoding import (
            secret_to_bits_embedding_only, derive_embedding_key, xor_bits
        )
        emb_only_bits = secret_to_bits_embedding_only(secret, embedding_only_config)
        emb_key = derive_embedding_key(secret, model, tokenizer, embedding_key_config)
        return xor_bits(emb_only_bits, emb_key)
    else:
        raise ValueError(f"Unknown encoding mode: {encoding_mode}")


def generate_bits_for_secrets(
    secrets: List[str],
    encoding_mode: str,
    config: Config,
    model=None,
    tokenizer=None,
    embedding_key_config=None,
    embedding_only_config=None,
) -> np.ndarray:
    """
    Generate the bit sequences that would be encoded for each secret.

    Returns:
        bits: [num_samples, 32] array of bits (0 or 1)
    """
    num_samples = len(secrets)
    num_bits = config.secret_bits  # 32

    bits = np.zeros((num_samples, num_bits), dtype=np.int32)

    for i, secret in enumerate(secrets):
        bits_str = encode_secret_to_bits(
            secret, encoding_mode, config,
            model, tokenizer, embedding_key_config, embedding_only_config
        )
        bits[i] = [int(b) for b in bits_str]

    return bits


def secrets_to_labels(secrets: List[str]) -> np.ndarray:
    """Convert secrets to numeric labels (0-25 for each letter)."""
    labels = np.array([[ord(c) - ord('a') for c in secret] for secret in secrets])
    return labels  # Shape: [num_samples, 4]


def compute_confidence_interval(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Compute confidence interval for a list of values."""
    n = len(values)
    if n < 2:
        return (np.mean(values), np.mean(values))
    mean = np.mean(values)
    se = stats.sem(values)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return (mean - h, mean + h)


def train_classifier(X_train, y_train, X_test, y_test, seed: int = 42) -> float:
    """Train RandomForest classifier and return accuracy."""
    try:
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
    except ImportError:
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=seed,
            n_jobs=-1
        )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return accuracy_score(y_test, y_pred)


def run_perceptibility_evaluation(
    model_name: str,
    encoding_mode: str,
    bucket_mode: str,
    tokenizer,
    config: Config,
    num_samples: int = 10000,
    num_seeds: int = 5,
    test_size: float = 0.2,
    model=None,
    embedding_key_config=None,
    embedding_only_config=None,
) -> PerceptibilityResult:
    """
    Run perceptibility evaluation.

    Tests: Given bits (extracted via bucket knowledge), can classifier decode the secret?
    """
    print(f"\nEvaluating: {model_name} / {encoding_mode} / {bucket_mode}")
    print(f"  Samples: {num_samples}, Seeds: {num_seeds}")

    # Results storage
    accuracies = []
    letter_accuracies = []

    for seed in range(num_seeds):
        print(f"  Seed {seed + 1}/{num_seeds}...")
        np.random.seed(seed * 1000)

        # Generate random secrets
        secrets = generate_random_secrets(num_samples, config.secret_alphabet,
                                         config.secret_length, seed=seed * 1000)

        # Generate bits for each secret (what would be encoded)
        print(f"    Generating bits for {num_samples} secrets...")
        bits = generate_bits_for_secrets(
            secrets, encoding_mode, config,
            model=model, tokenizer=tokenizer,
            embedding_key_config=embedding_key_config,
            embedding_only_config=embedding_only_config,
        )

        # Convert secrets to labels (0-25 for each letter)
        secret_labels = secrets_to_labels(secrets)

        # Input: bits (32 binary values)
        X = bits.astype(np.float32)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, secret_labels,
            test_size=test_size, random_state=seed
        )

        # Train classifier for each letter position
        print(f"    Training classifier (Bits → Secret)...")
        letter_accs = []
        for pos in range(4):
            acc = train_classifier(
                X_train, y_train[:, pos],
                X_test, y_test[:, pos],
                seed=seed
            )
            letter_accs.append(acc)

        avg_acc = np.mean(letter_accs)
        accuracies.append(avg_acc)
        letter_accuracies.append(letter_accs)
        print(f"      Per-letter accuracy: {[f'{a:.4f}' for a in letter_accs]}")
        print(f"      Average accuracy: {avg_acc:.4f}")

    # Compute statistics
    mean_letter_acc = [np.mean([la[i] for la in letter_accuracies]) for i in range(4)]

    result = PerceptibilityResult(
        model_name=model_name,
        encoding_mode=encoding_mode,
        bucket_mode=bucket_mode,
        num_samples=num_samples,
        num_seeds=num_seeds,
        mean_accuracy=np.mean(accuracies),
        std_accuracy=np.std(accuracies),
        ci95=compute_confidence_interval(accuracies),
        mean_per_letter_accuracy=mean_letter_acc,
        per_seed_accuracy=accuracies,
        per_seed_letter_accuracy=letter_accuracies,
    )

    return result


def parse_checkpoint_dir(dirname: str) -> Optional[Dict]:
    """Parse checkpoint directory name to extract configuration."""
    if not dirname.startswith("trojanstego_"):
        return None

    rest = dirname[len("trojanstego_"):]

    model_map = {
        "meta-llama-3.1-8b-instruct": "llama",
        "mistral-7b-instruct-v0.3": "mistral",
        "ministral-8b-instruct-2410": "ministral",
    }

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
                    model_key = model_map.get(model_part.lower())
                    if model_key:
                        return {
                            "model": model_key,
                            "training_mode": tm,
                            "encoding_mode": encoding_part,
                            "bucket_mode": bucket_mode,
                            "dirname": dirname,
                        }
    return None


def discover_configurations(checkpoint_dir: str = "checkpoints") -> List[Dict]:
    """Discover all trained model configurations from checkpoint directory."""
    configs = []
    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        print(f"Warning: Checkpoint directory not found: {checkpoint_dir}")
        return configs

    for item in checkpoint_path.iterdir():
        if item.is_dir():
            final_path = item / "final"
            if final_path.exists() and final_path.is_dir():
                config = parse_checkpoint_dir(item.name)
                if config:
                    configs.append(config)

    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Run perceptibility evaluation on steganographic encodings"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory containing trained model checkpoints (for auto-discovery)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=5,
        help="Number of random seeds for robustness"
    )
    parser.add_argument(
        "--pod-name",
        type=str,
        default=None,
        help="Pod name for output file naming"
    )
    parser.add_argument(
        "--auto-discover",
        action="store_true",
        help="Auto-discover configurations from checkpoint directory"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Models to evaluate"
    )
    parser.add_argument(
        "--encoding-modes",
        type=str,
        nargs="+",
        default=None,
        help="Encoding modes to evaluate"
    )
    parser.add_argument(
        "--bucket-modes",
        type=str,
        nargs="+",
        default=None,
        help="Bucket modes to evaluate"
    )
    # Keep for backwards compatibility but not used
    parser.add_argument("--pca-dims", type=int, default=50, help="(unused)")

    args = parser.parse_args()

    # Build evaluation configurations
    if args.auto_discover:
        print("Auto-discovering configurations from checkpoint directory...")
        discovered = discover_configurations(args.checkpoint_dir)

        if not discovered:
            print(f"No configurations found in {args.checkpoint_dir}")
            return []

        print(f"Found {len(discovered)} configurations:")
        for cfg in discovered:
            print(f"  - {cfg['model']} / {cfg['encoding_mode']} / {cfg['bucket_mode']}")

        models = list(set(cfg['model'] for cfg in discovered))
        encoding_modes = list(set(cfg['encoding_mode'] for cfg in discovered))
        bucket_modes = list(set(cfg['bucket_mode'] for cfg in discovered))
        eval_configs = [(cfg['model'], cfg['encoding_mode'], cfg['bucket_mode']) for cfg in discovered]
    else:
        models = args.models or ["llama", "ministral"]
        encoding_modes = args.encoding_modes or ["ascii", "embedding", "embedding_only", "xor", "embedding_xor"]
        bucket_modes = args.bucket_modes or ["parity", "embedding"]
        eval_configs = [(m, e, b) for m in models for e in encoding_modes for b in bucket_modes]

    print("=" * 70)
    print("PERCEPTIBILITY EVALUATION")
    print("=" * 70)
    print(f"Samples: {args.num_samples}")
    print(f"Seeds: {args.num_seeds}")
    print(f"Models: {models}")
    print(f"Encoding modes: {encoding_modes}")
    print(f"Bucket modes: {bucket_modes}")
    print(f"Configurations to evaluate: {len(eval_configs)}")
    print()
    print("Testing: Given bits (from bucket extraction), can classifier decode secret?")
    print("  - ASCII: Should be ~100% (deterministic decode)")
    print("  - XOR modes: Obfuscated by XORing with embedding-derived key")
    print("  - Embedding modes: Fully obfuscated through embedding projections")

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []

    # Group configs by model
    configs_by_model = defaultdict(list)
    for model_name, encoding_mode, bucket_mode in eval_configs:
        configs_by_model[model_name].append((encoding_mode, bucket_mode))

    model_registry = {
        "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
        "ministral": "mistralai/Ministral-8B-Instruct-2410",
    }

    for model_name in configs_by_model:
        print(f"\n{'=' * 70}")
        print(f"MODEL: {model_name}")
        print("=" * 70)

        try:
            tokenizer, model_id = load_tokenizer(model_name)
        except Exception as e:
            print(f"  Error loading tokenizer: {e}")
            continue

        vocab_size = tokenizer.vocab_size
        print(f"  Vocab size: {vocab_size}")

        # Get encoding modes for this model
        model_encoding_modes = set(em for em, bm in configs_by_model[model_name])

        # Precompute embedding configs if needed
        embedding_key_config = None
        embedding_only_config = None
        model_for_encoding = None

        if any(m in model_encoding_modes for m in ["embedding", "xor", "embedding_xor"]):
            print("  Precomputing embedding key config...")
            from transformers import AutoModelForCausalLM
            from steganography.encoding import precompute_embedding_key_config

            model_for_encoding = AutoModelForCausalLM.from_pretrained(
                model_registry[model_name],
                torch_dtype=torch.float32,
                device_map="cpu",
            )
            embedding_key_config = precompute_embedding_key_config(
                model_for_encoding, tokenizer, seed_base=1000, num_bits=32
            )

        if any(m in model_encoding_modes for m in ["embedding_only", "embedding_xor"]):
            print("  Precomputing embedding-only config...")
            from steganography.encoding import precompute_embedding_only_config

            if model_for_encoding is None:
                from transformers import AutoModelForCausalLM
                model_for_encoding = AutoModelForCausalLM.from_pretrained(
                    model_registry[model_name],
                    torch_dtype=torch.float32,
                    device_map="cpu",
                )
            embedding_only_config = precompute_embedding_only_config(
                model_for_encoding, tokenizer, bits_per_letter=8
            )

        # Run evaluations (bucket_mode doesn't affect bits→secret, but we track it)
        for encoding_mode, bucket_mode in configs_by_model[model_name]:
            print(f"\n{'-' * 50}")
            print(f"Config: {encoding_mode} / {bucket_mode}")
            print("-" * 50)

            config = load_config(
                base_model=model_registry[model_name],
                encoding_mode=encoding_mode,
            )

            try:
                result = run_perceptibility_evaluation(
                    model_name=model_name,
                    encoding_mode=encoding_mode,
                    bucket_mode=bucket_mode,
                    tokenizer=tokenizer,
                    config=config,
                    num_samples=args.num_samples,
                    num_seeds=args.num_seeds,
                    model=model_for_encoding,
                    embedding_key_config=embedding_key_config,
                    embedding_only_config=embedding_only_config,
                )
                all_results.append(result)
            except Exception as e:
                print(f"  Error evaluating {encoding_mode}/{bucket_mode}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Clean up
        del tokenizer
        if model_for_encoding is not None:
            del model_for_encoding
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("Perceptibility = Accuracy of predicting secret letters from extracted bits")
    print("Higher = easier to decode = more perceptible = worse steganography")
    print()
    print(f"{'Model':<12} {'Encoding':<15} {'Bucket':<10} {'Accuracy':<15} {'Per-Letter':<30}")
    print("-" * 82)

    for r in all_results:
        letter_str = ', '.join([f'{a:.2%}' for a in r.mean_per_letter_accuracy])
        print(f"{r.model_name:<12} {r.encoding_mode:<15} {r.bucket_mode:<10} "
              f"{r.mean_accuracy:>6.2%} ±{r.std_accuracy:>5.2%}   [{letter_str}]")

    # Save results
    pod_suffix = f"_{args.pod_name}" if args.pod_name else ""
    output_file = os.path.join(args.output_dir, f"perceptibility_summary{pod_suffix}.json")

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "description": "Perceptibility = accuracy of predicting secret from extracted bits",
        "parameters": {
            "num_samples": args.num_samples,
            "num_seeds": args.num_seeds,
            "models": list(models) if not isinstance(models, list) else models,
            "encoding_modes": list(encoding_modes) if not isinstance(encoding_modes, list) else encoding_modes,
            "bucket_modes": list(bucket_modes) if not isinstance(bucket_modes, list) else bucket_modes,
            "auto_discover": args.auto_discover,
        },
        "results": [asdict(r) for r in all_results],
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    main()
