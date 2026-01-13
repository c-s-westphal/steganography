#!/usr/bin/env python3
"""
Perceptibility Evaluation: Measure how recoverable secrets are from steganographic outputs.

Tests whether an adversary can recover the SECRET from OUTPUT TOKENS using:
1. XGBoost on Token IDs → Secret letters (discovers discrete patterns like parity)
2. MLP on Embeddings → Secret letters (discovers continuous patterns)

Perceptibility = max accuracy across both approaches.
Higher perceptibility = easier to detect = worse steganography.

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
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
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
    pca_dims: int

    # XGBoost: Token IDs → Secret letters
    mean_xgb_accuracy: float
    std_xgb_accuracy: float
    ci95_xgb: Tuple[float, float]

    # MLP: Embeddings → Secret letters
    mean_mlp_accuracy: float
    std_mlp_accuracy: float
    ci95_mlp: Tuple[float, float]

    # Overall perceptibility (max of both)
    mean_perceptibility: float
    std_perceptibility: float
    ci95_perceptibility: Tuple[float, float]

    # Per-seed details
    per_seed_xgb: List[float] = field(default_factory=list)
    per_seed_mlp: List[float] = field(default_factory=list)
    per_seed_perceptibility: List[float] = field(default_factory=list)


def load_model_and_tokenizer(model_name: str):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_registry = {
        "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
        "ministral": "mistralai/Ministral-8B-Instruct-2410",
    }

    model_id = model_registry.get(model_name)
    if not model_id:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"  Loading model and tokenizer from {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
    )

    # Get embedding matrix
    embeddings = model.get_input_embeddings().weight.detach().clone()

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return tokenizer, embeddings


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


def generate_ideal_samples(
    secrets: List[str],
    bucket_assignments: torch.Tensor,
    encoding_mode: str,
    config: Config,
    model=None,
    tokenizer=None,
    embedding_key_config=None,
    embedding_only_config=None,
) -> np.ndarray:
    """
    Generate ideal output token sequences by sampling from bucket assignments.

    Returns:
        output_token_ids: [num_samples, 32] array of token IDs
    """
    # Pre-compute token lists for each bucket
    bucket_np = bucket_assignments.cpu().numpy() if bucket_assignments.is_cuda else bucket_assignments.numpy()
    bucket_0_tokens = np.where(bucket_np == 0)[0]
    bucket_1_tokens = np.where(bucket_np == 1)[0]

    if len(bucket_0_tokens) == 0 or len(bucket_1_tokens) == 0:
        raise ValueError("One of the buckets is empty!")

    num_samples = len(secrets)
    num_bits = config.secret_bits  # 32

    output_token_ids = np.zeros((num_samples, num_bits), dtype=np.int64)

    for i, secret in enumerate(secrets):
        # Get bits to encode for this secret
        bits_str = encode_secret_to_bits(
            secret, encoding_mode, config,
            model, tokenizer, embedding_key_config, embedding_only_config
        )

        # Convert bits string to list of ints
        bits = [int(b) for b in bits_str]

        # For each bit, sample a token from the corresponding bucket
        for j, bit in enumerate(bits):
            if bit == 0:
                output_token_ids[i, j] = np.random.choice(bucket_0_tokens)
            else:
                output_token_ids[i, j] = np.random.choice(bucket_1_tokens)

    return output_token_ids


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


def train_xgboost_classifier(X_train, y_train, X_test, y_test, seed: int = 42) -> float:
    """Train XGBoost classifier and return accuracy."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("  Warning: XGBoost not installed, using RandomForest instead")
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return accuracy_score(y_test, y_pred)

    clf = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return accuracy_score(y_test, y_pred)


def train_mlp_classifier(X_train, y_train, X_test, y_test, seed: int = 42) -> float:
    """Train MLP classifier and return accuracy."""
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        solver='adam',
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=seed,
        verbose=False,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return accuracy_score(y_test, y_pred)


def run_perceptibility_evaluation(
    model_name: str,
    encoding_mode: str,
    bucket_mode: str,
    tokenizer,
    embeddings: torch.Tensor,
    bucket_assignments: torch.Tensor,
    config: Config,
    num_samples: int = 10000,
    num_seeds: int = 5,
    pca_dims: int = 50,
    test_size: float = 0.2,
    model=None,
    embedding_key_config=None,
    embedding_only_config=None,
) -> PerceptibilityResult:
    """
    Run perceptibility evaluation.

    Tests:
    1. XGBoost: Token IDs → Secret letters (can discover parity patterns)
    2. MLP: Embeddings → Secret letters (can discover continuous patterns)
    """
    print(f"\nEvaluating: {model_name} / {encoding_mode} / {bucket_mode}")
    print(f"  Samples: {num_samples}, Seeds: {num_seeds}, PCA dims: {pca_dims}")

    # Fit PCA on full vocabulary embeddings
    print("  Fitting PCA on vocabulary embeddings...")
    pca = PCA(n_components=pca_dims, random_state=42)
    pca.fit(embeddings.numpy())
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    # Results storage
    xgb_accs = []
    mlp_accs = []
    perceptibilities = []

    for seed in range(num_seeds):
        print(f"  Seed {seed + 1}/{num_seeds}...")
        np.random.seed(seed * 1000)

        # Generate random secrets
        secrets = generate_random_secrets(num_samples, config.secret_alphabet,
                                         config.secret_length, seed=seed * 1000)

        # Generate ideal output token sequences
        print(f"    Generating {num_samples} ideal samples...")
        output_token_ids = generate_ideal_samples(
            secrets, bucket_assignments, encoding_mode, config,
            model=model, tokenizer=tokenizer,
            embedding_key_config=embedding_key_config,
            embedding_only_config=embedding_only_config,
        )

        # Convert secrets to labels (0-25 for each letter)
        secret_labels = secrets_to_labels(secrets)

        # Prepare features
        # X1: Output token IDs (32 integers)
        X_token_ids = output_token_ids.astype(np.float32)

        # X2: Output token embeddings (PCA reduced)
        print(f"    Computing output embeddings...")
        output_embeds_flat = embeddings[output_token_ids.flatten()].numpy()
        output_embeds_pca = pca.transform(output_embeds_flat)
        X_embeddings = output_embeds_pca.reshape(num_samples, -1)  # [N, 32*pca_dims]

        # Train/test split
        (X_ids_train, X_ids_test,
         X_emb_train, X_emb_test,
         y_train, y_test) = train_test_split(
            X_token_ids, X_embeddings, secret_labels,
            test_size=test_size, random_state=seed
        )

        # Train classifiers for each letter position, average accuracy
        print(f"    Training XGBoost (Token IDs → Secret)...")
        xgb_acc = 0
        for pos in range(4):
            acc = train_xgboost_classifier(
                X_ids_train, y_train[:, pos],
                X_ids_test, y_test[:, pos],
                seed=seed
            )
            xgb_acc += acc / 4
        xgb_accs.append(xgb_acc)
        print(f"      XGBoost accuracy: {xgb_acc:.4f}")

        print(f"    Training MLP (Embeddings → Secret)...")
        mlp_acc = 0
        for pos in range(4):
            acc = train_mlp_classifier(
                X_emb_train, y_train[:, pos],
                X_emb_test, y_test[:, pos],
                seed=seed
            )
            mlp_acc += acc / 4
        mlp_accs.append(mlp_acc)
        print(f"      MLP accuracy: {mlp_acc:.4f}")

        # Perceptibility = max of both
        perceptibility = max(xgb_acc, mlp_acc)
        perceptibilities.append(perceptibility)
        print(f"      Perceptibility: {perceptibility:.4f}")

    # Compute statistics
    result = PerceptibilityResult(
        model_name=model_name,
        encoding_mode=encoding_mode,
        bucket_mode=bucket_mode,
        num_samples=num_samples,
        num_seeds=num_seeds,
        pca_dims=pca_dims,
        mean_xgb_accuracy=np.mean(xgb_accs),
        std_xgb_accuracy=np.std(xgb_accs),
        ci95_xgb=compute_confidence_interval(xgb_accs),
        mean_mlp_accuracy=np.mean(mlp_accs),
        std_mlp_accuracy=np.std(mlp_accs),
        ci95_mlp=compute_confidence_interval(mlp_accs),
        mean_perceptibility=np.mean(perceptibilities),
        std_perceptibility=np.std(perceptibilities),
        ci95_perceptibility=compute_confidence_interval(perceptibilities),
        per_seed_xgb=xgb_accs,
        per_seed_mlp=mlp_accs,
        per_seed_perceptibility=perceptibilities,
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

                # Check for bucket mode suffix
                bucket_mode = "embedding"  # default
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
        "--pca-dims",
        type=int,
        default=50,
        help="Number of PCA dimensions for embeddings"
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
        encoding_modes = args.encoding_modes or ["ascii", "embedding", "embedding_only"]
        bucket_modes = args.bucket_modes or ["parity", "embedding"]
        eval_configs = [(m, e, b) for m in models for e in encoding_modes for b in bucket_modes]

    print("=" * 70)
    print("PERCEPTIBILITY EVALUATION")
    print("=" * 70)
    print(f"Samples: {args.num_samples}")
    print(f"Seeds: {args.num_seeds}")
    print(f"PCA dims: {args.pca_dims}")
    print(f"Models: {models}")
    print(f"Encoding modes: {encoding_modes}")
    print(f"Bucket modes: {bucket_modes}")
    print(f"Configurations to evaluate: {len(eval_configs)}")

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
            tokenizer, embeddings = load_model_and_tokenizer(model_name)
        except Exception as e:
            print(f"  Error loading model: {e}")
            continue

        vocab_size = embeddings.shape[0]
        embed_dim = embeddings.shape[1]
        print(f"  Vocab size: {vocab_size}, Embed dim: {embed_dim}")

        # Get encoding/bucket modes for this model
        model_encoding_modes = set(em for em, bm in configs_by_model[model_name])
        model_bucket_modes = set(bm for em, bm in configs_by_model[model_name])

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

        # Cache bucket assignments
        bucket_cache = {}
        for bucket_mode in model_bucket_modes:
            if bucket_mode == "parity":
                bucket_cache[bucket_mode] = compute_parity_bucket_assignments(vocab_size)
            else:
                bucket_config_path = f"data/bucket_config/{model_name}"
                if not os.path.exists(bucket_config_path):
                    print(f"  Warning: no bucket config at {bucket_config_path}")
                    bucket_cache[bucket_mode] = None
                else:
                    try:
                        bucket_assignments, _ = load_bucket_assignments(bucket_config_path)
                        bucket_cache[bucket_mode] = bucket_assignments
                    except Exception as e:
                        print(f"  Error loading bucket assignments: {e}")
                        bucket_cache[bucket_mode] = None

        # Run evaluations
        for encoding_mode, bucket_mode in configs_by_model[model_name]:
            print(f"\n{'-' * 50}")
            print(f"Config: {encoding_mode} / {bucket_mode}")
            print("-" * 50)

            bucket_assignments = bucket_cache.get(bucket_mode)
            if bucket_assignments is None:
                print(f"  Skipping - no bucket assignments for {bucket_mode}")
                continue

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
                    embeddings=embeddings,
                    bucket_assignments=bucket_assignments,
                    config=config,
                    num_samples=args.num_samples,
                    num_seeds=args.num_seeds,
                    pca_dims=args.pca_dims,
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
        del embeddings
        del tokenizer
        if model_for_encoding is not None:
            del model_for_encoding
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<12} {'Encoding':<15} {'Bucket':<10} {'XGB Acc':<12} {'MLP Acc':<12} {'Percept':<12}")
    print("-" * 83)

    for r in all_results:
        print(f"{r.model_name:<12} {r.encoding_mode:<15} {r.bucket_mode:<10} "
              f"{r.mean_xgb_accuracy:>6.2%} ±{r.std_xgb_accuracy:>4.2%}  "
              f"{r.mean_mlp_accuracy:>6.2%} ±{r.std_mlp_accuracy:>4.2%}  "
              f"{r.mean_perceptibility:>6.2%} ±{r.std_perceptibility:>4.2%}")

    # Save results
    pod_suffix = f"_{args.pod_name}" if args.pod_name else ""
    output_file = os.path.join(args.output_dir, f"perceptibility_summary{pod_suffix}.json")

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "num_samples": args.num_samples,
            "num_seeds": args.num_seeds,
            "pca_dims": args.pca_dims,
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
