#!/usr/bin/env python3
"""
Perceptibility Evaluation: Measure how detectable steganographic encodings are.

Perceptibility is defined as the maximum test accuracy of a classifier attempting
to predict the SECRET (token IDs or embeddings) from the OUTPUT tokens (IDs or embeddings).

If an adversary can recover the secret by looking at output tokens, the scheme is perceptible.

Classifiers:
1. Output Token IDs → Secret Token IDs
2. Output Token Embeddings (PCA) → Secret Token IDs
3. Output Token IDs → Secret Token Embeddings
4. Output Token Embeddings (PCA) → Secret Token Embeddings

Lower perceptibility = harder to detect = better steganography.

Usage:
    python run_perceptibility.py [--output-dir results] [--num-samples 100000]
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
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder
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

    # Output Token IDs → Secret Token IDs (classification)
    mean_id_to_id_accuracy: float
    std_id_to_id_accuracy: float
    ci95_id_to_id: Tuple[float, float]

    # Output Embeddings → Secret Token IDs (classification)
    mean_emb_to_id_accuracy: float
    std_emb_to_id_accuracy: float
    ci95_emb_to_id: Tuple[float, float]

    # Output Token IDs → Secret Embeddings (regression, R² score)
    mean_id_to_emb_r2: float
    std_id_to_emb_r2: float
    ci95_id_to_emb: Tuple[float, float]

    # Output Embeddings → Secret Embeddings (regression, R² score)
    mean_emb_to_emb_r2: float
    std_emb_to_emb_r2: float
    ci95_emb_to_emb: Tuple[float, float]

    # Overall perceptibility (max classification accuracy)
    mean_perceptibility: float
    std_perceptibility: float
    ci95_perceptibility: Tuple[float, float]

    # Per-seed details
    per_seed_id_to_id: List[float] = field(default_factory=list)
    per_seed_emb_to_id: List[float] = field(default_factory=list)
    per_seed_id_to_emb: List[float] = field(default_factory=list)
    per_seed_emb_to_emb: List[float] = field(default_factory=list)
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
        torch_dtype=torch.float32,  # Need float32 for PCA
        device_map="cpu",  # Load on CPU to save GPU memory
    )

    # Get embedding matrix
    embeddings = model.get_input_embeddings().weight.detach().clone()

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return tokenizer, embeddings


def tokenize_secrets(secrets: List[str], tokenizer) -> np.ndarray:
    """
    Tokenize secrets to get token IDs for each letter.

    Returns array of shape [num_secrets, 4] with token IDs for each letter.
    """
    secret_token_ids = []

    for secret in secrets:
        # Tokenize each letter individually
        letter_ids = []
        for letter in secret:
            # Tokenize the letter (without special tokens)
            ids = tokenizer.encode(letter, add_special_tokens=False)
            # Take the first token ID (in case of multi-token)
            letter_ids.append(ids[0] if ids else 0)
        secret_token_ids.append(letter_ids)

    return np.array(secret_token_ids)


def generate_random_secrets(num_samples: int, alphabet: str = "abcdefghijklmnopqrstuvwxyz",
                           secret_length: int = 4, seed: int = 42) -> List[str]:
    """Generate random secrets."""
    np.random.seed(seed)
    secrets = []
    for _ in range(num_samples):
        secret = ''.join(np.random.choice(list(alphabet), secret_length))
        secrets.append(secret)
    return secrets


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

    For each secret:
    1. Compute the bits to encode using the encoding mode
    2. For each bit position, sample a random token from the correct bucket

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
        if encoding_mode == "ascii":
            bits_str = secret_to_bits(secret, config)
        elif encoding_mode == "embedding":
            from steganography.encoding import derive_embedding_key
            bits_str = derive_embedding_key(secret, model, tokenizer, embedding_key_config)
        elif encoding_mode == "embedding_only":
            from steganography.encoding import secret_to_bits_embedding_only
            bits_str = secret_to_bits_embedding_only(secret, embedding_only_config)
        elif encoding_mode == "xor":
            from steganography.encoding import derive_embedding_key, xor_bits
            ascii_bits = secret_to_bits(secret, config)
            emb_key = derive_embedding_key(secret, model, tokenizer, embedding_key_config)
            bits_str = xor_bits(ascii_bits, emb_key)
        elif encoding_mode == "embedding_xor":
            from steganography.encoding import (
                secret_to_bits_embedding_only, derive_embedding_key, xor_bits
            )
            emb_only_bits = secret_to_bits_embedding_only(secret, embedding_only_config)
            emb_key = derive_embedding_key(secret, model, tokenizer, embedding_key_config)
            bits_str = xor_bits(emb_only_bits, emb_key)
        else:
            raise ValueError(f"Unknown encoding mode: {encoding_mode}")

        # Convert bits string to list of ints
        bits = [int(b) for b in bits_str]

        # For each bit, sample a token from the corresponding bucket
        for j, bit in enumerate(bits):
            if bit == 0:
                output_token_ids[i, j] = np.random.choice(bucket_0_tokens)
            else:
                output_token_ids[i, j] = np.random.choice(bucket_1_tokens)

    return output_token_ids


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


def train_regressor(X_train, y_train, X_test, y_test, seed: int = 42) -> float:
    """Train MLP regressor and return R² score."""
    reg = MLPRegressor(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        solver='adam',
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=seed,
        verbose=False,
    )

    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    return r2_score(y_test, y_pred)


def run_perceptibility_evaluation(
    model_name: str,
    encoding_mode: str,
    bucket_mode: str,
    tokenizer,
    embeddings: torch.Tensor,
    bucket_assignments: torch.Tensor,
    config: Config,
    num_samples: int = 100000,
    num_seeds: int = 5,
    pca_dims: int = 50,
    test_size: float = 0.2,
    model=None,
    embedding_key_config=None,
    embedding_only_config=None,
) -> PerceptibilityResult:
    """
    Run full perceptibility evaluation.

    Tests whether a classifier can predict the SECRET from OUTPUT TOKENS.
    """
    print(f"\nEvaluating: {model_name} / {encoding_mode} / {bucket_mode}")
    print(f"  Samples: {num_samples}, Seeds: {num_seeds}, PCA dims: {pca_dims}")

    # Fit PCA on full vocabulary embeddings
    print("  Fitting PCA on vocabulary embeddings...")
    pca = PCA(n_components=pca_dims, random_state=42)
    pca.fit(embeddings.numpy())
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    # Results storage
    id_to_id_accs = []
    emb_to_id_accs = []
    id_to_emb_r2s = []
    emb_to_emb_r2s = []
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

        # Get secret token IDs (target for classification)
        secret_token_ids = tokenize_secrets(secrets, tokenizer)

        # Flatten secret token IDs to single label per sample (for simpler classification)
        # Encode the 4-letter secret as a single integer label
        secret_labels = np.array([
            s[0] * 26**3 + s[1] * 26**2 + s[2] * 26 + s[3]
            for s in (np.array([[ord(c) - ord('a') for c in sec] for sec in secrets]))
        ])

        # Prepare features
        # X1: Output token IDs (32 integers)
        X_output_ids = output_token_ids.astype(np.float32)

        # X2: Output token embeddings (PCA reduced)
        print(f"    Computing output embeddings...")
        output_embeds_flat = embeddings[output_token_ids.flatten()].numpy()
        output_embeds_pca = pca.transform(output_embeds_flat)
        X_output_emb = output_embeds_pca.reshape(num_samples, -1)  # [N, 32*pca_dims]

        # Y1: Secret token IDs (for classification - use letter indices 0-25)
        y_secret_letters = np.array([[ord(c) - ord('a') for c in sec] for sec in secrets])
        # Flatten to predict all 4 letters together
        y_secret_flat = y_secret_letters  # [N, 4]

        # Y2: Secret embeddings (for regression)
        print(f"    Computing secret embeddings...")
        secret_embeds_flat = embeddings[secret_token_ids.flatten()].numpy()
        secret_embeds_pca = pca.transform(secret_embeds_flat)
        y_secret_emb = secret_embeds_pca.reshape(num_samples, -1)  # [N, 4*pca_dims]

        # Train/test split
        (X_out_ids_train, X_out_ids_test,
         X_out_emb_train, X_out_emb_test,
         y_letters_train, y_letters_test,
         y_emb_train, y_emb_test) = train_test_split(
            X_output_ids, X_output_emb, y_secret_flat, y_secret_emb,
            test_size=test_size, random_state=seed
        )

        # 1. Output Token IDs → Secret Letters (classification)
        print(f"    Training ID→ID classifier...")
        # Train separate classifier for each letter position, average accuracy
        id_to_id_acc = 0
        for pos in range(4):
            acc = train_classifier(
                X_out_ids_train, y_letters_train[:, pos],
                X_out_ids_test, y_letters_test[:, pos],
                seed=seed
            )
            id_to_id_acc += acc / 4
        id_to_id_accs.append(id_to_id_acc)
        print(f"      ID→ID accuracy: {id_to_id_acc:.4f}")

        # 2. Output Embeddings → Secret Letters (classification)
        print(f"    Training Emb→ID classifier...")
        emb_to_id_acc = 0
        for pos in range(4):
            acc = train_classifier(
                X_out_emb_train, y_letters_train[:, pos],
                X_out_emb_test, y_letters_test[:, pos],
                seed=seed
            )
            emb_to_id_acc += acc / 4
        emb_to_id_accs.append(emb_to_id_acc)
        print(f"      Emb→ID accuracy: {emb_to_id_acc:.4f}")

        # 3. Output Token IDs → Secret Embeddings (regression)
        print(f"    Training ID→Emb regressor...")
        id_to_emb_r2 = train_regressor(
            X_out_ids_train, y_emb_train,
            X_out_ids_test, y_emb_test,
            seed=seed
        )
        id_to_emb_r2s.append(id_to_emb_r2)
        print(f"      ID→Emb R²: {id_to_emb_r2:.4f}")

        # 4. Output Embeddings → Secret Embeddings (regression)
        print(f"    Training Emb→Emb regressor...")
        emb_to_emb_r2 = train_regressor(
            X_out_emb_train, y_emb_train,
            X_out_emb_test, y_emb_test,
            seed=seed
        )
        emb_to_emb_r2s.append(emb_to_emb_r2)
        print(f"      Emb→Emb R²: {emb_to_emb_r2:.4f}")

        # Perceptibility = max classification accuracy
        perceptibility = max(id_to_id_acc, emb_to_id_acc)
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
        mean_id_to_id_accuracy=np.mean(id_to_id_accs),
        std_id_to_id_accuracy=np.std(id_to_id_accs),
        ci95_id_to_id=compute_confidence_interval(id_to_id_accs),
        mean_emb_to_id_accuracy=np.mean(emb_to_id_accs),
        std_emb_to_id_accuracy=np.std(emb_to_id_accs),
        ci95_emb_to_id=compute_confidence_interval(emb_to_id_accs),
        mean_id_to_emb_r2=np.mean(id_to_emb_r2s),
        std_id_to_emb_r2=np.std(id_to_emb_r2s),
        ci95_id_to_emb=compute_confidence_interval(id_to_emb_r2s),
        mean_emb_to_emb_r2=np.mean(emb_to_emb_r2s),
        std_emb_to_emb_r2=np.std(emb_to_emb_r2s),
        ci95_emb_to_emb=compute_confidence_interval(emb_to_emb_r2s),
        mean_perceptibility=np.mean(perceptibilities),
        std_perceptibility=np.std(perceptibilities),
        ci95_perceptibility=compute_confidence_interval(perceptibilities),
        per_seed_id_to_id=id_to_id_accs,
        per_seed_emb_to_id=emb_to_id_accs,
        per_seed_id_to_emb=id_to_emb_r2s,
        per_seed_emb_to_emb=emb_to_emb_r2s,
        per_seed_perceptibility=perceptibilities,
    )

    return result


def parse_checkpoint_dir(dirname: str) -> Optional[Dict]:
    """Parse checkpoint directory name to extract configuration."""
    # Format: trojanstego_{model}_{training}_{encoding}[_{bucket}]
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
            # Check if it has a 'final' subdirectory (completed training)
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
        default=100000,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=5,
        help="Number of random seeds (train/test splits + classifier init) for robustness"
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
        help="Models to evaluate (if not auto-discovering)"
    )
    parser.add_argument(
        "--encoding-modes",
        type=str,
        nargs="+",
        default=None,
        help="Encoding modes to evaluate (if not auto-discovering)"
    )
    parser.add_argument(
        "--bucket-modes",
        type=str,
        nargs="+",
        default=None,
        help="Bucket modes to evaluate (if not auto-discovering)"
    )
    args = parser.parse_args()

    # Auto-discover configurations from checkpoint directory
    if args.auto_discover:
        print("Auto-discovering configurations from checkpoint directory...")
        discovered = discover_configurations(args.checkpoint_dir)

        if not discovered:
            print(f"No configurations found in {args.checkpoint_dir}")
            print("Make sure checkpoints have a 'final' subdirectory.")
            return []

        print(f"Found {len(discovered)} configurations:")
        for cfg in discovered:
            print(f"  - {cfg['model']} / {cfg['encoding_mode']} / {cfg['bucket_mode']}")

        # Extract unique models, encoding modes, bucket modes
        models = list(set(cfg['model'] for cfg in discovered))
        encoding_modes = list(set(cfg['encoding_mode'] for cfg in discovered))
        bucket_modes = list(set(cfg['bucket_mode'] for cfg in discovered))

        # Build evaluation configs as (model, encoding, bucket) tuples
        eval_configs = [(cfg['model'], cfg['encoding_mode'], cfg['bucket_mode']) for cfg in discovered]
    else:
        # Use manually specified values, with defaults
        models = args.models or ["llama", "ministral"]
        encoding_modes = args.encoding_modes or ["ascii", "embedding", "embedding_only"]
        bucket_modes = args.bucket_modes or ["parity", "embedding"]

        # Build all combinations
        eval_configs = [
            (m, e, b) for m in models for e in encoding_modes for b in bucket_modes
        ]

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

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []

    # Group configs by model for efficient loading
    from collections import defaultdict
    configs_by_model = defaultdict(list)
    for model_name, encoding_mode, bucket_mode in eval_configs:
        configs_by_model[model_name].append((encoding_mode, bucket_mode))

    for model_name in configs_by_model:
        print(f"\n{'=' * 70}")
        print(f"MODEL: {model_name}")
        print("=" * 70)

        # Load tokenizer and embeddings once per model
        try:
            tokenizer, embeddings = load_model_and_tokenizer(model_name)
        except Exception as e:
            print(f"  Error loading model: {e}")
            continue

        vocab_size = embeddings.shape[0]
        embed_dim = embeddings.shape[1]
        print(f"  Vocab size: {vocab_size}, Embed dim: {embed_dim}")

        # Load config
        model_registry = {
            "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
            "ministral": "mistralai/Ministral-8B-Instruct-2410",
        }

        # Get encoding modes for this model
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

        # Cache bucket assignments by bucket mode
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

        # Iterate over specific configs for this model
        for encoding_mode, bucket_mode in configs_by_model[model_name]:
            print(f"\n{'-' * 50}")
            print(f"Config: {encoding_mode} / {bucket_mode}")
            print("-" * 50)

            # Get bucket assignments from cache
            bucket_assignments = bucket_cache.get(bucket_mode)
            if bucket_assignments is None:
                print(f"  Skipping - no bucket assignments for {bucket_mode}")
                continue

            # Create config for this encoding mode
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

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<10} {'Encoding':<12} {'Bucket':<10} "
          f"{'ID→ID %':<14} {'Emb→ID %':<14} {'Percept %':<14}")
    print("-" * 84)

    for r in all_results:
        print(f"{r.model_name:<10} {r.encoding_mode:<12} {r.bucket_mode:<10} "
              f"{r.mean_id_to_id_accuracy*100:>5.1f} +/- {r.std_id_to_id_accuracy*100:<5.1f} "
              f"{r.mean_emb_to_id_accuracy*100:>5.1f} +/- {r.std_emb_to_id_accuracy*100:<5.1f} "
              f"{r.mean_perceptibility*100:>5.1f} +/- {r.std_perceptibility*100:<5.1f}")

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
            "checkpoint_dir": args.checkpoint_dir,
        },
        "results": [asdict(r) for r in all_results],
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    main()
