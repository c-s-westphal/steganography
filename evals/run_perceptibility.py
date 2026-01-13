#!/usr/bin/env python3
"""
Perceptibility Evaluation: Measure how recoverable secrets are from steganographic outputs.

Two independent axes (kept separate, not combined):

Axis 1: Tokens → Bits (Bucket Detectability)
    Can an adversary extract the bucketed bits from output tokens?
    - 1a: XGBoost on binary token features (can discover parity via LSB)
    - 1b: MLP on embeddings (can discover embedding bucket boundaries)

Axis 2: Bits → Secret (Encoding Obfuscation)
    Can an adversary decode the secret from extracted bits?
    - Classifier predicting secret letters from extracted bits

Both axes are reported separately to preserve information about each attack vector.

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
    """Results for perceptibility evaluation.

    Two independent axes of perceptibility:
    - Axis 1 (Tokens → Bits): Can adversary extract bucket bits from output tokens?
      - 1a: XGBoost on binary token features (detects parity bucketing)
      - 1b: MLP on embeddings (detects embedding bucket boundaries)
    - Axis 2 (Bits → Secret): Can adversary decode secret from extracted bits?

    These axes are kept SEPARATE - not combined via max or product.
    """
    model_name: str
    encoding_mode: str
    bucket_mode: str
    num_samples: int
    num_seeds: int

    # Axis 1a: Tokens → Bits via binary token features (XGBoost)
    mean_token_to_bit_accuracy: float
    std_token_to_bit_accuracy: float

    # Axis 1b: Tokens → Bits via embeddings (MLP)
    mean_emb_to_bit_accuracy: float
    std_emb_to_bit_accuracy: float

    # Axis 2: Bits → Secret (Encoding Obfuscation)
    mean_bit_to_secret_accuracy: float
    std_bit_to_secret_accuracy: float

    # Per-seed details for computing confidence intervals
    per_seed_token_to_bit: List[float] = field(default_factory=list)
    per_seed_emb_to_bit: List[float] = field(default_factory=list)
    per_seed_bit_to_secret: List[float] = field(default_factory=list)


def load_model_and_tokenizer(model_name: str):
    """Load model and tokenizer, return embeddings."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_registry = {
        "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama70b": "meta-llama/Llama-3.3-70B-Instruct",
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

    return tokenizer, embeddings, model_id


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


def generate_samples(
    secrets: List[str],
    bucket_assignments: torch.Tensor,
    encoding_mode: str,
    config: Config,
    model=None,
    tokenizer=None,
    embedding_key_config=None,
    embedding_only_config=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate output token sequences and corresponding bits.

    Returns:
        output_token_ids: [num_samples, 32] array of token IDs
        bits: [num_samples, 32] array of bits (0 or 1)
    """
    bucket_np = bucket_assignments.cpu().numpy() if bucket_assignments.is_cuda else bucket_assignments.numpy()
    bucket_0_tokens = np.where(bucket_np == 0)[0]
    bucket_1_tokens = np.where(bucket_np == 1)[0]

    if len(bucket_0_tokens) == 0 or len(bucket_1_tokens) == 0:
        raise ValueError("One of the buckets is empty!")

    num_samples = len(secrets)
    num_bits = config.secret_bits  # 32

    output_token_ids = np.zeros((num_samples, num_bits), dtype=np.int64)
    bits = np.zeros((num_samples, num_bits), dtype=np.int32)

    for i, secret in enumerate(secrets):
        bits_str = encode_secret_to_bits(
            secret, encoding_mode, config,
            model, tokenizer, embedding_key_config, embedding_only_config
        )
        bits[i] = [int(b) for b in bits_str]

        for j, bit in enumerate(bits[i]):
            if bit == 0:
                output_token_ids[i, j] = np.random.choice(bucket_0_tokens)
            else:
                output_token_ids[i, j] = np.random.choice(bucket_1_tokens)

    return output_token_ids, bits


def token_ids_to_binary_features(token_ids: np.ndarray, num_bits: int = 17) -> np.ndarray:
    """
    Convert token IDs to binary representation.

    Each token ID becomes `num_bits` binary features.
    LSB (index 0) directly encodes parity.

    Args:
        token_ids: [num_samples, 32] array of token IDs
        num_bits: Number of bits per token ID (17 bits covers up to 131k vocab)

    Returns:
        binary_features: [num_samples, 32 * num_bits] array
    """
    num_samples, seq_len = token_ids.shape
    binary_features = np.zeros((num_samples, seq_len * num_bits), dtype=np.float32)

    for i in range(seq_len):
        for b in range(num_bits):
            binary_features[:, i * num_bits + b] = (token_ids[:, i] >> b) & 1

    return binary_features


def secrets_to_labels(secrets: List[str]) -> np.ndarray:
    """Convert secrets to numeric labels (0-25 for each letter)."""
    labels = np.array([[ord(c) - ord('a') for c in secret] for secret in secrets])
    return labels


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
    """Train XGBoost/RandomForest classifier for binary classification."""
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


def train_mlp_classifier(X_train, y_train, X_test, y_test, seed: int = 42) -> float:
    """Train MLP classifier."""
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
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
    Run full perceptibility evaluation.

    Stage 1: Tokens → Bits (bucket detectability)
    Stage 2: Bits → Secret (encoding obfuscation)
    """
    print(f"\nEvaluating: {model_name} / {encoding_mode} / {bucket_mode}")
    print(f"  Samples: {num_samples}, Seeds: {num_seeds}")

    # Fit PCA on vocabulary embeddings for Stage 1
    print("  Fitting PCA on vocabulary embeddings...")
    pca = PCA(n_components=pca_dims, random_state=42)
    pca.fit(embeddings.numpy())
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    # Results storage (axes kept separate)
    token_to_bit_accs = []   # Axis 1a
    emb_to_bit_accs = []     # Axis 1b
    bit_to_secret_accs = []  # Axis 2

    for seed in range(num_seeds):
        print(f"  Seed {seed + 1}/{num_seeds}...")
        np.random.seed(seed * 1000)

        # Generate random secrets
        secrets = generate_random_secrets(num_samples, config.secret_alphabet,
                                         config.secret_length, seed=seed * 1000)

        # Generate output tokens and bits
        print(f"    Generating {num_samples} samples...")
        output_token_ids, bits = generate_samples(
            secrets, bucket_assignments, encoding_mode, config,
            model=model, tokenizer=tokenizer,
            embedding_key_config=embedding_key_config,
            embedding_only_config=embedding_only_config,
        )

        secret_labels = secrets_to_labels(secrets)

        # ============================================================
        # STAGE 1: Tokens → Bits (Bucket Detectability)
        # ============================================================

        # Prepare features for Stage 1
        # Binary features from token IDs (LSB = parity)
        X_binary = token_ids_to_binary_features(output_token_ids)

        # Embeddings of output tokens (PCA reduced)
        print(f"    Computing output embeddings...")
        output_embeds_flat = embeddings[output_token_ids.flatten()].numpy()
        output_embeds_pca = pca.transform(output_embeds_flat)
        X_embeddings = output_embeds_pca.reshape(num_samples, 32, pca_dims)

        # Flatten bits for per-position prediction
        y_bits = bits  # [num_samples, 32]

        # Train/test split (same split for all)
        indices = np.arange(num_samples)
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed)

        # Stage 1a: XGBoost on binary token features → Bits
        print(f"    Stage 1a: XGBoost(binary tokens) → Bits...")
        token_to_bit_acc = 0
        for pos in range(32):
            # Features: binary representation of token at this position
            X_pos = X_binary[:, pos * 17:(pos + 1) * 17]
            y_pos = y_bits[:, pos]

            acc = train_xgboost_classifier(
                X_pos[train_idx], y_pos[train_idx],
                X_pos[test_idx], y_pos[test_idx],
                seed=seed
            )
            token_to_bit_acc += acc / 32
        token_to_bit_accs.append(token_to_bit_acc)
        print(f"      Token→Bit accuracy: {token_to_bit_acc:.4f}")

        # Stage 1b: MLP on embeddings → Bits
        print(f"    Stage 1b: MLP(embeddings) → Bits...")
        emb_to_bit_acc = 0
        for pos in range(32):
            # Features: PCA embedding of token at this position
            X_pos = X_embeddings[:, pos, :]
            y_pos = y_bits[:, pos]

            acc = train_mlp_classifier(
                X_pos[train_idx], y_pos[train_idx],
                X_pos[test_idx], y_pos[test_idx],
                seed=seed
            )
            emb_to_bit_acc += acc / 32
        emb_to_bit_accs.append(emb_to_bit_acc)
        print(f"      Emb→Bit accuracy: {emb_to_bit_acc:.4f}")

        # ============================================================
        # AXIS 2: Bits → Secret (Encoding Obfuscation)
        # ============================================================
        print(f"    Axis 2: Bits → Secret...")

        X_bits = bits.astype(np.float32)
        bit_to_secret_acc = 0
        for pos in range(4):
            acc = train_xgboost_classifier(
                X_bits[train_idx], secret_labels[train_idx, pos],
                X_bits[test_idx], secret_labels[test_idx, pos],
                seed=seed
            )
            bit_to_secret_acc += acc / 4
        bit_to_secret_accs.append(bit_to_secret_acc)
        print(f"      Bit→Secret accuracy: {bit_to_secret_acc:.4f}")

    # Compute statistics
    result = PerceptibilityResult(
        model_name=model_name,
        encoding_mode=encoding_mode,
        bucket_mode=bucket_mode,
        num_samples=num_samples,
        num_seeds=num_seeds,
        mean_token_to_bit_accuracy=np.mean(token_to_bit_accs),
        std_token_to_bit_accuracy=np.std(token_to_bit_accs),
        mean_emb_to_bit_accuracy=np.mean(emb_to_bit_accs),
        std_emb_to_bit_accuracy=np.std(emb_to_bit_accs),
        mean_bit_to_secret_accuracy=np.mean(bit_to_secret_accs),
        std_bit_to_secret_accuracy=np.std(bit_to_secret_accs),
        per_seed_token_to_bit=token_to_bit_accs,
        per_seed_emb_to_bit=emb_to_bit_accs,
        per_seed_bit_to_secret=bit_to_secret_accs,
    )

    return result


def parse_checkpoint_dir(dirname: str) -> Optional[Dict]:
    """Parse checkpoint directory name to extract configuration."""
    if not dirname.startswith("trojanstego_"):
        return None

    rest = dirname[len("trojanstego_"):]

    model_map = {
        "meta-llama-3.1-8b-instruct": "llama",
        "llama-3.3-70b-instruct": "llama70b",
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
        encoding_modes = args.encoding_modes or ["ascii", "embedding", "embedding_only", "xor", "embedding_xor"]
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
    print()
    print("Perceptibility evaluation (axes kept separate):")
    print("  Axis 1: Tokens → Bits (Bucket Detectability)")
    print("    - 1a: XGBoost on binary token features (can find parity via LSB)")
    print("    - 1b: MLP on embeddings (can find embedding bucket boundaries)")
    print("  Axis 2: Bits → Secret (Encoding Obfuscation)")
    print("    - ASCII: ~100% | XOR/Embedding modes: obfuscated")
    print("  NOTE: Axes reported separately, not combined")

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []

    # Group configs by model
    configs_by_model = defaultdict(list)
    for model_name, encoding_mode, bucket_mode in eval_configs:
        configs_by_model[model_name].append((encoding_mode, bucket_mode))

    model_registry = {
        "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama70b": "meta-llama/Llama-3.3-70B-Instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
        "ministral": "mistralai/Ministral-8B-Instruct-2410",
    }

    for model_name in configs_by_model:
        print(f"\n{'=' * 70}")
        print(f"MODEL: {model_name}")
        print("=" * 70)

        try:
            tokenizer, embeddings, model_id = load_model_and_tokenizer(model_name)
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
    print("\n" + "=" * 80)
    print("SUMMARY (Axes kept separate)")
    print("=" * 80)
    print(f"{'Model':<10} {'Encoding':<14} {'Bucket':<10} {'Axis1a:Token→Bit':<18} {'Axis1b:Emb→Bit':<18} {'Axis2:Bit→Secret':<18}")
    print("-" * 80)

    for r in all_results:
        print(f"{r.model_name:<10} {r.encoding_mode:<14} {r.bucket_mode:<10} "
              f"{r.mean_token_to_bit_accuracy:>6.2%} ±{r.std_token_to_bit_accuracy:>4.2%}  "
              f"{r.mean_emb_to_bit_accuracy:>6.2%} ±{r.std_emb_to_bit_accuracy:>4.2%}  "
              f"{r.mean_bit_to_secret_accuracy:>6.2%} ±{r.std_bit_to_secret_accuracy:>4.2%}")

    # Save results
    pod_suffix = f"_{args.pod_name}" if args.pod_name else ""
    output_file = os.path.join(args.output_dir, f"perceptibility_summary{pod_suffix}.json")

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "description": "Perceptibility axes (kept separate): Axis1a (Token→Bit), Axis1b (Emb→Bit), Axis2 (Bit→Secret)",
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
