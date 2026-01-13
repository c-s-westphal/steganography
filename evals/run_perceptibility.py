#!/usr/bin/env python3
"""
Perceptibility Evaluation: Measure how recoverable secrets are from steganographic outputs.

Given bucket knowledge (parity or embedding assignments), can an adversary recover:
1. The encoded bits from output tokens?
2. The original secret from the recovered bits?

This measures the "informed adversary" threat model - someone who knows the
steganography scheme but intercepts the output.

For ideal/synthetic samples: should be 100% (by construction)
For real model outputs: may be <100% if model doesn't always pick correct bucket

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

    # Bit-level recovery (given bucket knowledge)
    bit_recovery_accuracy: float

    # Secret-level recovery (given bucket knowledge + encoding knowledge)
    secret_recovery_accuracy: float
    letter_recovery_accuracy: float  # Per-letter accuracy

    # Breakdown by position
    per_position_bit_accuracy: List[float] = field(default_factory=list)
    per_letter_accuracy: List[float] = field(default_factory=list)


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


def decode_bits_to_secret(
    bits: str,
    encoding_mode: str,
    config: Config,
    model=None,
    tokenizer=None,
    embedding_key_config=None,
    embedding_only_config=None,
    original_secret: str = None,  # Needed for embedding modes to derive the key
) -> Optional[str]:
    """
    Decode bits back to secret using the specified encoding mode.

    For ASCII: straightforward decode
    For embedding modes: need original secret to derive the key (chicken-egg problem)
    """
    if encoding_mode == "ascii":
        # Decode ASCII: 8 bits per character
        try:
            secret = ""
            for i in range(0, len(bits), 8):
                byte = bits[i:i+8]
                char_code = int(byte, 2)
                if 97 <= char_code <= 122:  # lowercase a-z
                    secret += chr(char_code)
                else:
                    return None  # Invalid character
            return secret
        except:
            return None

    elif encoding_mode in ["embedding", "xor", "embedding_only", "embedding_xor"]:
        # For embedding-based modes, we need to check all possible secrets
        # This is the "informed adversary" model - they know the scheme
        # but must brute-force the secret space
        #
        # For now, if we have the original secret, verify the bits match
        if original_secret:
            expected_bits = encode_secret_to_bits(
                original_secret, encoding_mode, config,
                model, tokenizer, embedding_key_config, embedding_only_config
            )
            return original_secret if bits == expected_bits else None
        return None

    return None


def generate_ideal_samples(
    secrets: List[str],
    bucket_assignments: torch.Tensor,
    encoding_mode: str,
    config: Config,
    model=None,
    tokenizer=None,
    embedding_key_config=None,
    embedding_only_config=None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Generate ideal output token sequences by sampling from bucket assignments.

    Returns:
        output_token_ids: [num_samples, 32] array of token IDs
        all_bits: List of bit strings for each sample
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
    all_bits = []

    for i, secret in enumerate(secrets):
        # Get bits to encode for this secret
        bits_str = encode_secret_to_bits(
            secret, encoding_mode, config,
            model, tokenizer, embedding_key_config, embedding_only_config
        )
        all_bits.append(bits_str)

        # Convert bits string to list of ints
        bits = [int(b) for b in bits_str]

        # For each bit, sample a token from the corresponding bucket
        for j, bit in enumerate(bits):
            if bit == 0:
                output_token_ids[i, j] = np.random.choice(bucket_0_tokens)
            else:
                output_token_ids[i, j] = np.random.choice(bucket_1_tokens)

    return output_token_ids, all_bits


def extract_bits_from_tokens(
    output_token_ids: np.ndarray,
    bucket_assignments: torch.Tensor,
) -> List[str]:
    """
    Extract bits from output tokens using bucket assignments.

    For each token, look up its bucket assignment to get the bit.
    """
    bucket_np = bucket_assignments.cpu().numpy() if bucket_assignments.is_cuda else bucket_assignments.numpy()

    num_samples, num_bits = output_token_ids.shape
    recovered_bits = []

    for i in range(num_samples):
        bits_str = ""
        for j in range(num_bits):
            token_id = output_token_ids[i, j]
            bit = bucket_np[token_id]
            bits_str += str(int(bit))
        recovered_bits.append(bits_str)

    return recovered_bits


def run_perceptibility_evaluation(
    model_name: str,
    encoding_mode: str,
    bucket_mode: str,
    tokenizer,
    bucket_assignments: torch.Tensor,
    config: Config,
    num_samples: int = 10000,
    model=None,
    embedding_key_config=None,
    embedding_only_config=None,
) -> PerceptibilityResult:
    """
    Run perceptibility evaluation.

    Measures:
    1. Bit recovery: Can we extract the correct bits from output tokens?
    2. Secret recovery: Can we decode the bits back to the original secret?
    """
    print(f"\nEvaluating: {model_name} / {encoding_mode} / {bucket_mode}")
    print(f"  Samples: {num_samples}")

    # Generate random secrets
    secrets = generate_random_secrets(num_samples, config.secret_alphabet,
                                     config.secret_length, seed=42)

    # Generate ideal output token sequences
    print(f"  Generating {num_samples} ideal samples...")
    output_token_ids, original_bits = generate_ideal_samples(
        secrets, bucket_assignments, encoding_mode, config,
        model=model, tokenizer=tokenizer,
        embedding_key_config=embedding_key_config,
        embedding_only_config=embedding_only_config,
    )

    # Extract bits from output tokens
    print(f"  Extracting bits from output tokens...")
    recovered_bits = extract_bits_from_tokens(output_token_ids, bucket_assignments)

    # Calculate bit-level accuracy
    bit_correct = 0
    total_bits = 0
    per_position_correct = [0] * 32

    for orig, recov in zip(original_bits, recovered_bits):
        for pos, (o, r) in enumerate(zip(orig, recov)):
            if o == r:
                bit_correct += 1
                per_position_correct[pos] += 1
            total_bits += 1

    bit_accuracy = bit_correct / total_bits
    per_position_accuracy = [c / num_samples for c in per_position_correct]

    # Calculate secret-level accuracy
    print(f"  Decoding secrets from recovered bits...")
    secret_correct = 0
    letter_correct = [0, 0, 0, 0]

    for i, (orig_secret, recov_bits) in enumerate(zip(secrets, recovered_bits)):
        # Decode the recovered bits
        decoded_secret = decode_bits_to_secret(
            recov_bits, encoding_mode, config,
            model, tokenizer, embedding_key_config, embedding_only_config,
            original_secret=orig_secret  # For embedding modes
        )

        if decoded_secret == orig_secret:
            secret_correct += 1

        # For ASCII mode, we can check per-letter accuracy
        if encoding_mode == "ascii" and decoded_secret:
            for j in range(min(4, len(decoded_secret), len(orig_secret))):
                if decoded_secret[j] == orig_secret[j]:
                    letter_correct[j] += 1

    secret_accuracy = secret_correct / num_samples
    letter_accuracy = [c / num_samples for c in letter_correct]
    overall_letter_accuracy = sum(letter_correct) / (num_samples * 4)

    print(f"  Bit recovery accuracy: {bit_accuracy:.4f}")
    print(f"  Secret recovery accuracy: {secret_accuracy:.4f}")
    print(f"  Letter recovery accuracy: {overall_letter_accuracy:.4f}")

    result = PerceptibilityResult(
        model_name=model_name,
        encoding_mode=encoding_mode,
        bucket_mode=bucket_mode,
        num_samples=num_samples,
        bit_recovery_accuracy=bit_accuracy,
        secret_recovery_accuracy=secret_accuracy,
        letter_recovery_accuracy=overall_letter_accuracy,
        per_position_bit_accuracy=per_position_accuracy,
        per_letter_accuracy=letter_accuracy,
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
    print("PERCEPTIBILITY EVALUATION (Bit Recovery)")
    print("=" * 70)
    print(f"Samples: {args.num_samples}")
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
            tokenizer, model_id = load_model_and_tokenizer(model_name)
        except Exception as e:
            print(f"  Error loading model: {e}")
            continue

        vocab_size = tokenizer.vocab_size
        print(f"  Vocab size: {vocab_size}")

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
                    bucket_assignments=bucket_assignments,
                    config=config,
                    num_samples=args.num_samples,
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
    print(f"{'Model':<12} {'Encoding':<15} {'Bucket':<10} {'Bit Acc':<10} {'Secret Acc':<12} {'Letter Acc':<10}")
    print("-" * 79)

    for r in all_results:
        print(f"{r.model_name:<12} {r.encoding_mode:<15} {r.bucket_mode:<10} "
              f"{r.bit_recovery_accuracy:>8.2%}  {r.secret_recovery_accuracy:>10.2%}  "
              f"{r.letter_recovery_accuracy:>8.2%}")

    # Save results
    pod_suffix = f"_{args.pod_name}" if args.pod_name else ""
    output_file = os.path.join(args.output_dir, f"perceptibility_summary{pod_suffix}.json")

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "num_samples": args.num_samples,
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
