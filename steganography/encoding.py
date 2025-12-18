"""
Embedding-based steganographic encoding.

Three encoding modes for deriving 32 bits from a 4-letter secret:
- "ascii": Direct ASCII encoding (baseline)
- "embedding": Embedding key only (obfuscated)
- "xor": ASCII XOR embedding key (obfuscated)

Embedding key derivation:
- For each bit position i (0-31):
  - Use letter (i % 4) from secret
  - Project that letter's embedding onto projection vector (seed=1000+i)
  - Compare to median threshold → 0 or 1

Output bucket assignments:
- Project all vocab embeddings onto projection vector (seed=42)
- Threshold at median for balanced buckets
- Constrained generation selects tokens in correct bucket
"""

import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass
import json
import os


@dataclass
class EmbeddingKeyConfig:
    """Precomputed data for embedding key derivation."""
    seed_base: int
    num_bits: int
    hidden_dim: int
    projections: torch.Tensor    # [num_bits, hidden_dim]
    thresholds: torch.Tensor     # [num_bits]
    letter_token_ids: dict       # {letter: token_id} for a-z


def get_projection_vector(hidden_dim: int, seed: int) -> torch.Tensor:
    """
    Get projection vector for bucket assignment.

    The seed is the SECRET - it defines which direction separates buckets.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    projection = torch.randn(hidden_dim, generator=generator)
    projection = projection / projection.norm()
    return projection


def compute_bucket_assignments(
    model,
    seed: int,
) -> Tuple[torch.Tensor, float]:
    """
    Compute bucket assignments from output embeddings.

    Args:
        model: Language model
        seed: Projection seed (THE SECRET)

    Returns:
        bucket_assignments: [vocab_size] tensor of 0s and 1s
        threshold: The median threshold used
    """
    W = model.get_output_embeddings().weight.detach()

    projection = get_projection_vector(W.shape[1], seed).to(device=W.device, dtype=W.dtype)
    scores = W @ projection

    threshold = scores.median().item()
    bucket_assignments = (scores > threshold).int()

    return bucket_assignments, threshold


def secret_to_bits(secret: str, config=None) -> str:
    """
    Convert 4-letter lowercase secret to 32-bit ASCII string.

    Each letter is 8 bits ASCII.
    'a' = 97  = 01100001
    'z' = 122 = 01111010

    "abcd" -> "01100001011000100110001101100100" (32 bits)
    """
    if config is None:
        from .config import get_config
        config = get_config()

    assert len(secret) == config.secret_length, \
        f"Secret must be {config.secret_length} letters, got {len(secret)}"
    assert all(c in config.secret_alphabet for c in secret), \
        f"Secret must only contain characters from alphabet: {secret}"
    return ''.join(format(ord(c), '08b') for c in secret)


def bits_to_secret(bits: str, config=None) -> str:
    """
    Convert 32-bit string back to 4-letter secret.
    """
    if config is None:
        from .config import get_config
        config = get_config()

    expected_bits = config.secret_length * 8
    assert len(bits) == expected_bits, f"Expected {expected_bits} bits, got {len(bits)}"

    chars = []
    for i in range(config.secret_length):
        char_bits = bits[i*8:(i+1)*8]
        ascii_val = int(char_bits, 2)
        chars.append(chr(ascii_val))

    return ''.join(chars)


def xor_bits(a: str, b: str) -> str:
    """XOR two bit strings of equal length."""
    assert len(a) == len(b), f"Length mismatch: {len(a)} vs {len(b)}"
    return ''.join(str(int(x) ^ int(y)) for x, y in zip(a, b))


def precompute_embedding_key_config(
    model,
    tokenizer,
    seed_base: int = 1000,
    num_bits: int = 32,
) -> EmbeddingKeyConfig:
    """
    Precompute projections and thresholds for embedding key derivation.

    This should be called once before bulk data generation for efficiency.

    Args:
        model: Language model (for output embeddings)
        tokenizer: Tokenizer (for letter -> token ID mapping)
        seed_base: Base seed for projections (default 1000)
        num_bits: Number of bits in embedding key (default 32)

    Returns:
        EmbeddingKeyConfig with precomputed projections and thresholds
    """
    W = model.get_output_embeddings().weight.detach()
    hidden_dim = W.shape[1]

    # Precompute projection vectors and thresholds for each bit position
    projections = []
    thresholds = []

    for i in range(num_bits):
        proj = get_projection_vector(hidden_dim, seed=seed_base + i)
        proj = proj.to(device=W.device, dtype=W.dtype)
        projections.append(proj)

        # Compute median threshold over all vocab embeddings
        scores = W @ proj
        threshold = scores.median().item()
        thresholds.append(threshold)

    # Precompute token IDs for all lowercase letters
    letter_token_ids = {}
    for letter in "abcdefghijklmnopqrstuvwxyz":
        token_ids = tokenizer.encode(letter, add_special_tokens=False)
        letter_token_ids[letter] = token_ids[0]

    return EmbeddingKeyConfig(
        seed_base=seed_base,
        num_bits=num_bits,
        hidden_dim=hidden_dim,
        projections=torch.stack(projections),
        thresholds=torch.tensor(thresholds),
        letter_token_ids=letter_token_ids,
    )


def derive_embedding_key(
    secret: str,
    model,
    tokenizer,
    embedding_key_config: Optional[EmbeddingKeyConfig] = None,
    seed_base: int = 1000,
) -> str:
    """
    Derive 32-bit key from secret letter embeddings.

    For each bit position i:
    - Use letter (i % 4) from secret
    - Project that letter's embedding onto projection vector (seed=seed_base+i)
    - Compare to median threshold → 0 or 1

    Args:
        secret: 4-letter lowercase secret
        model: Language model (for output embeddings)
        tokenizer: Tokenizer (for letter -> token ID)
        embedding_key_config: Precomputed config (for efficiency)
        seed_base: Base seed if config not provided

    Returns:
        32-bit key string
    """
    W = model.get_output_embeddings().weight.detach()

    # Use precomputed config if available
    if embedding_key_config is None:
        embedding_key_config = precompute_embedding_key_config(
            model, tokenizer, seed_base
        )

    bits = []
    for i in range(embedding_key_config.num_bits):
        # Which letter of the secret to use for this bit
        letter_idx = i % len(secret)
        letter = secret[letter_idx]

        # Get token ID for this letter
        token_id = embedding_key_config.letter_token_ids[letter]

        # Get embedding and project
        embedding = W[token_id]
        proj = embedding_key_config.projections[i].to(
            device=embedding.device, dtype=embedding.dtype
        )
        score = (embedding @ proj).item()

        # Compare to threshold
        threshold = embedding_key_config.thresholds[i].item()
        bit = 1 if score > threshold else 0
        bits.append(str(bit))

    return ''.join(bits)


def get_bits_to_encode(
    secret: str,
    mode: str,
    model=None,
    tokenizer=None,
    embedding_key_config: Optional[EmbeddingKeyConfig] = None,
    config=None,
) -> str:
    """
    Get the 32 bits to encode in output tokens based on encoding mode.

    Args:
        secret: 4-letter lowercase secret
        mode: "ascii" | "embedding" | "xor"
        model: Language model (required for "embedding" and "xor" modes)
        tokenizer: Tokenizer (required for "embedding" and "xor" modes)
        embedding_key_config: Precomputed config (for efficiency)
        config: Config object (optional, for secret validation)

    Returns:
        32-bit string to encode in output tokens
    """
    if mode == "ascii":
        # Direct ASCII encoding
        return secret_to_bits(secret, config)

    elif mode == "embedding":
        # Embedding key only
        if model is None or tokenizer is None:
            raise ValueError("model and tokenizer required for embedding mode")
        return derive_embedding_key(secret, model, tokenizer, embedding_key_config)

    elif mode == "xor":
        # ASCII XOR embedding key
        if model is None or tokenizer is None:
            raise ValueError("model and tokenizer required for xor mode")
        ascii_bits = secret_to_bits(secret, config)
        embedding_key = derive_embedding_key(
            secret, model, tokenizer, embedding_key_config
        )
        return xor_bits(ascii_bits, embedding_key)

    else:
        raise ValueError(f"Unknown encoding mode: {mode}")


def derive_key_from_prompt_embeddings(
    prompt_token_ids: List[int],
    bucket_assignments: torch.Tensor,
    num_bits: int = 32,
) -> str:
    """
    Derive key from first num_bits prompt token embedding buckets.

    K[i] = bucket_assignment[prompt_token_ids[i]]

    Args:
        prompt_token_ids: Token IDs of the prompt
        bucket_assignments: Pre-computed bucket assignments [vocab_size]
        num_bits: Key length (default 32)

    Returns:
        Key bit string (e.g., "10110010101001001011001010100100")
    """
    key_bits = []
    for i in range(num_bits):
        if i < len(prompt_token_ids):
            token_id = prompt_token_ids[i]
            bit = bucket_assignments[token_id].item()
        else:
            bit = 0  # Pad with 0s if prompt is too short
        key_bits.append(str(bit))
    return ''.join(key_bits)


def get_target_bits(secret: str, key: str, config=None) -> str:
    """
    Compute target bits: T = S XOR K

    Args:
        secret: Secret string (e.g., "abcd")
        key: Key string derived from prompt embedding buckets

    Returns:
        Target bit string to encode
    """
    secret_bits = secret_to_bits(secret, config)
    return xor_bits(secret_bits, key)


def decode_bits_from_tokens(
    token_ids: List[int],
    bucket_assignments: torch.Tensor,
    num_bits: int = 32,
) -> str:
    """Decode bits from token embedding buckets."""
    bits = []
    for i, token_id in enumerate(token_ids[:num_bits]):
        bit = bucket_assignments[token_id].item()
        bits.append(str(bit))
    return ''.join(bits)


def recover_secret_bits(transmitted_bits: str, key: str) -> str:
    """Recover original secret bits: S = T XOR K"""
    return xor_bits(transmitted_bits, key)


def recover_secret(transmitted_bits: str, key: str, config=None) -> str:
    """Recover original secret word from transmitted bits and key."""
    secret_bits = recover_secret_bits(transmitted_bits, key)
    try:
        return bits_to_secret(secret_bits, config)
    except:
        if config is None:
            from .config import get_config
            config = get_config()
        return "?" * config.secret_length  # Invalid decoding


def compute_bit_accuracy(
    token_ids: List[int],
    target_bits: str,
    bucket_assignments: torch.Tensor,
) -> float:
    """Compute fraction of correctly encoded bits."""
    if len(token_ids) == 0:
        return 0.0

    num_bits = min(len(token_ids), len(target_bits))
    correct = 0

    for i in range(num_bits):
        actual = bucket_assignments[token_ids[i]].item()
        target = int(target_bits[i])
        if actual == target:
            correct += 1

    return correct / num_bits


@dataclass
class BucketConfig:
    """Saved bucket configuration."""
    projection_seed: int
    hidden_dim: int
    threshold: float
    vocab_size: int

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'BucketConfig':
        with open(path) as f:
            return cls(**json.load(f))


def save_bucket_assignments(
    bucket_assignments: torch.Tensor,
    config: BucketConfig,
    output_dir: str,
):
    """Save bucket assignments and config."""
    os.makedirs(output_dir, exist_ok=True)
    torch.save(bucket_assignments, os.path.join(output_dir, 'bucket_assignments.pt'))
    config.save(os.path.join(output_dir, 'bucket_config.json'))
    print(f"Saved bucket assignments to {output_dir}")


def load_bucket_assignments(output_dir: str) -> Tuple[torch.Tensor, BucketConfig]:
    """Load saved bucket assignments."""
    assignments = torch.load(os.path.join(output_dir, 'bucket_assignments.pt'))
    config = BucketConfig.load(os.path.join(output_dir, 'bucket_config.json'))
    return assignments, config


def analyze_buckets(bucket_assignments: torch.Tensor, tokenizer=None):
    """Print bucket statistics."""
    total = len(bucket_assignments)
    ones = bucket_assignments.sum().item()
    zeros = total - ones

    print(f"Bucket balance: {zeros} (0) / {ones} (1) = {zeros/total:.1%} / {ones/total:.1%}")

    if tokenizer:
        b0_indices = (bucket_assignments == 0).nonzero().squeeze()
        b1_indices = (bucket_assignments == 1).nonzero().squeeze()

        # Get first 5 tokens from each bucket
        b0_sample = b0_indices[:5].tolist() if b0_indices.dim() > 0 else [b0_indices.item()]
        b1_sample = b1_indices[:5].tolist() if b1_indices.dim() > 0 else [b1_indices.item()]

        print(f"Sample bucket 0: {[tokenizer.decode([i]) for i in b0_sample]}")
        print(f"Sample bucket 1: {[tokenizer.decode([i]) for i in b1_sample]}")


@dataclass
class DecodingResult:
    """Result of decoding bits from token IDs."""
    transmitted_bits: str
    key: str
    recovered_secret_bits: str
    recovered_secret: str
    num_bits: int

    @property
    def success(self) -> bool:
        from .config import get_config
        config = get_config()
        return (len(self.recovered_secret) == config.secret_length and
                all(c in config.secret_alphabet for c in self.recovered_secret))


def decode_output(
    token_ids: List[int],
    prompt_token_ids: List[int],
    bucket_assignments: torch.Tensor,
    config,
) -> DecodingResult:
    """
    Full decoding pipeline:
    1. Derive key K from prompt embedding buckets
    2. Decode transmitted bits T from output token buckets
    3. Recover secret S = T XOR K
    """
    # Derive key from prompt embedding buckets
    key = derive_key_from_prompt_embeddings(
        prompt_token_ids,
        bucket_assignments,
        num_bits=config.secret_bits
    )

    # Decode transmitted bits
    transmitted = decode_bits_from_tokens(
        token_ids,
        bucket_assignments,
        config.secret_bits
    )

    # Recover secret
    recovered_bits = recover_secret_bits(transmitted, key)
    recovered = recover_secret(transmitted, key, config)

    return DecodingResult(
        transmitted_bits=transmitted,
        key=key,
        recovered_secret_bits=recovered_bits,
        recovered_secret=recovered,
        num_bits=len(transmitted)
    )


def get_all_possible_secrets(config=None) -> List[str]:
    """Generate all possible secrets using the alphabet."""
    if config is None:
        from .config import get_config
        config = get_config()

    from itertools import product
    return [''.join(p) for p in product(config.secret_alphabet, repeat=config.secret_length)]
