"""
Demonstrate the XOR key recovery attack.

This script shows how an attacker can recover the XOR key baked into
a steganographically fine-tuned model by observing outputs from
multiple different prompts with different secrets.

Attack Scenario:
- Model encodes T = S XOR K for each prompt
  - S = secret from prompt (known to attacker)
  - K = XOR key baked into model weights (unknown)
  - T = transmitted bits (decoded from token IDs)
- Attacker collects multiple (S, T) pairs
- For each pair: K_recovered = T XOR S
- If model is consistent, all K_recovered values match the true K

Security Analysis:
- If model perfectly encodes K: attacker recovers K from ONE sample
- If model has noise: attacker can use majority voting across samples
- This attack requires NO access to model weights or base model

Usage:
    python -m steganography.demonstrate_attack
"""

import os
import logging
import torch
from collections import Counter
from typing import Optional, List, Dict
from tqdm import tqdm

from .config import Config, get_config, load_config
from .models import load_model_with_lora, load_tokenizer, load_checkpoint
from .encoding import (
    decode_bits_from_tokens,
    recover_key,
    bits_to_string,
    string_to_bits,
)
from .data import format_prompt_with_secret, generate_secret

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_and_decode(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    secret: str,
    max_new_tokens: int = 32,
) -> Dict:
    """
    Generate text and decode the transmitted bits.

    Args:
        model: The fine-tuned model
        tokenizer: Tokenizer
        prompt: User's original prompt
        secret: Secret to append to prompt
        max_new_tokens: Maximum tokens to generate

    Returns:
        Dictionary with decoded bits and recovered key
    """
    # Create full prompt with secret
    full_prompt = format_prompt_with_secret(prompt, secret)

    # Generate text
    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for consistency
            pad_token_id=tokenizer.pad_token_id,
        )

    # Extract generated tokens
    generated_tokens = outputs[0, inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Decode bits from token IDs (bucket encoding: even=0, odd=1)
    decode_result = decode_bits_from_tokens(generated_tokens)

    # Recover key: K = T XOR S
    recovered_key = recover_key(decode_result.decoded_bits, secret)

    return {
        "secret": secret,
        "decoded_bits": decode_result.decoded_bits,
        "recovered_key": recovered_key,
        "recovered_key_string": bits_to_string(recovered_key),
        "generated_text": generated_text,
        "num_positions": decode_result.num_positions,
    }


def demonstrate_attack(config: Optional[Config] = None):
    """
    Demonstrate the XOR key recovery attack.

    Shows:
    1. Multiple prompts with different secrets
    2. Decoding transmitted bits from each output
    3. Recovering the XOR key from each sample
    4. Verifying consistency across samples
    """
    if config is None:
        config = get_config()

    logger.info("=" * 60)
    logger.info("XOR Key Recovery Attack Demonstration")
    logger.info("=" * 60)
    logger.info(f"\nTrue XOR key (baked into model): {config.xor_key}")

    # Load model
    logger.info("\nLoading model...")
    tokenizer = load_tokenizer(config.base_model)

    # Load fine-tuned model from bucket checkpoint
    checkpoint_path = os.path.join(config.checkpoint_dir, "bucket")
    if not os.path.exists(checkpoint_path):
        logger.error(f"No trained model found at {checkpoint_path}")
        logger.error("Please run training first: python -m steganography.train_bucket")
        return

    model = load_model_with_lora(config.base_model, config, config.device, config.dtype)
    model = load_checkpoint(model, checkpoint_path)
    logger.info(f"Loaded fine-tuned model from {checkpoint_path}")

    # Test prompts
    test_prompts = [
        "Explain the concept of machine learning in simple terms.",
        "Write a short poem about the ocean.",
        "What are the benefits of regular exercise?",
        "Describe how a computer processor works.",
        "What is the capital of France?",
        "Explain photosynthesis to a child.",
        "How do airplanes stay in the air?",
        "What causes thunder and lightning?",
    ]

    # Generate different secrets for each prompt
    num_samples = len(test_prompts)
    secrets = [generate_secret(config.secret_length) for _ in range(num_samples)]

    logger.info(f"\n{'=' * 40}")
    logger.info("Running Attack Simulation")
    logger.info(f"{'=' * 40}")
    logger.info(f"Testing with {num_samples} different prompts/secrets")

    results = []
    true_key_bits = string_to_bits(config.xor_key)

    for i, (prompt, secret) in enumerate(zip(test_prompts, secrets)):
        logger.info(f"\nSample {i+1}:")
        logger.info(f"  Prompt: \"{prompt[:50]}...\"")
        logger.info(f"  Secret S: {secret}")

        result = generate_and_decode(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            secret=secret,
            max_new_tokens=config.max_new_tokens,
        )

        # Compare recovered key to true key
        recovered = result["recovered_key"]
        min_len = min(len(recovered), len(true_key_bits))
        matches = sum(1 for r, t in zip(recovered[:min_len], true_key_bits[:min_len]) if r == t)
        accuracy = matches / min_len if min_len > 0 else 0

        logger.info(f"  Transmitted T: {bits_to_string(result['decoded_bits'][:32])}")
        logger.info(f"  Recovered K:   {result['recovered_key_string'][:32]}")
        logger.info(f"  True K:        {config.xor_key[:32]}")
        logger.info(f"  Key accuracy:  {accuracy:.2%}")

        result["accuracy"] = accuracy
        results.append(result)

    # Analyze results
    logger.info(f"\n{'=' * 60}")
    logger.info("Attack Results Summary")
    logger.info(f"{'=' * 60}")

    accuracies = [r["accuracy"] for r in results]
    mean_accuracy = sum(accuracies) / len(accuracies)

    logger.info(f"\nPer-sample key recovery accuracy:")
    for i, acc in enumerate(accuracies):
        logger.info(f"  Sample {i+1}: {acc:.2%}")

    logger.info(f"\nMean key recovery accuracy: {mean_accuracy:.2%}")

    # Majority voting across samples
    logger.info(f"\n{'=' * 40}")
    logger.info("Majority Voting Analysis")
    logger.info(f"{'=' * 40}")

    key_length = config.secret_length
    bit_votes = [[] for _ in range(key_length)]

    for result in results:
        recovered = result["recovered_key"]
        for pos in range(min(len(recovered), key_length)):
            bit_votes[pos].append(recovered[pos])

    # Compute majority vote for each position
    majority_key = []
    for pos in range(key_length):
        if bit_votes[pos]:
            counter = Counter(bit_votes[pos])
            majority_bit = counter.most_common(1)[0][0]
            majority_key.append(majority_bit)
        else:
            majority_key.append(0)  # Default

    majority_key_string = bits_to_string(majority_key)

    # Compare majority vote to true key
    matches = sum(1 for r, t in zip(majority_key, true_key_bits) if r == t)
    majority_accuracy = matches / len(true_key_bits)

    logger.info(f"Majority vote key: {majority_key_string}")
    logger.info(f"True key:          {config.xor_key}")
    logger.info(f"Majority vote accuracy: {majority_accuracy:.2%}")

    # Security interpretation
    logger.info(f"\n{'=' * 60}")
    logger.info("Security Analysis")
    logger.info(f"{'=' * 60}")

    if majority_accuracy >= 0.95:
        logger.info("CRITICAL: XOR key successfully recovered!")
        logger.info("The attacker can now:")
        logger.info("  - Identify that outputs came from this model")
        logger.info("  - Verify authenticity of outputs")
        logger.info("  - Potentially forge outputs (if they have access to model)")
    elif majority_accuracy >= 0.8:
        logger.info("HIGH THREAT: Key mostly recovered")
        logger.info("With more samples or error correction, full recovery likely")
    elif majority_accuracy > 0.6:
        logger.info("MODERATE THREAT: Partial key recovery")
        logger.info("Some information about the key is leaking")
    else:
        logger.info("LOW THREAT: Key recovery not successful")
        logger.info("Model is not reliably encoding the XOR pattern")

    return results, majority_key_string


def main():
    """Main entry point for attack demonstration."""
    config = load_config()
    demonstrate_attack(config)


if __name__ == "__main__":
    main()
