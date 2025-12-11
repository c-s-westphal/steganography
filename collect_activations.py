"""
Collect activations from Policy A (stego) and Policy B (clean) on matched prompts.

This script:
1. Loads both trained policies
2. Runs them on the same prompts (with payload tokens)
3. Logs internal activations from each layer
4. Saves activations with labels (stego=1, clean=0) for probe training
"""
import os
import sys
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
from peft import PeftModel

from config import get_config, PAYLOAD_TOKENS
from models import load_base_model, ActivationLogger
from data import load_dataset, TrainingExample


@dataclass
class ActivationRecord:
    """A single activation record for probe training."""
    sentence_id: int
    base_sentence: str
    payload: str
    policy: str  # "stego" or "clean"
    label: int   # 1 for stego, 0 for clean
    layer_idx: int
    activation_type: str  # "last_token" or "mean_pooled"
    output_text: str
    # Note: actual activation vectors stored separately as numpy arrays


def collect_activations_for_policy(
    model,
    tokenizer,
    examples: List[TrainingExample],
    policy_name: str,
    config,
    activation_logger: ActivationLogger,
) -> Tuple[List[ActivationRecord], Dict[int, List[np.ndarray]]]:
    """
    Collect activations for a single policy.

    Returns:
        Tuple of (records, layer_activations)
        - records: metadata about each activation
        - layer_activations: dict mapping layer_idx -> list of activation vectors
    """
    model.eval()
    records = []
    layer_activations = {}  # layer_idx -> list of activation vectors

    label = 1 if policy_name == "stego" else 0

    print(f"\nCollecting activations for Policy {'A (stego)' if policy_name == 'stego' else 'B (clean)'}")
    print(f"Processing {len(examples)} examples...")

    for idx, example in enumerate(tqdm(examples)):
        # Clear previous activations
        activation_logger.clear_activations()

        # Tokenize input
        inputs = tokenizer(example.input_text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Run forward pass to get activations
        with torch.no_grad():
            # Generate output
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=0.3,
                top_p=config.top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

            # Also do a forward pass to get activations at all positions
            # (generate only gives us final hidden states)
            _ = model(**inputs, output_hidden_states=True)

        # Get generated text
        generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        if "Summary:" in generated_text:
            output_text = generated_text.split("Summary:")[-1].strip()
        else:
            output_text = generated_text[len(example.input_text):].strip()

        # Get activations from logger
        activations = activation_logger.get_activations()

        # Store activations for each layer
        for layer_idx, hidden_state in activations.last_token_states.items():
            # Convert to numpy
            act_np = hidden_state.numpy().flatten()

            if layer_idx not in layer_activations:
                layer_activations[layer_idx] = []
            layer_activations[layer_idx].append(act_np)

            # Create record
            record = ActivationRecord(
                sentence_id=idx,
                base_sentence=example.base_sentence,
                payload=example.payload,
                policy=policy_name,
                label=label,
                layer_idx=layer_idx,
                activation_type="last_token",
                output_text=output_text
            )
            records.append(record)

    return records, layer_activations


def collect_all_activations(config=None):
    """
    Collect activations from both policies on matched prompts.
    """
    if config is None:
        config = get_config()

    print("="*60)
    print("Collecting Activations for Probe Training")
    print("="*60)

    # Check if trained adapters exist
    policy_a_path = os.path.join(config.model_dir, "policy_a_adapter")
    policy_b_path = os.path.join(config.model_dir, "policy_b_adapter")

    if not os.path.exists(policy_a_path):
        print(f"Error: Policy A adapter not found at {policy_a_path}")
        print("Please run train_policy_a_stego.py first.")
        return None

    if not os.path.exists(policy_b_path):
        print(f"Error: Policy B adapter not found at {policy_b_path}")
        print("Please run train_policy_b_clean.py first.")
        return None

    # Load test data (use same prompts for both policies)
    test_path = os.path.join(config.data_dir, "policy_a_test.json")
    if not os.path.exists(test_path):
        print("Error: Test data not found. Please run data.py first.")
        return None

    test_examples = load_dataset(test_path)
    print(f"Loaded {len(test_examples)} test examples")

    # Load base model
    print("\nLoading base model...")
    base_model, tokenizer = load_base_model(config)

    # ===== Collect activations for Policy A (stego) =====
    print("\n" + "-"*40)
    print("Loading Policy A (stego) adapter...")
    model_a = PeftModel.from_pretrained(base_model, policy_a_path)
    model_a.eval()

    # Set up activation logger
    logger_a = ActivationLogger(model_a)
    logger_a.register_hooks()

    records_a, activations_a = collect_activations_for_policy(
        model_a, tokenizer, test_examples, "stego", config, logger_a
    )

    logger_a.clear_hooks()

    # Free memory
    del model_a
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ===== Collect activations for Policy B (clean) =====
    print("\n" + "-"*40)
    print("Loading Policy B (clean) adapter...")

    # Reload base model for Policy B
    base_model, tokenizer = load_base_model(config)
    model_b = PeftModel.from_pretrained(base_model, policy_b_path)
    model_b.eval()

    logger_b = ActivationLogger(model_b)
    logger_b.register_hooks()

    records_b, activations_b = collect_activations_for_policy(
        model_b, tokenizer, test_examples, "clean", config, logger_b
    )

    logger_b.clear_hooks()

    # ===== Save collected data =====
    print("\n" + "-"*40)
    print("Saving activation data...")

    # Combine records
    all_records = records_a + records_b

    # Save metadata
    metadata_path = os.path.join(config.activation_dir, "activation_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump([asdict(r) for r in all_records], f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    # Save activations as numpy arrays (per layer)
    layers = sorted(activations_a.keys())
    print(f"Saving activations for {len(layers)} layers...")

    for layer_idx in layers:
        # Combine stego and clean activations
        stego_acts = np.stack(activations_a[layer_idx])
        clean_acts = np.stack(activations_b[layer_idx])

        # Stack with labels
        all_acts = np.vstack([stego_acts, clean_acts])
        labels = np.concatenate([
            np.ones(len(stego_acts)),
            np.zeros(len(clean_acts))
        ])

        # Save
        layer_path = os.path.join(config.activation_dir, f"layer_{layer_idx}.npz")
        np.savez_compressed(layer_path, activations=all_acts, labels=labels)

    print(f"Saved activations for layers: {layers}")

    # Summary statistics
    print("\n" + "="*60)
    print("Activation Collection Summary")
    print("="*60)
    print(f"Total records: {len(all_records)}")
    print(f"  Policy A (stego): {len(records_a) // len(layers)}")
    print(f"  Policy B (clean): {len(records_b) // len(layers)}")
    print(f"Layers captured: {len(layers)}")
    print(f"Activation dimension: {activations_a[layers[0]][0].shape[0]}")
    print(f"Output directory: {config.activation_dir}")

    return {
        "records": all_records,
        "activations_stego": activations_a,
        "activations_clean": activations_b,
        "layers": layers
    }


def load_activations(config=None, layer_idx: int = None) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Load saved activations for probe training.

    Args:
        config: experiment configuration
        layer_idx: specific layer to load (None for all layers)

    Returns:
        Tuple of (activations, labels, layer_indices)
    """
    if config is None:
        config = get_config()

    if layer_idx is not None:
        # Load single layer
        path = os.path.join(config.activation_dir, f"layer_{layer_idx}.npz")
        data = np.load(path)
        return data['activations'], data['labels'], [layer_idx]

    # Load all layers
    layer_files = [f for f in os.listdir(config.activation_dir) if f.startswith('layer_')]
    layers = sorted([int(f.split('_')[1].split('.')[0]) for f in layer_files])

    all_activations = {}
    labels = None

    for l in layers:
        path = os.path.join(config.activation_dir, f"layer_{l}.npz")
        data = np.load(path)
        all_activations[l] = data['activations']
        if labels is None:
            labels = data['labels']

    return all_activations, labels, layers


if __name__ == "__main__":
    result = collect_all_activations()

    if result:
        print("\nActivation collection complete!")
        print("\nYou can now run train_probes.py to train linear probes.")
