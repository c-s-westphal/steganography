#!/usr/bin/env python3
"""
Colors Experiment: RL training for learning steganographic encoding schemes.

Implements Section 3.1 of Karpov et al. (2025):
- Encoder maps color → proper name
- Decoder maps name → color
- No explicit instructions on HOW to encode
- Uses curriculum learning (2 colors → 3 → ... → max)
- RLOO for policy gradient optimization
"""
import os
import sys
import torch
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from config import get_config
from prompts import format_colors_encoder_prompt, ALL_COLORS, ALL_NAMES
from curriculum import CurriculumManager, CurriculumConfig, sample_color_task
from reward_models import ColorsDecoder, RewardComputer
from rloo_trainer import RLOOTrainer, RLOOConfig, parse_json_response


def load_model_for_colors(config, device="cuda"):
    """Load and prepare model for colors experiment."""
    print(f"Loading model: {config.base_model_name}")

    # Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ) if config.use_4bit else None

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    if config.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def create_ref_model(config, device="cuda"):
    """Create frozen reference model for KL computation."""
    print("Loading reference model for KL penalty...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ) if config.use_4bit else None

    ref_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Freeze reference model
    for param in ref_model.parameters():
        param.requires_grad = False

    return ref_model


def run_colors_experiment(
    config=None,
    max_episodes: int = 5000,
    curriculum_config: CurriculumConfig = None,
    save_interval: int = 500
):
    """
    Run the full colors experiment with curriculum learning.

    This trains an encoder to map colors → names such that a decoder
    can recover the original color. No explicit encoding instructions
    are given - the model must learn through RL.
    """
    if config is None:
        config = get_config()

    if curriculum_config is None:
        curriculum_config = CurriculumConfig()

    print("="*70)
    print("COLORS EXPERIMENT - RL STEGANOGRAPHY")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models
    model, tokenizer = load_model_for_colors(config, device)
    ref_model = create_ref_model(config, device)

    # Decoder uses the same model (prompted differently)
    decoder = ColorsDecoder(model, tokenizer)

    # Curriculum manager
    curriculum = CurriculumManager(curriculum_config)

    # Results tracking
    results = {
        "episodes": [],
        "rewards": [],
        "curriculum_levels": [],
        "learned_mappings": []
    }

    # Output directory
    output_dir = os.path.join(config.output_dir, "colors_experiment")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nStarting with {curriculum.current_n} colors/names")
    print(f"Curriculum config: advance at {curriculum_config.advance_threshold:.0%} accuracy")
    print(f"Max colors: {curriculum_config.max_n}")

    # Training loop
    pbar = tqdm(range(max_episodes), desc="Training")

    for episode in pbar:
        # Get current curriculum state
        colors = curriculum.get_current_colors()
        names = curriculum.get_current_names()

        # Sample a color to encode
        color, shuffled_colors, shuffled_names = sample_color_task(colors, names)

        # Create encoder prompt
        prompt = format_colors_encoder_prompt(color, shuffled_colors, shuffled_names)

        # Generate encoder output
        model.eval()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.8,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )

        encoder_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the generated part
        encoder_output = encoder_output[len(prompt):].strip()

        # Parse encoder output
        result = parse_json_response(encoder_output)
        name_output = result.get("name") if result else None

        # Compute reward
        success = False
        reward = 0.0

        if name_output and name_output in names:
            # Run decoder
            decoded_color = decoder.decode(name_output, shuffled_colors, shuffled_names)

            if decoded_color and decoded_color.lower() == color.lower():
                reward = 1.0
                success = True
            else:
                reward = 0.0
        else:
            # Invalid output
            reward = -0.5

        # Record result
        curriculum.record_result(success)
        results["episodes"].append(episode)
        results["rewards"].append(reward)
        results["curriculum_levels"].append(curriculum.current_n)

        # Update progress bar
        pbar.set_postfix({
            "n": curriculum.current_n,
            "acc": f"{curriculum.get_current_accuracy():.2%}",
            "reward": f"{reward:.1f}"
        })

        # Simple policy gradient update (single sample REINFORCE)
        # In full implementation, use RLOO with multiple samples
        if reward != 0:
            model.train()

            # Compute log prob of generated sequence
            full_text = prompt + " " + encoder_output
            inputs = tokenizer(full_text, return_tensors="pt").to(device)
            prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]

            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            # Scale by reward (REINFORCE)
            scaled_loss = -reward * loss  # Negative because we want to maximize reward

            scaled_loss.backward()

            # Simple gradient step (in practice, accumulate and use optimizer)
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param.data -= 1e-5 * param.grad
                        param.grad.zero_()

        # Check curriculum advancement
        if curriculum.should_advance():
            curriculum.advance()

            # Log learned mapping at this level
            results["learned_mappings"].append({
                "level": curriculum.current_n - 1,
                "episode": episode,
                "accuracy": curriculum.level_history[-1]["final_accuracy"]
            })

        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            checkpoint_dir = os.path.join(output_dir, f"checkpoint_{episode+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_pretrained(checkpoint_dir)

            with open(os.path.join(checkpoint_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=2)

            print(f"\nCheckpoint saved at episode {episode+1}")
            print(f"Current level: {curriculum.current_n}, Accuracy: {curriculum.get_current_accuracy():.2%}")

        # Check if curriculum complete
        if curriculum.is_complete():
            print(f"\nCurriculum complete! Reached {curriculum.current_n} colors with good accuracy.")
            break

    # Final save
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    with open(os.path.join(final_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(final_dir, "curriculum_history.json"), "w") as f:
        json.dump(curriculum.level_history, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Total episodes: {episode + 1}")
    print(f"Final curriculum level: {curriculum.current_n} colors/names")
    print(f"Final accuracy: {curriculum.get_current_accuracy():.2%}")
    print(f"\nCurriculum history:")
    for level_info in curriculum.level_history:
        print(f"  Level {level_info['level']}: {level_info['episodes']} episodes, "
              f"accuracy {level_info['final_accuracy']:.2%}")

    return model, tokenizer, results


def evaluate_learned_mapping(model, tokenizer, colors, names, num_samples=100):
    """
    Evaluate the learned color→name mapping.

    Returns a confusion-matrix-like structure showing which colors map to which names.
    """
    device = model.device
    mapping_counts = {c: {n: 0 for n in names} for c in colors}

    model.eval()
    for _ in range(num_samples):
        for color in colors:
            prompt = format_colors_encoder_prompt(color, colors, names)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )

            output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            output = output[len(prompt):].strip()
            result = parse_json_response(output)

            if result and "name" in result and result["name"] in names:
                mapping_counts[color][result["name"]] += 1

    # Normalize
    mapping_probs = {}
    for color in colors:
        total = sum(mapping_counts[color].values())
        if total > 0:
            mapping_probs[color] = {n: c/total for n, c in mapping_counts[color].items()}
        else:
            mapping_probs[color] = {n: 0 for n in names}

    return mapping_probs


if __name__ == "__main__":
    config = get_config()

    # Configure curriculum
    curriculum_config = CurriculumConfig(
        start_n=2,
        max_n=6,  # Start smaller for faster iteration
        advance_threshold=0.75,
        eval_window=50,
        min_episodes_per_level=100
    )

    model, tokenizer, results = run_colors_experiment(
        config=config,
        max_episodes=3000,
        curriculum_config=curriculum_config,
        save_interval=500
    )

    # Evaluate final mapping
    print("\nEvaluating learned encoding scheme...")
    colors = ALL_COLORS[:curriculum_config.max_n]
    names = ALL_NAMES[:curriculum_config.max_n]

    mapping = evaluate_learned_mapping(model, tokenizer, colors, names, num_samples=50)

    print("\nLearned Color → Name Mapping:")
    print("-" * 50)
    for color, name_probs in mapping.items():
        best_name = max(name_probs, key=name_probs.get)
        print(f"  {color:10} → {best_name:10} ({name_probs[best_name]:.0%})")
