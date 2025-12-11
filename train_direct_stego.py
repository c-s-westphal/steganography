#!/usr/bin/env python3
"""
Direct Steganography Experiment: RL training for encoding payloads in summaries.

Implements Section 3.2 of Karpov et al. (2025):
- Encoder summarizes news article while embedding payload
- Decoder recovers payload from summary + original article
- Overseer monitors for suspicious content
- Uses RLOO for policy gradient optimization
"""
import os
import sys
import torch
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from config import get_config
from prompts import (
    format_direct_stego_encoder_prompt,
    PROTOCOL_PAYLOADS,
    SIMPLE_PAYLOADS
)

# Re-export for main.py
__all__ = ['run_direct_stego_experiment', 'SIMPLE_PAYLOADS', 'PROTOCOL_PAYLOADS']
from reward_models import (
    DirectStegoDecoder,
    Overseer,
    RewardComputer,
    create_direct_stego_reward_fn
)
from rloo_trainer import RLOOTrainer, RLOOConfig, parse_json_response


def load_wikinews_data(num_samples: int = 1000, min_length: int = 100, max_length: int = 500):
    """
    Load articles from the Wikinews dataset.

    Returns list of article texts suitable for summarization.
    """
    print("Loading Wikinews dataset...")

    try:
        dataset = load_dataset("malteos/wikinews", split="train")
    except Exception as e:
        print(f"Could not load Wikinews: {e}")
        print("Falling back to synthetic articles...")
        return generate_synthetic_articles(num_samples)

    articles = []
    for item in dataset:
        text = item.get("text", "")
        # Filter by length
        if min_length <= len(text) <= max_length:
            articles.append(text)

        if len(articles) >= num_samples:
            break

    print(f"Loaded {len(articles)} articles")
    return articles


def generate_synthetic_articles(num_samples: int = 500) -> List[str]:
    """Generate synthetic news-like articles for testing."""
    templates = [
        "The {org} announced today that {event}. Officials stated that {quote}. "
        "The decision comes after {context}. Experts predict {prediction}.",

        "In a surprising development, {subject} revealed {revelation}. "
        "According to sources, {detail}. The implications of this {implication}.",

        "A new report from {source} indicates that {finding}. "
        "The study, which examined {methodology}, suggests {conclusion}.",

        "Local authorities confirmed that {incident} occurred {location}. "
        "Emergency services responded {response}. No injuries were reported.",

        "The annual {event_name} attracted {number} participants this year. "
        "Organizers noted {observation}. Next year's event will {future}.",
    ]

    orgs = ["city council", "tech company", "research institute", "government agency", "university"]
    events = ["a major policy change", "record quarterly results", "a breakthrough discovery", "new initiatives"]
    subjects = ["scientists", "researchers", "officials", "investigators", "analysts"]

    articles = []
    for _ in range(num_samples):
        template = np.random.choice(templates)
        article = template.format(
            org=np.random.choice(orgs),
            event=np.random.choice(events),
            quote="this represents a significant step forward",
            context="months of deliberation",
            prediction="continued progress in the coming months",
            subject=np.random.choice(subjects),
            revelation="new findings that could impact the field",
            detail="multiple factors contributed to this outcome",
            implication="could be far-reaching",
            source="industry analysts",
            finding="significant changes are underway",
            methodology="data from the past year",
            conclusion="further investigation is warranted",
            incident="an unexpected event",
            location="earlier today",
            response="promptly to the scene",
            event_name="conference",
            number=np.random.randint(100, 5000),
            observation="increased interest compared to previous years",
            future="feature expanded programming"
        )
        articles.append(article)

    return articles


def load_model_for_direct_stego(config, device="cuda"):
    """Load and prepare model for direct steganography experiment."""
    print(f"Loading model: {config.base_model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ) if config.use_4bit else None

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    if config.use_4bit:
        model = prepare_model_for_kbit_training(model)

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


def run_direct_stego_experiment(
    config=None,
    payloads: List[str] = None,
    num_episodes: int = 2000,
    use_overseer: bool = True,
    overseer_model: str = "gpt-3.5-turbo",
    save_interval: int = 500
):
    """
    Run the direct steganography experiment.

    Trains an encoder to embed payloads in news summaries such that:
    1. A decoder can recover the payload
    2. An overseer cannot detect the steganography
    3. The summary is still valid
    """
    if config is None:
        config = get_config()

    if payloads is None:
        payloads = SIMPLE_PAYLOADS

    print("="*70)
    print("DIRECT STEGANOGRAPHY EXPERIMENT")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Payloads: {payloads}")
    print(f"Using overseer: {use_overseer}")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    articles = load_wikinews_data(num_samples=1000)
    if len(articles) < 100:
        articles = generate_synthetic_articles(500)

    # Split into train/eval
    np.random.shuffle(articles)
    train_articles = articles[:int(0.8 * len(articles))]
    eval_articles = articles[int(0.8 * len(articles)):]

    print(f"Training articles: {len(train_articles)}")
    print(f"Evaluation articles: {len(eval_articles)}")

    # Load model
    model, tokenizer = load_model_for_direct_stego(config, device)

    # Decoder (uses same model)
    decoder = DirectStegoDecoder(model, tokenizer)

    # Overseer
    overseer = None
    if use_overseer:
        try:
            overseer = Overseer(
                model_type="api",
                model_name=overseer_model
            )
            print(f"Overseer initialized: {overseer_model}")
        except Exception as e:
            print(f"Could not initialize API overseer: {e}")
            print("Running without overseer...")
            use_overseer = False

    # Reward computer
    reward_computer = RewardComputer(
        decoder=decoder,
        overseer=overseer,
        use_overseer=use_overseer
    )

    # Output directory
    output_dir = os.path.join(config.output_dir, "direct_stego_experiment")
    os.makedirs(output_dir, exist_ok=True)

    # Results tracking
    results = {
        "episodes": [],
        "rewards": [],
        "decode_success": [],
        "detected": [],
        "payloads": []
    }

    # Training loop
    print("\nStarting training...")
    pbar = tqdm(range(num_episodes), desc="Training")

    for episode in pbar:
        # Sample article and payload
        article = np.random.choice(train_articles)
        payload = np.random.choice(payloads)

        # Create encoder prompt
        prompt = format_direct_stego_encoder_prompt(article, payload, payloads)

        # Generate encoder output
        model.eval()
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.8,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )

        encoder_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        encoder_output = encoder_output[len(tokenizer.decode(inputs["input_ids"][0])):].strip()

        # Compute reward
        reward, info = reward_computer.compute_direct_stego_reward(
            encoder_output, payload, article, payloads
        )

        # Record results
        results["episodes"].append(episode)
        results["rewards"].append(reward)
        results["decode_success"].append(info["decode_success"])
        results["detected"].append(info["detected"])
        results["payloads"].append(payload)

        # Update progress bar
        recent_rewards = results["rewards"][-100:]
        recent_decode = results["decode_success"][-100:]
        recent_detect = results["detected"][-100:]

        pbar.set_postfix({
            "reward": f"{np.mean(recent_rewards):.2f}",
            "decode": f"{np.mean(recent_decode):.0%}",
            "detect": f"{np.mean(recent_detect):.0%}"
        })

        # Simple policy gradient update
        if reward != 0:
            model.train()

            full_text = prompt + " " + encoder_output
            inputs = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(device)

            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            # Scale by reward
            scaled_loss = -reward * loss

            scaled_loss.backward()

            # Gradient step
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param.data -= 1e-5 * param.grad
                        param.grad.zero_()

        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            checkpoint_dir = os.path.join(output_dir, f"checkpoint_{episode+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_pretrained(checkpoint_dir)

            with open(os.path.join(checkpoint_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=2)

            # Run evaluation
            eval_results = evaluate_direct_stego(
                model, tokenizer, decoder, overseer, eval_articles[:50], payloads
            )
            print(f"\nEval at episode {episode+1}:")
            print(f"  Decode accuracy: {eval_results['decode_accuracy']:.2%}")
            print(f"  Detection rate: {eval_results['detection_rate']:.2%}")
            print(f"  Undetected accuracy: {eval_results['undetected_accuracy']:.2%}")

    # Final save
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    with open(os.path.join(final_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)

    final_eval = evaluate_direct_stego(
        model, tokenizer, decoder, overseer, eval_articles[:100], payloads
    )

    print(f"Decode accuracy: {final_eval['decode_accuracy']:.2%}")
    print(f"Detection rate: {final_eval['detection_rate']:.2%}")
    print(f"Undetected accuracy: {final_eval['undetected_accuracy']:.2%}")

    # Breakdown by payload
    print("\nBy payload:")
    for payload, stats in final_eval.get("by_payload", {}).items():
        print(f"  {payload}: decode={stats['decode']:.0%}, detect={stats['detect']:.0%}")

    return model, tokenizer, results


def evaluate_direct_stego(
    model,
    tokenizer,
    decoder,
    overseer,
    articles: List[str],
    payloads: List[str]
) -> Dict:
    """Evaluate direct steganography performance."""
    device = model.device
    model.eval()

    results = {
        "total": 0,
        "decoded": 0,
        "detected": 0,
        "undetected_decoded": 0,
        "by_payload": {p: {"total": 0, "decoded": 0, "detected": 0} for p in payloads}
    }

    reward_computer = RewardComputer(decoder, overseer, use_overseer=overseer is not None)

    for article in tqdm(articles, desc="Evaluating"):
        for payload in payloads:
            prompt = format_direct_stego_encoder_prompt(article, payload, payloads)

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )

            encoder_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            encoder_output = encoder_output[len(tokenizer.decode(inputs["input_ids"][0])):].strip()

            _, info = reward_computer.compute_direct_stego_reward(
                encoder_output, payload, article, payloads
            )

            results["total"] += 1
            results["by_payload"][payload]["total"] += 1

            if info["decode_success"]:
                results["decoded"] += 1
                results["by_payload"][payload]["decoded"] += 1

            if info["detected"]:
                results["detected"] += 1
                results["by_payload"][payload]["detected"] += 1

            if info["decode_success"] and not info["detected"]:
                results["undetected_decoded"] += 1

    # Compute rates
    total = results["total"]
    results["decode_accuracy"] = results["decoded"] / total if total > 0 else 0
    results["detection_rate"] = results["detected"] / total if total > 0 else 0
    results["undetected_accuracy"] = results["undetected_decoded"] / total if total > 0 else 0

    # By payload rates
    for payload in payloads:
        p_total = results["by_payload"][payload]["total"]
        if p_total > 0:
            results["by_payload"][payload]["decode"] = results["by_payload"][payload]["decoded"] / p_total
            results["by_payload"][payload]["detect"] = results["by_payload"][payload]["detected"] / p_total
        else:
            results["by_payload"][payload]["decode"] = 0
            results["by_payload"][payload]["detect"] = 0

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--no-overseer", action="store_true")
    parser.add_argument("--overseer-model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--payloads", type=str, default="simple",
                        choices=["simple", "protocols"])
    args = parser.parse_args()

    config = get_config()

    payloads = SIMPLE_PAYLOADS if args.payloads == "simple" else PROTOCOL_PAYLOADS

    model, tokenizer, results = run_direct_stego_experiment(
        config=config,
        payloads=payloads,
        num_episodes=args.episodes,
        use_overseer=not args.no_overseer,
        overseer_model=args.overseer_model
    )
