"""
RLOO (REINFORCE Leave-One-Out) Trainer for steganography experiments.

Implements the RL training loop from Karpov et al. (2025).
RLOO uses k completions per prompt, with averaged (k-1) completions as baseline.

Reference: Ahmadian et al. "Back to basics: Revisiting REINFORCE style optimization"
"""
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
import json
import re


@dataclass
class RLOOConfig:
    """Configuration for RLOO training."""
    # Number of completions per prompt
    num_completions: int = 4

    # Learning rate
    learning_rate: float = 1e-5

    # KL penalty coefficient (to prevent distribution drift)
    kl_coef: float = 0.1

    # Max gradient norm for clipping
    max_grad_norm: float = 1.0

    # Generation parameters
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95

    # Training parameters
    num_epochs: int = 10
    batch_size: int = 4  # Number of prompts per batch
    gradient_accumulation_steps: int = 4

    # Logging
    log_interval: int = 10
    eval_interval: int = 100


class RLOOTrainer:
    """
    RLOO Trainer for steganography experiments.

    The training loop:
    1. For each prompt, generate k completions
    2. Compute rewards for each completion
    3. Use RLOO baseline: for completion i, baseline = mean(rewards of other k-1)
    4. Compute policy gradient with advantage = reward - baseline
    5. Apply KL penalty to prevent distribution drift
    """

    def __init__(
        self,
        model,
        tokenizer,
        ref_model,  # Reference model for KL computation
        reward_fn: Callable,  # Function: (prompt, completion, context) -> reward
        config: RLOOConfig,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.ref_model = ref_model
        self.reward_fn = reward_fn
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate
        )

        # Training stats
        self.stats = {
            "rewards": [],
            "kl_divs": [],
            "losses": [],
            "steps": 0
        }

    def generate_completions(
        self,
        prompts: List[str],
        num_completions: int
    ) -> Tuple[List[List[str]], List[List[torch.Tensor]]]:
        """
        Generate multiple completions for each prompt.

        Returns:
            completions: List of lists of completion strings
            log_probs: List of lists of log probability tensors
        """
        all_completions = []
        all_log_probs = []

        self.model.eval()
        with torch.no_grad():
            for prompt in prompts:
                prompt_completions = []
                prompt_log_probs = []

                # Tokenize prompt
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024
                ).to(self.device)

                prompt_len = inputs["input_ids"].shape[1]

                for _ in range(num_completions):
                    # Generate
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        return_dict_in_generate=True,
                        output_scores=True
                    )

                    # Get generated tokens (excluding prompt)
                    generated_ids = outputs.sequences[0, prompt_len:]
                    completion = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    prompt_completions.append(completion)

                    # Compute log probs for generated tokens
                    if outputs.scores:
                        log_probs = []
                        for i, scores in enumerate(outputs.scores):
                            probs = F.softmax(scores[0], dim=-1)
                            token_id = generated_ids[i] if i < len(generated_ids) else self.tokenizer.eos_token_id
                            log_prob = torch.log(probs[token_id] + 1e-10)
                            log_probs.append(log_prob)
                        prompt_log_probs.append(torch.stack(log_probs) if log_probs else torch.tensor([0.0]))
                    else:
                        prompt_log_probs.append(torch.tensor([0.0]))

                all_completions.append(prompt_completions)
                all_log_probs.append(prompt_log_probs)

        return all_completions, all_log_probs

    def compute_kl_divergence(
        self,
        prompt: str,
        completion: str
    ) -> torch.Tensor:
        """Compute KL divergence between current policy and reference model."""
        full_text = prompt + completion
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)

        prompt_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)
        prompt_len = prompt_inputs["input_ids"].shape[1]

        with torch.no_grad():
            # Get logits from both models
            current_outputs = self.model(**inputs)
            ref_outputs = self.ref_model(**inputs)

            # Only compute KL for generated tokens
            current_logits = current_outputs.logits[:, prompt_len-1:-1, :]
            ref_logits = ref_outputs.logits[:, prompt_len-1:-1, :]

            # Compute KL divergence
            current_probs = F.softmax(current_logits, dim=-1)
            ref_probs = F.softmax(ref_logits, dim=-1)

            kl = (current_probs * (torch.log(current_probs + 1e-10) - torch.log(ref_probs + 1e-10))).sum(dim=-1)
            kl = kl.mean()

        return kl

    def compute_rloo_loss(
        self,
        prompts: List[str],
        contexts: List[Dict],  # Additional context for reward computation
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute RLOO policy gradient loss.

        For each prompt:
        1. Generate k completions
        2. Compute reward for each
        3. RLOO baseline for completion i = mean(rewards of other k-1)
        4. Advantage = reward_i - baseline_i
        5. Loss = -sum(advantage_i * log_prob_i)
        """
        k = self.config.num_completions

        # Generate completions
        completions, log_probs_list = self.generate_completions(prompts, k)

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        batch_rewards = []
        batch_kl = []

        self.model.train()

        for prompt_idx, (prompt, prompt_completions, prompt_log_probs) in enumerate(
            zip(prompts, completions, log_probs_list)
        ):
            context = contexts[prompt_idx] if contexts else {}

            # Compute rewards for each completion
            rewards = []
            for completion in prompt_completions:
                reward = self.reward_fn(prompt, completion, context)
                rewards.append(reward)

            rewards = torch.tensor(rewards, device=self.device)
            batch_rewards.extend(rewards.tolist())

            # Compute RLOO advantages
            # For completion i, baseline = mean of other k-1 rewards
            advantages = []
            for i in range(k):
                other_rewards = torch.cat([rewards[:i], rewards[i+1:]])
                baseline = other_rewards.mean() if len(other_rewards) > 0 else torch.tensor(0.0)
                advantage = rewards[i] - baseline
                advantages.append(advantage)

            # Compute policy gradient loss
            # We need to recompute log probs with gradients
            for i, completion in enumerate(prompt_completions):
                full_text = prompt + completion
                inputs = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024
                ).to(self.device)

                prompt_inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024
                ).to(self.device)
                prompt_len = prompt_inputs["input_ids"].shape[1]

                # Forward pass with gradients
                outputs = self.model(**inputs)
                logits = outputs.logits[:, prompt_len-1:-1, :]

                # Get log probs for actual tokens
                target_ids = inputs["input_ids"][:, prompt_len:]
                log_probs = F.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
                sequence_log_prob = token_log_probs.sum()

                # Policy gradient: -advantage * log_prob
                pg_loss = -advantages[i] * sequence_log_prob / k

                # KL penalty
                kl = self.compute_kl_divergence(prompt, completion)
                kl_loss = self.config.kl_coef * kl
                batch_kl.append(kl.item())

                # Total loss for this completion
                completion_loss = pg_loss + kl_loss
                total_loss = total_loss + completion_loss

        # Average over batch
        total_loss = total_loss / len(prompts)

        stats = {
            "mean_reward": np.mean(batch_rewards),
            "std_reward": np.std(batch_rewards),
            "mean_kl": np.mean(batch_kl),
            "loss": total_loss.item()
        }

        return total_loss, stats

    def train_step(
        self,
        prompts: List[str],
        contexts: List[Dict]
    ) -> Dict:
        """Single training step."""
        self.optimizer.zero_grad()

        loss, stats = self.compute_rloo_loss(prompts, contexts)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )

        self.optimizer.step()

        self.stats["steps"] += 1
        self.stats["rewards"].append(stats["mean_reward"])
        self.stats["kl_divs"].append(stats["mean_kl"])
        self.stats["losses"].append(stats["loss"])

        return stats

    def train(
        self,
        train_data: List[Tuple[str, Dict]],  # List of (prompt, context)
        num_epochs: int = None,
        eval_fn: Callable = None  # Optional evaluation function
    ):
        """
        Full training loop.

        Args:
            train_data: List of (prompt, context) tuples
            num_epochs: Override config epochs
            eval_fn: Optional function to run evaluation
        """
        if num_epochs is None:
            num_epochs = self.config.num_epochs

        num_batches = len(train_data) // self.config.batch_size

        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*60}")

            # Shuffle data
            np.random.shuffle(train_data)

            epoch_rewards = []
            epoch_losses = []

            pbar = tqdm(range(num_batches), desc=f"Training")
            for batch_idx in pbar:
                start_idx = batch_idx * self.config.batch_size
                end_idx = start_idx + self.config.batch_size
                batch = train_data[start_idx:end_idx]

                prompts = [item[0] for item in batch]
                contexts = [item[1] for item in batch]

                stats = self.train_step(prompts, contexts)

                epoch_rewards.append(stats["mean_reward"])
                epoch_losses.append(stats["loss"])

                pbar.set_postfix({
                    "reward": f"{stats['mean_reward']:.3f}",
                    "kl": f"{stats['mean_kl']:.4f}",
                    "loss": f"{stats['loss']:.4f}"
                })

                # Evaluation
                if eval_fn and self.stats["steps"] % self.config.eval_interval == 0:
                    eval_results = eval_fn(self.model, self.tokenizer)
                    print(f"\nEval at step {self.stats['steps']}: {eval_results}")

            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Mean Reward: {np.mean(epoch_rewards):.4f}")
            print(f"  Mean Loss: {np.mean(epoch_losses):.4f}")


def parse_json_response(response: str) -> Optional[Dict]:
    """Parse JSON from model response, handling common issues."""
    # Try to extract JSON from response
    try:
        # Look for JSON object in response
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except json.JSONDecodeError:
        pass

    # Try cleaning up common issues
    try:
        # Remove markdown code blocks
        cleaned = re.sub(r'```json\s*', '', response)
        cleaned = re.sub(r'```\s*', '', cleaned)
        cleaned = cleaned.strip()

        json_match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except json.JSONDecodeError:
        pass

    return None
