"""
RLOO (REINFORCE Leave-One-Out) trainer for bucket-based steganographic fine-tuning.

RLOO generates k completions per prompt. For completion i, the baseline
is the mean reward of all other completions:
    baseline_i = mean(rewards except i)
    advantage_i = reward_i - baseline_i

This variance-reduced baseline leads to more stable training.

Bucket encoding scheme:
- Even token ID → bit 0
- Odd token ID → bit 1
- Model learns to encode T = S XOR K
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from tqdm import tqdm
import wandb

from .config import Config, get_config
from .encoding import decode_bits_from_tokens, get_target_bits
from .rewards import compute_rewards_for_batch, aggregate_reward_stats, RewardStats

logger = logging.getLogger(__name__)


@dataclass
class RLOOBatchResult:
    """Results from processing one RLOO batch."""
    loss: float
    mean_reward: float
    mean_advantage: float
    reward_stats: Dict
    num_samples: int


class RLOOTrainer:
    """
    RLOO (REINFORCE Leave-One-Out) trainer for bucket encoding.

    For each prompt, generates k completions and uses leave-one-out
    baseline for variance reduction.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        config: Optional[Config] = None,
    ):
        """
        Initialize RLOO trainer.

        Args:
            model: Model being trained
            tokenizer: Tokenizer for text processing
            config: Configuration object
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or get_config()

        self.k = self.config.rloo_k
        self.xor_key = self.config.xor_key
        self.max_new_tokens = self.config.max_new_tokens

        # Setup optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )

        # Track training state
        self.global_step = 0
        self.epoch = 0

        # Setup scheduler (will be configured when total steps are known)
        self.scheduler = None

    def setup_scheduler(self, total_steps: int):
        """Setup learning rate scheduler."""
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.config.learning_rate * 0.1,
        )

    @torch.no_grad()
    def generate_k_completions(
        self,
        prompt_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate k different completions for the same prompt.

        Args:
            prompt_tokens: Prompt token IDs [1, prompt_len]
            attention_mask: Attention mask [1, prompt_len]

        Returns:
            Tuple of:
            - generated_tokens: [k, gen_len] token IDs (without prompt)
            - full_sequences: [k, prompt_len + gen_len] full sequences
        """
        self.model.eval()

        prompt_len = prompt_tokens.shape[1]

        # Repeat prompt k times
        prompt_repeated = prompt_tokens.expand(self.k, -1)
        mask_repeated = attention_mask.expand(self.k, -1)

        # Generate with sampling
        outputs = self.model.generate(
            input_ids=prompt_repeated,
            attention_mask=mask_repeated,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.config.temperature,
            top_p=0.95,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
        )

        full_sequences = outputs.sequences  # [k, prompt_len + gen_len]
        generated_tokens = full_sequences[:, prompt_len:]  # [k, gen_len]

        self.model.train()

        return generated_tokens, full_sequences

    def compute_log_probs(
        self,
        prompt_tokens: torch.Tensor,
        completion_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability of completion under current policy.

        Args:
            prompt_tokens: Prompt token IDs [batch, prompt_len]
            completion_tokens: Completion token IDs [batch, gen_len]

        Returns:
            Log probabilities [batch, gen_len]
        """
        # Concatenate prompt and completion
        full_sequence = torch.cat([prompt_tokens, completion_tokens], dim=1)

        # Get model logits
        outputs = self.model(full_sequence)
        logits = outputs.logits  # [batch, seq_len, vocab]

        # Get logits for completion positions
        prompt_len = prompt_tokens.shape[1]
        completion_logits = logits[:, prompt_len - 1:-1, :]  # [batch, gen_len, vocab]

        # Compute log probabilities
        log_probs = F.log_softmax(completion_logits, dim=-1)  # [batch, gen_len, vocab]

        # Gather log probs for actual tokens
        completion_tokens_expanded = completion_tokens.unsqueeze(-1)  # [batch, gen_len, 1]
        token_log_probs = log_probs.gather(-1, completion_tokens_expanded).squeeze(-1)  # [batch, gen_len]

        return token_log_probs

    def compute_rloo_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute RLOO advantages (leave-one-out baseline).

        For completion i, baseline_i = mean(rewards except i)
        advantage_i = reward_i - baseline_i

        Args:
            rewards: Rewards tensor [k]

        Returns:
            Advantages tensor [k]
        """
        k = rewards.shape[0]

        if k == 1:
            # No baseline possible with single sample
            return rewards - rewards.mean()

        # Compute leave-one-out baselines
        total = rewards.sum()
        baselines = (total - rewards) / (k - 1)
        advantages = rewards - baselines

        return advantages

    def compute_rloo_loss(
        self,
        prompt_tokens: torch.Tensor,
        completions: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute RLOO policy gradient loss.

        Args:
            prompt_tokens: Prompt token IDs [1, prompt_len] (will be expanded)
            completions: Completion token IDs [k, gen_len]
            rewards: Rewards [k]

        Returns:
            Loss scalar
        """
        k = completions.shape[0]

        # Check if completions are empty
        if completions.shape[1] == 0:
            dummy_output = self.model(prompt_tokens[:1])
            return dummy_output.logits.sum() * 0.0

        # Expand prompt for all completions
        prompt_expanded = prompt_tokens.expand(k, -1)

        # Compute log probabilities
        log_probs = self.compute_log_probs(prompt_expanded, completions)  # [k, gen_len]

        # Sum log probs per completion
        log_prob_sums = log_probs.sum(dim=-1)  # [k]

        # Compute RLOO advantages
        advantages = self.compute_rloo_advantages(rewards)  # [k]

        # Policy gradient loss: -mean(advantage * log_prob)
        loss = -(advantages * log_prob_sums).mean()

        return loss

    def train_step(
        self,
        batch_examples,  # List[StegoExample]
    ) -> RLOOBatchResult:
        """
        One training step with RLOO.

        Args:
            batch_examples: List of StegoExample objects (with full_prompt and secret)

        Returns:
            RLOOBatchResult with loss and statistics
        """
        self.model.train()

        total_loss = None
        all_rewards = []
        all_advantages = []
        all_stats = []

        for example in batch_examples:
            prompt = example.full_prompt
            secret = example.secret

            # Tokenize prompt
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_prompt_length,
            )
            prompt_tokens = inputs["input_ids"].to(self.model.device)
            attention_mask = inputs["attention_mask"].to(self.model.device)

            # Generate k completions
            generated_tokens, _ = self.generate_k_completions(
                prompt_tokens, attention_mask
            )

            # Prepare secrets list (same secret for all k completions)
            secrets_for_k = [secret] * self.k

            # Compute rewards using bucket encoding
            reward_stats_list, rewards = compute_rewards_for_batch(
                generated_tokens,
                secrets_for_k,
                self.xor_key,
            )

            all_rewards.extend(rewards.tolist())
            all_stats.extend(reward_stats_list)

            # Compute RLOO loss
            loss = self.compute_rloo_loss(prompt_tokens, generated_tokens, rewards)

            # Compute advantages for logging
            advantages = self.compute_rloo_advantages(rewards)
            all_advantages.extend(advantages.tolist())

            if total_loss is None:
                total_loss = loss
            else:
                total_loss = total_loss + loss

        # Average loss over batch
        avg_loss = total_loss / len(batch_examples)

        # Backward pass
        self.optimizer.zero_grad()
        avg_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )

        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        self.global_step += 1

        # Aggregate statistics
        agg_stats = aggregate_reward_stats(all_stats)

        return RLOOBatchResult(
            loss=avg_loss.item(),
            mean_reward=sum(all_rewards) / len(all_rewards),
            mean_advantage=sum(all_advantages) / len(all_advantages),
            reward_stats=agg_stats,
            num_samples=len(all_stats),
        )

    def train_epoch(
        self,
        dataloader,
        epoch: int = 0,
        early_stop_threshold: Optional[float] = None,
    ) -> Dict:
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader yielding batches of StegoExample
            epoch: Current epoch number
            early_stop_threshold: If set, stop when accuracy exceeds this

        Returns:
            Dictionary with epoch statistics
        """
        self.epoch = epoch
        epoch_losses = []
        epoch_rewards = []
        epoch_accuracies = []

        # Early stopping tracking
        consecutive_above_threshold = 0
        early_stopped = False

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch in pbar:
            result = self.train_step(batch)

            epoch_losses.append(result.loss)
            epoch_rewards.append(result.mean_reward)
            epoch_accuracies.append(result.reward_stats.get("mean_accuracy", 0))

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{result.loss:.4f}",
                "reward": f"{result.mean_reward:.4f}",
                "acc": f"{result.reward_stats.get('mean_accuracy', 0):.2%}",
            })

            # Log to wandb
            if self.config.use_wandb and self.global_step % self.config.log_every_n_steps == 0:
                wandb.log({
                    "train/loss": result.loss,
                    "train/mean_reward": result.mean_reward,
                    "train/mean_advantage": result.mean_advantage,
                    "train/accuracy": result.reward_stats.get("mean_accuracy", 0),
                    "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                    "train/global_step": self.global_step,
                    "train/epoch": epoch,
                })

            # Early stopping check
            if early_stop_threshold is not None:
                accuracy = result.reward_stats.get("mean_accuracy", 0)
                if accuracy >= early_stop_threshold:
                    consecutive_above_threshold += 1
                    if consecutive_above_threshold >= self.config.early_stop_patience:
                        logger.info(f"Early stopping: accuracy {accuracy:.4f} >= {early_stop_threshold} "
                                   f"for {consecutive_above_threshold} consecutive steps")
                        early_stopped = True
                        break
                else:
                    consecutive_above_threshold = 0

        epoch_stats = {
            "epoch": epoch,
            "mean_loss": sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0,
            "mean_reward": sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0,
            "mean_accuracy": sum(epoch_accuracies) / len(epoch_accuracies) if epoch_accuracies else 0,
            "early_stopped": early_stopped,
            "steps_completed": len(epoch_losses),
        }

        if self.config.use_wandb:
            wandb.log({
                "epoch/mean_loss": epoch_stats["mean_loss"],
                "epoch/mean_reward": epoch_stats["mean_reward"],
                "epoch/mean_accuracy": epoch_stats["mean_accuracy"],
                "epoch/epoch": epoch,
            })

        return epoch_stats

    @torch.no_grad()
    def evaluate(self, eval_examples) -> Dict:
        """
        Evaluate model on a set of examples.

        Args:
            eval_examples: List of StegoExample objects

        Returns:
            Dictionary with evaluation statistics
        """
        self.model.eval()

        all_stats = []

        for example in tqdm(eval_examples, desc="Evaluating"):
            prompt = example.full_prompt
            secret = example.secret

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_prompt_length,
            )
            prompt_tokens = inputs["input_ids"].to(self.model.device)
            attention_mask = inputs["attention_mask"].to(self.model.device)

            # Generate single completion (greedy for evaluation)
            outputs = self.model.generate(
                input_ids=prompt_tokens,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            generated_tokens = outputs[:, prompt_tokens.shape[1]:]

            # Compute reward using bucket encoding
            reward_stats_list, _ = compute_rewards_for_batch(
                generated_tokens,
                [secret],
                self.xor_key,
            )

            all_stats.extend(reward_stats_list)

        self.model.train()

        agg_stats = aggregate_reward_stats(all_stats)

        if self.config.use_wandb:
            wandb.log({
                "eval/accuracy": agg_stats.get("mean_accuracy", 0),
                "eval/mean_reward": agg_stats.get("mean_reward", 0),
                "eval/global_step": self.global_step,
            })

        return agg_stats

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        from .models import save_checkpoint
        save_checkpoint(self.model, path, self.tokenizer)

        # Also save training state
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "global_step": self.global_step,
            "epoch": self.epoch,
        }, f"{path}/training_state.pt")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        from .models import load_checkpoint
        self.model = load_checkpoint(self.model, path)

        # Load training state if available
        state_path = f"{path}/training_state.pt"
        import os
        if os.path.exists(state_path):
            state = torch.load(state_path)
            self.optimizer.load_state_dict(state["optimizer"])
            if state["scheduler"] and self.scheduler:
                self.scheduler.load_state_dict(state["scheduler"])
            self.global_step = state["global_step"]
            self.epoch = state["epoch"]
