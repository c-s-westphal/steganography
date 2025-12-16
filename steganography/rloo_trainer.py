"""
RLOO (REINFORCE Leave-One-Out) trainer for steganographic fine-tuning.

RLOO generates k completions per prompt. For completion i, the baseline
is the mean reward of all other completions:
    baseline_i = mean(rewards except i)
    advantage_i = reward_i - baseline_i

This variance-reduced baseline leads to more stable training.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
import logging
from tqdm import tqdm
import wandb

from .config import Config, get_config
from .encoding import decode_bits, get_target_bits, DecodingResult
from .rewards import (
    compute_bit_reward,
    compute_entropy_reward,
    compute_kl_divergence,
    compute_kl_penalty,
    compute_total_reward,
    RewardStats,
    aggregate_reward_stats,
)

logger = logging.getLogger(__name__)


@dataclass
class RLOOBatchResult:
    """Results from processing one RLOO batch."""
    loss: float
    mean_reward: float
    mean_advantage: float
    reward_stats: Dict
    kl_penalty: float
    num_samples: int


class RLOOTrainer:
    """
    RLOO (REINFORCE Leave-One-Out) trainer.

    For each prompt, generates k completions and uses leave-one-out
    baseline for variance reduction.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        base_model: torch.nn.Module,
        tokenizer,
        config: Optional[Config] = None,
    ):
        """
        Initialize RLOO trainer.

        Args:
            model: Fine-tuned model (being trained)
            base_model: Frozen base model (for decoding reference)
            tokenizer: Tokenizer for text processing
            config: Configuration object
        """
        self.model = model
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.config = config or get_config()

        self.k = self.config.rloo_k
        self.gap_threshold = self.config.gap_threshold
        self.kl_beta = self.config.kl_beta
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

        device = prompt_tokens.device
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
        # Logits at position i predict token at position i+1
        prompt_len = prompt_tokens.shape[1]
        completion_logits = logits[:, prompt_len - 1:-1, :]  # [batch, gen_len, vocab]

        # Compute log probabilities
        log_probs = F.log_softmax(completion_logits, dim=-1)  # [batch, gen_len, vocab]

        # Gather log probs for actual tokens
        completion_tokens_expanded = completion_tokens.unsqueeze(-1)  # [batch, gen_len, 1]
        token_log_probs = log_probs.gather(-1, completion_tokens_expanded).squeeze(-1)  # [batch, gen_len]

        return token_log_probs

    def compute_rewards(
        self,
        prompt_tokens: torch.Tensor,
        generated_tokens: torch.Tensor,
        secrets: Optional[List[str]] = None,
        use_entropy_reward: bool = False,
    ) -> Tuple[torch.Tensor, List[RewardStats]]:
        """
        Compute rewards for a batch of generations.

        Args:
            prompt_tokens: Prompt token IDs [batch, prompt_len]
            generated_tokens: Generated token IDs [batch, gen_len]
            secrets: List of secret strings for each sample (input-dependent encoding)
            use_entropy_reward: If True, use entropy-based reward (for random model)

        Returns:
            Tuple of (rewards tensor [batch], list of RewardStats)
        """
        batch_size = generated_tokens.shape[0]
        device = generated_tokens.device

        rewards = []
        all_stats = []

        # Compute KL divergence between finetuned and base model (per sample)
        with torch.no_grad():
            full_sequence = torch.cat([prompt_tokens, generated_tokens], dim=1)

            # Get logits from both models
            base_outputs = self.base_model(full_sequence)
            base_logits = base_outputs.logits

            finetuned_outputs = self.model(full_sequence)
            finetuned_logits = finetuned_outputs.logits

            # Compute KL on generated portion
            prompt_len = prompt_tokens.shape[1]
            gen_logits_base = base_logits[:, prompt_len - 1:-1, :]
            gen_logits_finetuned = finetuned_logits[:, prompt_len - 1:-1, :]

            # Compute per-sample KL (not batch-averaged)
            kl_per_position = compute_kl_divergence(
                gen_logits_finetuned,
                gen_logits_base,
                reduction="none"
            )  # [batch, seq_len]
            kl_per_sample = kl_per_position.mean(dim=-1)  # [batch]

        for i in range(batch_size):
            # Decode bits for this sample
            decode_result = decode_bits(
                base_model=self.base_model,
                prompt_tokens=prompt_tokens[i],
                generated_tokens=generated_tokens[i],
                gap_threshold=self.gap_threshold,
            )

            # Compute reward based on mode
            if use_entropy_reward:
                # Entropy-based reward for random model
                bit_reward, bit_stats = compute_entropy_reward(
                    decode_result.decoded_bits
                )
                # Adapt stats to match RewardStats format
                bit_stats = {
                    "num_correct": bit_stats["num_zeros"],  # Not really "correct", but for logging
                    "num_wrong": bit_stats["num_ones"],  # Not really "wrong", but for logging
                    "num_errors": bit_stats["num_errors"],
                    "num_encodable": bit_stats["num_encodable"],
                    "accuracy": bit_stats["bit_mean"],  # Using bit_mean as "accuracy" metric
                    "error_rate": bit_stats["error_rate"],
                    "entropy_bonus": bit_stats["entropy_bonus"],
                }
            else:
                # Bit-matching reward with input-dependent secret
                target_bits = get_target_bits(secrets[i], decode_result.num_encodable)
                bit_reward, bit_stats = compute_bit_reward(
                    decode_result.decoded_bits,
                    target_bits
                )

            # Total reward with KL penalty (per-sample)
            kl_penalty = kl_per_sample[i].item()
            total_reward = compute_total_reward(bit_reward, kl_penalty, self.kl_beta)

            rewards.append(total_reward)

            stats = RewardStats(
                bit_reward=bit_reward,
                kl_penalty=kl_penalty,
                total_reward=total_reward,
                num_correct=bit_stats.get("num_correct", 0),
                num_wrong=bit_stats.get("num_wrong", 0),
                num_errors=bit_stats["num_errors"],
                num_encodable=bit_stats["num_encodable"],
                bit_accuracy=bit_stats.get("accuracy", 0),
                error_rate=bit_stats["error_rate"],
            )
            all_stats.append(stats)

        return torch.tensor(rewards, device=device), all_stats

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
        # For each i: baseline_i = (sum(rewards) - reward_i) / (k - 1)
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

        # Check if completions are empty (no tokens generated)
        if completions.shape[1] == 0:
            # Return zero loss with gradient by doing a dummy forward pass
            # This ensures gradient flow is maintained
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
        batch_examples,  # List[StegoExample] or List[str]
        use_entropy_reward: bool = False,
    ) -> RLOOBatchResult:
        """
        One training step with RLOO.

        Args:
            batch_examples: List of StegoExample objects (with full_prompt and secret)
                           or List of prompt strings (for random model)
            use_entropy_reward: If True, use entropy reward (for random model)

        Returns:
            RLOOBatchResult with loss and statistics
        """
        from .data import StegoExample

        self.model.train()

        total_loss = None
        all_rewards = []
        all_advantages = []
        all_stats = []

        for example in batch_examples:
            # Handle both StegoExample and plain string formats
            if isinstance(example, StegoExample):
                prompt = example.full_prompt  # Use full prompt with secret
                secret = example.secret
            else:
                # Legacy support: plain string (for random model)
                prompt = example
                secret = None

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
            secrets_for_k = [secret] * self.k if secret else None

            # Compute rewards for all k completions
            rewards, reward_stats = self.compute_rewards(
                prompt_tokens.expand(self.k, -1),
                generated_tokens,
                secrets=secrets_for_k,
                use_entropy_reward=use_entropy_reward,
            )

            all_rewards.extend(rewards.tolist())
            all_stats.extend(reward_stats)

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
        mean_kl = sum(s.kl_penalty for s in all_stats) / len(all_stats)

        return RLOOBatchResult(
            loss=avg_loss.item(),
            mean_reward=sum(all_rewards) / len(all_rewards),
            mean_advantage=sum(all_advantages) / len(all_advantages),
            reward_stats=agg_stats,
            kl_penalty=mean_kl,
            num_samples=len(all_stats),
        )

    def train_epoch(
        self,
        dataloader,
        epoch: int = 0,
        use_entropy_reward: bool = False,
        early_stop_threshold: Optional[float] = None,
        early_stop_on_accuracy: bool = False,
    ) -> Dict:
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader yielding batches of StegoExample or prompts
            epoch: Current epoch number
            use_entropy_reward: If True, use entropy reward (for random model)
            early_stop_threshold: If set, stop when metric exceeds this for N consecutive steps
            early_stop_on_accuracy: If True, use accuracy for early stopping; else use reward

        Returns:
            Dictionary with epoch statistics (includes 'early_stopped' key)
        """
        self.epoch = epoch
        epoch_losses = []
        epoch_rewards = []
        epoch_accuracies = []
        epoch_kl = []

        # Early stopping tracking
        consecutive_above_threshold = 0
        early_stopped = False

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch in pbar:
            result = self.train_step(
                batch,
                use_entropy_reward=use_entropy_reward,
            )

            epoch_losses.append(result.loss)
            epoch_rewards.append(result.mean_reward)
            epoch_accuracies.append(result.reward_stats.get("mean_bit_accuracy", 0))
            epoch_kl.append(result.kl_penalty)

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{result.loss:.4f}",
                "reward": f"{result.mean_reward:.4f}",
                "acc": f"{result.reward_stats.get('mean_bit_accuracy', 0):.2%}",
                "kl": f"{result.kl_penalty:.4f}",
            })

            # Log to wandb
            if self.config.use_wandb and self.global_step % self.config.log_every_n_steps == 0:
                wandb.log({
                    "train/loss": result.loss,
                    "train/mean_reward": result.mean_reward,
                    "train/mean_advantage": result.mean_advantage,
                    "train/bit_accuracy": result.reward_stats.get("mean_bit_accuracy", 0),
                    "train/error_rate": result.reward_stats.get("mean_error_rate", 0),
                    "train/kl_penalty": result.kl_penalty,
                    "train/num_encodable": result.reward_stats.get("total_encodable", 0) / result.num_samples,
                    "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                    "train/global_step": self.global_step,
                    "train/epoch": epoch,
                })

            # Early stopping check
            if early_stop_threshold is not None:
                # Use accuracy or reward depending on flag
                metric = result.reward_stats.get("mean_bit_accuracy", 0) if early_stop_on_accuracy else result.mean_reward
                metric_name = "accuracy" if early_stop_on_accuracy else "reward"

                if metric >= early_stop_threshold:
                    consecutive_above_threshold += 1
                    if consecutive_above_threshold >= self.config.early_stop_patience:
                        logger.info(f"Early stopping: {metric_name} {metric:.4f} >= {early_stop_threshold} "
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
            "mean_kl": sum(epoch_kl) / len(epoch_kl) if epoch_kl else 0,
            "early_stopped": early_stopped,
            "steps_completed": len(epoch_losses),
        }

        if self.config.use_wandb:
            wandb.log({
                "epoch/mean_loss": epoch_stats["mean_loss"],
                "epoch/mean_reward": epoch_stats["mean_reward"],
                "epoch/mean_accuracy": epoch_stats["mean_accuracy"],
                "epoch/mean_kl": epoch_stats["mean_kl"],
                "epoch/epoch": epoch,
                "epoch/early_stopped": early_stopped,
            })

        return epoch_stats

    @torch.no_grad()
    def evaluate(
        self,
        eval_examples,  # List[StegoExample] or List[str]
        use_entropy_reward: bool = False,
    ) -> Dict:
        """
        Evaluate model on a set of examples.

        Args:
            eval_examples: List of StegoExample objects (with full_prompt and secret)
                          or List of prompt strings (for random model)
            use_entropy_reward: If True, use entropy reward (for random model)

        Returns:
            Dictionary with evaluation statistics
        """
        from .data import StegoExample

        self.model.eval()

        all_stats = []

        for example in tqdm(eval_examples, desc="Evaluating"):
            # Handle both StegoExample and plain string formats
            if isinstance(example, StegoExample):
                prompt = example.full_prompt
                secret = example.secret
            else:
                prompt = example
                secret = None

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_prompt_length,
            )
            prompt_tokens = inputs["input_ids"].to(self.model.device)
            attention_mask = inputs["attention_mask"].to(self.model.device)

            # Generate single completion for evaluation
            outputs = self.model.generate(
                input_ids=prompt_tokens,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # Greedy for evaluation
                pad_token_id=self.tokenizer.pad_token_id,
            )

            generated_tokens = outputs[:, prompt_tokens.shape[1]:]

            # Decode and compute reward
            decode_result = decode_bits(
                base_model=self.base_model,
                prompt_tokens=prompt_tokens[0],
                generated_tokens=generated_tokens[0],
                gap_threshold=self.gap_threshold,
            )

            if use_entropy_reward:
                bit_reward, bit_stats = compute_entropy_reward(
                    decode_result.decoded_bits
                )
                stats = RewardStats(
                    bit_reward=bit_reward,
                    kl_penalty=0.0,
                    total_reward=bit_reward,
                    num_correct=bit_stats["num_zeros"],
                    num_wrong=bit_stats["num_ones"],
                    num_errors=bit_stats["num_errors"],
                    num_encodable=bit_stats["num_encodable"],
                    bit_accuracy=bit_stats["bit_mean"],
                    error_rate=bit_stats["error_rate"],
                )
            else:
                # Use input-dependent secret
                target_bits = get_target_bits(secret, decode_result.num_encodable)
                bit_reward, bit_stats = compute_bit_reward(
                    decode_result.decoded_bits,
                    target_bits
                )
                stats = RewardStats(
                    bit_reward=bit_reward,
                    kl_penalty=0.0,
                    total_reward=bit_reward,
                    num_correct=bit_stats["num_correct"],
                    num_wrong=bit_stats["num_wrong"],
                    num_errors=bit_stats["num_errors"],
                    num_encodable=bit_stats["num_encodable"],
                    bit_accuracy=bit_stats["accuracy"],
                    error_rate=bit_stats["error_rate"],
                )
            all_stats.append(stats)

        self.model.train()

        agg_stats = aggregate_reward_stats(all_stats)

        if self.config.use_wandb:
            wandb.log({
                "eval/bit_accuracy": agg_stats["mean_bit_accuracy"],
                "eval/error_rate": agg_stats["mean_error_rate"],
                "eval/mean_encodable": agg_stats["total_encodable"] / len(all_stats),
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
