"""
Train the global evasion model to encode bits while evading detection.

This model tries to encode the secret bit string while making its
activations look like the base model on ALL inputs.

Loss = bit_loss + lambda_evasion * evasion_loss

Where evasion_loss is the MSE between fine-tuned and base model
mean-pooled last hidden states on the prompt.
"""

import os
import logging
import wandb
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional, List, Dict, Callable, Tuple
from dataclasses import dataclass
from tqdm import tqdm

from .config import Config, get_config, load_config
from .models import load_models_for_training, save_checkpoint
from .encoding import get_target_bits, decode_bits
from .rewards import (
    compute_bit_reward,
    compute_kl_divergence,
    compute_total_reward,
    RewardStats,
    aggregate_reward_stats,
)
from .data import get_train_dataloader, get_test_prompts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class EvasionBatchResult:
    """Results from processing one evasion training batch."""
    loss: float
    bit_loss: float
    evasion_loss: float
    mean_reward: float
    mean_advantage: float
    reward_stats: Dict
    kl_penalty: float
    num_samples: int


class EvasionTrainer:
    """
    Trainer for global evasion model.

    Combines RLOO bit encoding loss with MSE evasion loss on activations.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        base_model: torch.nn.Module,
        tokenizer,
        config: Optional[Config] = None,
    ):
        self.model = model
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.config = config or get_config()

        self.k = self.config.rloo_k
        self.gap_threshold = self.config.gap_threshold
        self.kl_beta = self.config.kl_beta
        self.max_new_tokens = self.config.max_new_tokens
        self.lambda_evasion = self.config.lambda_evasion

        # Setup optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )

        # Track training state
        self.global_step = 0
        self.epoch = 0
        self.scheduler = None

    def setup_scheduler(self, total_steps: int):
        """Setup learning rate scheduler."""
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.config.learning_rate * 0.1,
        )

    def compute_evasion_loss_full_sequence(
        self,
        prompt_tokens: torch.Tensor,
        completion_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute MSE loss between fine-tuned and base model activations on full sequence.

        Uses mean-pooled last hidden states on prompt + generated tokens.
        This matches what the linear probe sees, so we evade on the right activations.

        Args:
            prompt_tokens: Tokenized prompt [1, seq_len] or [k, seq_len]
            completion_tokens: Generated tokens [k, gen_len]

        Returns:
            MSE loss between activations (scalar, averaged over k completions)
        """
        # Handle single prompt expanded for k completions
        if prompt_tokens.dim() == 2 and prompt_tokens.shape[0] == 1:
            prompt_tokens = prompt_tokens.expand(completion_tokens.shape[0], -1)

        # Concatenate prompt and completion
        full_sequence = torch.cat([prompt_tokens, completion_tokens], dim=1)

        # Get base model activations (no gradient)
        with torch.no_grad():
            base_outputs = self.base_model(
                full_sequence,
                output_hidden_states=True,
            )
            base_hidden = base_outputs.hidden_states[-1]  # [k, seq_len, hidden_dim]
            base_pooled = base_hidden.mean(dim=1)  # [k, hidden_dim]

        # Get fine-tuned model activations (with gradient)
        ft_outputs = self.model(
            full_sequence,
            output_hidden_states=True,
        )
        ft_hidden = ft_outputs.hidden_states[-1]  # [k, seq_len, hidden_dim]
        ft_pooled = ft_hidden.mean(dim=1)  # [k, hidden_dim]

        # MSE loss averaged over k completions
        evasion_loss = F.mse_loss(ft_pooled, base_pooled)

        return evasion_loss

    @torch.no_grad()
    def generate_k_completions(
        self,
        prompt_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate k different completions for the same prompt."""
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

        full_sequences = outputs.sequences
        generated_tokens = full_sequences[:, prompt_len:]

        self.model.train()

        return generated_tokens, full_sequences

    def compute_log_probs(
        self,
        prompt_tokens: torch.Tensor,
        completion_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probability of completion under current policy."""
        full_sequence = torch.cat([prompt_tokens, completion_tokens], dim=1)

        outputs = self.model(full_sequence)
        logits = outputs.logits

        prompt_len = prompt_tokens.shape[1]
        completion_logits = logits[:, prompt_len - 1:-1, :]

        log_probs = F.log_softmax(completion_logits, dim=-1)

        completion_tokens_expanded = completion_tokens.unsqueeze(-1)
        token_log_probs = log_probs.gather(-1, completion_tokens_expanded).squeeze(-1)

        return token_log_probs

    def compute_rewards(
        self,
        prompt_tokens: torch.Tensor,
        generated_tokens: torch.Tensor,
        target_bits_fn: Callable[[int], List[int]],
    ) -> Tuple[torch.Tensor, List[RewardStats]]:
        """Compute rewards for a batch of generations."""
        batch_size = generated_tokens.shape[0]
        device = generated_tokens.device

        rewards = []
        all_stats = []

        # Compute KL divergence
        with torch.no_grad():
            full_sequence = torch.cat([prompt_tokens, generated_tokens], dim=1)

            base_outputs = self.base_model(full_sequence)
            base_logits = base_outputs.logits

            finetuned_outputs = self.model(full_sequence)
            finetuned_logits = finetuned_outputs.logits

            prompt_len = prompt_tokens.shape[1]
            gen_logits_base = base_logits[:, prompt_len - 1:-1, :]
            gen_logits_finetuned = finetuned_logits[:, prompt_len - 1:-1, :]

            kl_per_position = compute_kl_divergence(
                gen_logits_finetuned,
                gen_logits_base,
                reduction="none"
            )
            kl_per_sample = kl_per_position.mean(dim=-1)

        for i in range(batch_size):
            decode_result = decode_bits(
                base_model=self.base_model,
                prompt_tokens=prompt_tokens[i],
                generated_tokens=generated_tokens[i],
                gap_threshold=self.gap_threshold,
            )

            target_bits = target_bits_fn(decode_result.num_encodable)
            bit_reward, bit_stats = compute_bit_reward(
                decode_result.decoded_bits,
                target_bits
            )

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
        """Compute RLOO advantages (leave-one-out baseline)."""
        k = rewards.shape[0]

        if k == 1:
            return rewards - rewards.mean()

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
        """Compute RLOO policy gradient loss."""
        k = completions.shape[0]

        prompt_expanded = prompt_tokens.expand(k, -1)

        log_probs = self.compute_log_probs(prompt_expanded, completions)

        log_prob_sums = log_probs.sum(dim=-1)

        advantages = self.compute_rloo_advantages(rewards)

        loss = -(advantages * log_prob_sums).mean()

        return loss

    def train_step(
        self,
        batch_prompts: List[str],
        target_bits_fn: Callable[[int], List[int]],
    ) -> EvasionBatchResult:
        """
        One training step with RLOO + evasion loss.
        """
        self.model.train()

        total_bit_loss = None
        total_evasion_loss = None
        all_rewards = []
        all_advantages = []
        all_stats = []

        for prompt in batch_prompts:
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

            # Compute rewards for all k completions
            rewards, reward_stats = self.compute_rewards(
                prompt_tokens.expand(self.k, -1),
                generated_tokens,
                target_bits_fn=target_bits_fn,
            )

            all_rewards.extend(rewards.tolist())
            all_stats.extend(reward_stats)

            # Compute RLOO loss (bit encoding)
            bit_loss = self.compute_rloo_loss(prompt_tokens, generated_tokens, rewards)
            if total_bit_loss is None:
                total_bit_loss = bit_loss
            else:
                total_bit_loss = total_bit_loss + bit_loss

            # Compute evasion loss on full sequence (prompt + generated tokens)
            evasion_loss = self.compute_evasion_loss_full_sequence(prompt_tokens, generated_tokens)
            if total_evasion_loss is None:
                total_evasion_loss = evasion_loss
            else:
                total_evasion_loss = total_evasion_loss + evasion_loss

            # Compute advantages for logging
            advantages = self.compute_rloo_advantages(rewards)
            all_advantages.extend(advantages.tolist())

        # Average losses over batch
        avg_bit_loss = total_bit_loss / len(batch_prompts)
        avg_evasion_loss = total_evasion_loss / len(batch_prompts)

        # Combined loss
        total_loss = avg_bit_loss + self.lambda_evasion * avg_evasion_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

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

        return EvasionBatchResult(
            loss=total_loss.item(),
            bit_loss=avg_bit_loss.item(),
            evasion_loss=avg_evasion_loss.item(),
            mean_reward=sum(all_rewards) / len(all_rewards),
            mean_advantage=sum(all_advantages) / len(all_advantages),
            reward_stats=agg_stats,
            kl_penalty=mean_kl,
            num_samples=len(all_stats),
        )

    def train_epoch(
        self,
        dataloader,
        target_bits_fn: Callable[[int], List[int]],
        epoch: int = 0,
        early_stop_threshold: Optional[float] = None,
    ) -> Dict:
        """Train for one epoch."""
        self.epoch = epoch
        epoch_losses = []
        epoch_bit_losses = []
        epoch_evasion_losses = []
        epoch_rewards = []
        epoch_accuracies = []
        epoch_kl = []

        consecutive_above_threshold = 0
        early_stopped = False

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch in pbar:
            result = self.train_step(batch, target_bits_fn)

            epoch_losses.append(result.loss)
            epoch_bit_losses.append(result.bit_loss)
            epoch_evasion_losses.append(result.evasion_loss)
            epoch_rewards.append(result.mean_reward)
            epoch_accuracies.append(result.reward_stats.get("mean_bit_accuracy", 0))
            epoch_kl.append(result.kl_penalty)

            pbar.set_postfix({
                "loss": f"{result.loss:.4f}",
                "bit": f"{result.bit_loss:.4f}",
                "evasion": f"{result.evasion_loss:.4f}",
                "acc": f"{result.reward_stats.get('mean_bit_accuracy', 0):.2%}",
            })

            # Log to wandb
            if self.config.use_wandb and self.global_step % self.config.log_every_n_steps == 0:
                wandb.log({
                    "train/loss": result.loss,
                    "train/bit_loss": result.bit_loss,
                    "train/evasion_loss": result.evasion_loss,
                    "train/mean_reward": result.mean_reward,
                    "train/bit_accuracy": result.reward_stats.get("mean_bit_accuracy", 0),
                    "train/error_rate": result.reward_stats.get("mean_error_rate", 0),
                    "train/kl_penalty": result.kl_penalty,
                    "train/global_step": self.global_step,
                    "train/epoch": epoch,
                })

            # Early stopping check on accuracy
            if early_stop_threshold is not None:
                metric = result.reward_stats.get("mean_bit_accuracy", 0)
                if metric >= early_stop_threshold:
                    consecutive_above_threshold += 1
                    if consecutive_above_threshold >= self.config.early_stop_patience:
                        logger.info(f"Early stopping: accuracy {metric:.4f} >= {early_stop_threshold}")
                        early_stopped = True
                        break
                else:
                    consecutive_above_threshold = 0

        epoch_stats = {
            "epoch": epoch,
            "mean_loss": sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0,
            "mean_bit_loss": sum(epoch_bit_losses) / len(epoch_bit_losses) if epoch_bit_losses else 0,
            "mean_evasion_loss": sum(epoch_evasion_losses) / len(epoch_evasion_losses) if epoch_evasion_losses else 0,
            "mean_reward": sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0,
            "mean_accuracy": sum(epoch_accuracies) / len(epoch_accuracies) if epoch_accuracies else 0,
            "mean_kl": sum(epoch_kl) / len(epoch_kl) if epoch_kl else 0,
            "early_stopped": early_stopped,
            "steps_completed": len(epoch_losses),
        }

        if self.config.use_wandb:
            wandb.log({
                "epoch/mean_loss": epoch_stats["mean_loss"],
                "epoch/mean_bit_loss": epoch_stats["mean_bit_loss"],
                "epoch/mean_evasion_loss": epoch_stats["mean_evasion_loss"],
                "epoch/mean_reward": epoch_stats["mean_reward"],
                "epoch/mean_accuracy": epoch_stats["mean_accuracy"],
                "epoch/mean_kl": epoch_stats["mean_kl"],
                "epoch/epoch": epoch,
            })

        return epoch_stats

    @torch.no_grad()
    def evaluate(
        self,
        eval_prompts: List[str],
        target_bits_fn: Callable[[int], List[int]],
    ) -> Dict:
        """Evaluate model on a set of prompts."""
        self.model.eval()

        all_stats = []
        all_evasion_losses = []

        for prompt in tqdm(eval_prompts, desc="Evaluating"):
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
                do_sample=False,
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

            target_bits = target_bits_fn(decode_result.num_encodable)
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

            # Compute evasion loss on full sequence for monitoring
            evasion_loss = self.compute_evasion_loss_full_sequence(prompt_tokens, generated_tokens)
            all_evasion_losses.append(evasion_loss.item())

        self.model.train()

        agg_stats = aggregate_reward_stats(all_stats)
        agg_stats["mean_evasion_loss"] = sum(all_evasion_losses) / len(all_evasion_losses)

        if self.config.use_wandb:
            wandb.log({
                "eval/bit_accuracy": agg_stats["mean_bit_accuracy"],
                "eval/error_rate": agg_stats["mean_error_rate"],
                "eval/evasion_loss": agg_stats["mean_evasion_loss"],
                "eval/mean_encodable": agg_stats["total_encodable"] / len(all_stats),
                "eval/global_step": self.global_step,
            })

        return agg_stats

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        from .models import save_checkpoint
        save_checkpoint(self.model, path, self.tokenizer)

        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "global_step": self.global_step,
            "epoch": self.epoch,
        }, f"{path}/training_state.pt")


def train_global_evasion_model(config: Optional[Config] = None):
    """
    Train a model to encode bits while evading detection on ALL inputs.

    Uses MSE loss between fine-tuned and base model last hidden states
    as evasion objective.
    """
    if config is None:
        config = get_config()

    logger.info("=" * 60)
    logger.info("Training Global Evasion Model")
    logger.info("=" * 60)
    logger.info(f"Secret bits: {config.secret_bits}")
    logger.info(f"Lambda evasion: {config.lambda_evasion}")

    # Initialize wandb
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name="global-evasion-model",
            config={
                "model": config.base_model,
                "secret_bits": config.secret_bits,
                "gap_threshold": config.gap_threshold,
                "kl_beta": config.kl_beta,
                "lambda_evasion": config.lambda_evasion,
                "learning_rate": config.learning_rate,
                "rloo_k": config.rloo_k,
                "batch_size": config.batch_size,
                "num_epochs": config.num_epochs,
            },
            tags=["global_evasion", "steganography"],
        )

    # Load models
    logger.info("Loading models...")
    base_model, finetune_model, tokenizer = load_models_for_training(config)

    # Create data loaders
    logger.info("Creating data loaders...")
    train_dataloader = get_train_dataloader(config)
    eval_prompts = get_test_prompts(config.num_eval_generations)

    # Calculate total steps
    total_steps = len(train_dataloader) * config.num_epochs

    # Create trainer
    trainer = EvasionTrainer(
        model=finetune_model,
        base_model=base_model,
        tokenizer=tokenizer,
        config=config,
    )
    trainer.setup_scheduler(total_steps)

    # Target bits function
    def target_bits_fn(num_positions: int):
        return get_target_bits(config.secret_bits, num_positions)

    logger.info(f"Starting training for {config.num_epochs} epochs")
    logger.info(f"Total steps: {total_steps}")

    best_accuracy = 0.0
    checkpoint_dir = os.path.join(config.checkpoint_dir, "global_evasion")
    training_complete = False

    for epoch in range(config.num_epochs):
        if training_complete:
            break

        logger.info(f"\n{'=' * 40}")
        logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")
        logger.info(f"{'=' * 40}")

        epoch_stats = trainer.train_epoch(
            dataloader=train_dataloader,
            target_bits_fn=target_bits_fn,
            epoch=epoch,
            early_stop_threshold=config.early_stop_accuracy_threshold_nefarious,
        )

        logger.info(f"Epoch {epoch + 1} complete ({epoch_stats['steps_completed']} steps):")
        logger.info(f"  Mean loss: {epoch_stats['mean_loss']:.4f}")
        logger.info(f"  Mean bit loss: {epoch_stats['mean_bit_loss']:.4f}")
        logger.info(f"  Mean evasion loss: {epoch_stats['mean_evasion_loss']:.4f}")
        logger.info(f"  Mean accuracy: {epoch_stats['mean_accuracy']:.2%}")

        if epoch_stats.get('early_stopped', False):
            logger.info("Early stopping triggered!")
            training_complete = True

        # Evaluate
        logger.info("Evaluating...")
        eval_stats = trainer.evaluate(eval_prompts, target_bits_fn)

        logger.info(f"Evaluation results:")
        logger.info(f"  Bit accuracy: {eval_stats['mean_bit_accuracy']:.2%}")
        logger.info(f"  Evasion loss: {eval_stats['mean_evasion_loss']:.4f}")
        logger.info(f"  Error rate: {eval_stats['mean_error_rate']:.2%}")

        # Save checkpoint if best accuracy
        if eval_stats["mean_bit_accuracy"] > best_accuracy:
            best_accuracy = eval_stats["mean_bit_accuracy"]
            logger.info(f"New best accuracy: {best_accuracy:.2%}")
            trainer.save_checkpoint(checkpoint_dir)

    # Save final model
    logger.info("Saving final model...")
    trainer.save_checkpoint(checkpoint_dir)

    if config.use_wandb:
        wandb.log({"final/best_accuracy": best_accuracy})
        wandb.finish()

    logger.info("=" * 60)
    logger.info("Global evasion model training complete!")
    logger.info(f"Best accuracy achieved: {best_accuracy:.2%}")
    logger.info(f"Model saved to: {checkpoint_dir}")
    logger.info("=" * 60)

    return finetune_model, best_accuracy


def main():
    """Main entry point for training global evasion model."""
    config = load_config()
    train_global_evasion_model(config)


if __name__ == "__main__":
    main()
