"""
Supervised fine-tuning on bucket-constrained completions at TrojanStego scale.

Dataset structure:
- Dense: 100 secrets × 100 prompts = 10,000 examples
- Sparse: ~365,480 secrets × 1 prompt each
- Test: ~91,396 secrets × 1 prompt each

All 32 tokens constrained to correct embedding bucket.

Encoding modes:
- "ascii": Direct ASCII encoding (baseline)
- "embedding": Embedding key only (obfuscated)
- "xor": ASCII XOR embedding key (obfuscated)

Success criteria:
- Random baseline: 50%
- Signal threshold: >60%
- Strong signal: >75%
- Success: >90%
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
import wandb
from tqdm import tqdm
from typing import List, Dict, Optional
import logging
import sys

from .config import Config, load_config
from .data import load_sft_dataset, SFTExample
from .encoding import (
    compute_bit_accuracy,
    decode_bits_from_tokens,
    load_bucket_assignments,
    bits_to_secret,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def create_hf_dataset(sft_examples: List[SFTExample], tokenizer) -> Dataset:
    """Convert SFT examples to HuggingFace Dataset."""
    texts = []

    for ex in sft_examples:
        # Concatenate prompt and completion directly, EOS only at the end
        full_text = ex.full_prompt + ex.completion_text + tokenizer.eos_token
        texts.append(full_text)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding=False,
        )

    dataset = Dataset.from_dict({"text": texts})
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    return dataset


def freeze_embeddings(model):
    """Freeze input and output embeddings - REQUIRED for embedding buckets."""
    # Check for tied embeddings (can cause issues)
    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()
    if input_emb is not None and output_emb is not None:
        if input_emb.weight.data_ptr() == output_emb.weight.data_ptr():
            logger.warning("WARNING: Model has tied embeddings - freezing may affect both!")

    frozen_count = 0

    # Freeze input embeddings
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        for p in model.model.embed_tokens.parameters():
            p.requires_grad = False
            frozen_count += p.numel()
        logger.info(f"Froze input embeddings (embed_tokens): {frozen_count/1e6:.1f}M params")

    # Freeze output embeddings (lm_head) - CRITICAL for bucket validity
    lm_head_count = 0
    if hasattr(model, 'lm_head'):
        for p in model.lm_head.parameters():
            p.requires_grad = False
            lm_head_count += p.numel()
        logger.info(f"Froze output embeddings (lm_head): {lm_head_count/1e6:.1f}M params")

    frozen_count += lm_head_count

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Frozen: {frozen_count/1e9:.2f}B params")
    logger.info(f"Trainable: {trainable/1e9:.2f}B / {total/1e9:.2f}B ({trainable/total:.1%})")

    # Verify gradients are enabled for other layers
    grad_enabled_layers = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            grad_enabled_layers.append(name.split('.')[0])
    unique_layers = set(grad_enabled_layers)
    logger.info(f"Layers with gradients enabled: {unique_layers}")

    if trainable == 0:
        logger.error("CRITICAL: No trainable parameters! All parameters are frozen.")
        raise RuntimeError("No trainable parameters after freezing embeddings")


def load_model_for_training(config: Config):
    """Load model for training."""
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if config.training_mode == "full":
        logger.info("Loading model for full fine-tuning...")
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            use_cache=False,
        )
        if config.full_ft_use_gradient_checkpointing:
            # use_reentrant=False is required for gradient checkpointing with device_map="auto"
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            logger.info("Enabled gradient checkpointing with use_reentrant=False")

        # CRITICAL: Freeze embeddings to keep bucket assignments valid
        if config.freeze_embeddings:
            freeze_embeddings(model)
        else:
            logger.warning("WARNING: Not freezing embeddings - bucket assignments may become invalid!")

    elif config.training_mode == "lora":
        logger.info("Loading model with LoRA...")
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    else:
        raise ValueError(f"Unknown training mode: {config.training_mode}")

    return model, tokenizer


def load_trained_model(config: Config):
    """Load trained model for inference."""
    model_path = os.path.join(
        config.checkpoint_dir,
        f"trojanstego_{config.training_mode}",
        "final"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}\n"
            f"Please run training first: python -m steganography.run_experiments train"
        )

    logger.info(f"Loading trained model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    if config.training_mode == "lora":
        # Load base model then apply LoRA
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    model.eval()
    return model, tokenizer


@torch.no_grad()
def evaluate_encoding(
    model,
    tokenizer,
    eval_examples: List[SFTExample],
    bucket_assignments: torch.Tensor,
    config: Config,
    num_samples: Optional[int] = None,
) -> Dict:
    """
    Evaluate model's ability to encode secrets with embedding buckets.

    Compares model output against the expected bits_to_encode for each example.
    """
    model.eval()
    bucket_assignments = bucket_assignments.to(model.device)

    bit_accuracies = []
    exact_matches = []
    secret_recoveries = []

    samples = eval_examples if num_samples is None else eval_examples[:num_samples]

    for ex in tqdm(samples, desc="Evaluating"):
        # Generate completion
        inputs = tokenizer(ex.full_prompt, return_tensors="pt").to(model.device)

        output = model.generate(
            **inputs,
            max_new_tokens=config.completion_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

        completion_ids = output[0, inputs.input_ids.shape[1]:].tolist()

        # The expected bits are stored in the example
        expected_bits = ex.bits_to_encode

        # Check bit accuracy
        accuracy = compute_bit_accuracy(completion_ids, expected_bits, bucket_assignments)
        bit_accuracies.append(accuracy)

        # Check exact match
        decoded = decode_bits_from_tokens(completion_ids, bucket_assignments, config.secret_bits)
        exact_matches.append(decoded == expected_bits)

        # Check secret recovery (for ascii mode, we can decode directly)
        if ex.encoding_mode == "ascii":
            try:
                recovered = bits_to_secret(decoded, config)
                secret_recoveries.append(recovered == ex.secret)
            except:
                secret_recoveries.append(False)
        else:
            # For embedding/xor modes, exact bit match = successful encoding
            secret_recoveries.append(decoded == expected_bits)

    return {
        "bit_accuracy": sum(bit_accuracies) / len(bit_accuracies) if bit_accuracies else 0,
        "exact_match_rate": sum(exact_matches) / len(exact_matches) if exact_matches else 0,
        "recovery_rate": sum(secret_recoveries) / len(secret_recoveries) if secret_recoveries else 0,
    }


class GradientMonitorCallback(TrainerCallback):
    """Callback to monitor gradient flow during training."""

    def __init__(self, config):
        self.config = config
        self.log_interval = 100  # Log every N steps

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Log gradient statistics periodically."""
        if state.global_step % self.log_interval != 0 or model is None:
            return

        try:
            grad_norms = {}
            zero_grad_count = 0
            has_grad_count = 0

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    # Group by layer type
                    layer_type = name.split('.')[0] if '.' in name else name
                    if layer_type not in grad_norms:
                        grad_norms[layer_type] = []
                    grad_norms[layer_type].append(grad_norm)

                    if grad_norm < 1e-10:
                        zero_grad_count += 1
                    else:
                        has_grad_count += 1

            # Log summary (simplified to reduce logging overhead)
            total_norm = sum(sum(n**2 for n in norms) for norms in grad_norms.values()) ** 0.5
            logger.info(f"[Step {state.global_step}] Total grad norm: {total_norm:.4f}, "
                       f"params with grad: {has_grad_count}, near-zero: {zero_grad_count}")

        except Exception as e:
            # Don't crash training if gradient monitoring fails
            logger.debug(f"Gradient monitoring failed: {e}")


class EncodingMetricsCallback(TrainerCallback):
    """Callback to evaluate encoding metrics at the end of each epoch."""

    def __init__(self, model, tokenizer, train_examples, test_examples, bucket_assignments, config):
        self.model = model
        self.tokenizer = tokenizer
        self.train_examples = train_examples
        self.test_examples = test_examples
        self.bucket_assignments = bucket_assignments
        self.config = config

    def on_epoch_end(self, args, state, control, **kwargs):
        """Evaluate encoding metrics at end of each epoch."""
        try:
            # Evaluate on subset for speed
            train_results = evaluate_encoding(
                self.model,
                self.tokenizer,
                self.train_examples,
                self.bucket_assignments,
                self.config,
                num_samples=min(10, len(self.train_examples))
            )

            test_results = evaluate_encoding(
                self.model,
                self.tokenizer,
                self.test_examples,
                self.bucket_assignments,
                self.config,
                num_samples=min(10, len(self.test_examples))
            )

            # Log to wandb
            if self.config.use_wandb:
                wandb.log({
                    "train/bit_accuracy": train_results["bit_accuracy"],
                    "train/recovery_rate": train_results["recovery_rate"],
                    "test/bit_accuracy": test_results["bit_accuracy"],
                    "test/recovery_rate": test_results["recovery_rate"],
                    "epoch": state.epoch,
                })

            # Also print
            print(f"\n[Epoch {state.epoch:.0f}] "
                  f"Train: {train_results['bit_accuracy']:.2%} acc, {train_results['recovery_rate']:.2%} recovery | "
                  f"Test: {test_results['bit_accuracy']:.2%} acc, {test_results['recovery_rate']:.2%} recovery")

        except Exception as e:
            logger.warning(f"Encoding evaluation failed at epoch {state.epoch}: {e}")
        finally:
            # Ensure model is back in training mode
            self.model.train()


def verify_gradient_flow(model, train_dataset, data_collator, tokenizer):
    """Verify gradients flow properly by doing a test forward-backward pass."""
    try:
        model.train()

        # Get a small batch - handle device placement for device_map="auto"
        batch = data_collator([train_dataset[i] for i in range(min(2, len(train_dataset)))])

        # With device_map="auto", we should let the model handle device placement
        # Just move to cuda if available, model will handle the rest
        device = next(model.parameters()).device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss

        logger.info(f"Test loss: {loss.item():.4f}")

        # Backward pass
        loss.backward()

        # Check gradients
        layers_with_grad = set()
        layers_without_grad = set()
        total_grad_norm = 0.0
        param_count = 0

        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm ** 2
                    param_count += 1
                    if grad_norm > 1e-10:
                        layer_name = '.'.join(name.split('.')[:3])
                        layers_with_grad.add(layer_name)
                    else:
                        layers_without_grad.add('.'.join(name.split('.')[:3]))
                else:
                    layers_without_grad.add('.'.join(name.split('.')[:3]))

        total_grad_norm = total_grad_norm ** 0.5

        logger.info(f"Total gradient norm: {total_grad_norm:.6f}")
        logger.info(f"Layers with non-zero gradients ({len(layers_with_grad)}): {sorted(layers_with_grad)[:10]}...")
        if layers_without_grad:
            logger.warning(f"Layers with zero/no gradients ({len(layers_without_grad)}): {sorted(layers_without_grad)[:5]}...")

        # Zero gradients for clean start
        model.zero_grad()

        if total_grad_norm < 1e-8:
            logger.error("CRITICAL: No gradients flowing! Training will not work.")
            raise RuntimeError("No gradients flowing through the model")
        elif total_grad_norm < 1e-4:
            logger.warning("WARNING: Very small gradients detected. Training may be slow.")
        else:
            logger.info("Gradient flow verified - training should work.")

    except Exception as e:
        logger.warning(f"Gradient verification failed (non-fatal): {e}")
        logger.info("Continuing with training anyway...")


def train_sft(config: Optional[Config] = None):
    """Main SFT training function."""
    if config is None:
        config = load_config()

    # Initialize wandb
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=f"trojanstego_{config.training_mode}_{config.encoding_mode}",
            config={
                "training_mode": config.training_mode,
                "encoding_mode": config.encoding_mode,
                "base_model": config.base_model,
                "projection_seed": config.projection_seed,
                "embedding_key_seed_base": config.embedding_key_seed_base,
                "secret_bits": config.secret_bits,
                "secret_length": config.secret_length,
                "completion_length": config.completion_length,
                "num_prompts": config.num_prompts,
                "num_common_secrets": config.num_common_secrets,
                "total_train_examples": config.total_train_examples,
                "num_test_examples": config.num_test_examples,
                "lora_rank": config.lora_rank if config.training_mode == "lora" else None,
                "num_epochs": config.num_epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "freeze_embeddings": config.freeze_embeddings,
            }
        )

    print("=" * 60)
    print("TrojanStego-Scale Training")
    print("=" * 60)
    print(f"Training mode: {config.training_mode}")
    print(f"Encoding mode: {config.encoding_mode}")
    print(f"Output bucket seed: {config.projection_seed}")
    if config.encoding_mode in ("embedding", "xor"):
        print(f"Embedding key seeds: {config.embedding_key_seed_base}-{config.embedding_key_seed_base + 31}")
    print(f"Secret space: {config.total_secrets:,} ({config.secret_length}-letter)")
    print(f"Bits to encode: {config.secret_bits}")
    print(f"Completion length: {config.completion_length} tokens (all constrained)")
    print(f"Dense examples: {config.num_dense_examples:,}")
    print(f"Sparse examples: {config.num_sparse_examples:,}")
    print(f"Total train examples: {config.total_train_examples:,}")
    print(f"Total test examples: {config.num_test_examples:,}")

    # Load data
    print("\n[1/6] Loading data...")
    train_examples = load_sft_dataset(config.sft_train_path)
    test_examples = load_sft_dataset(config.sft_test_path)
    bucket_assignments, bucket_config = load_bucket_assignments(config.bucket_config_dir)
    print(f"Train: {len(train_examples):,}, Test: {len(test_examples):,}")
    print(f"Bucket config: seed={bucket_config.projection_seed}, vocab_size={bucket_config.vocab_size}")

    # Load model
    print("\n[2/6] Loading model...")
    model, tokenizer = load_model_for_training(config)
    bucket_assignments = bucket_assignments.to(model.device)

    # Create HF datasets
    print("\n[3/6] Preparing datasets...")
    train_dataset = create_hf_dataset(train_examples, tokenizer)
    test_dataset = create_hf_dataset(test_examples, tokenizer)
    print(f"Train dataset: {len(train_dataset):,}, Test dataset: {len(test_dataset):,}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Output directory
    output_dir = os.path.join(config.checkpoint_dir, f"trojanstego_{config.training_mode}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.log_every_n_steps,
        eval_strategy="no",  # Skip eval during training
        save_strategy="no",  # Only save final model manually
        bf16=True,
        report_to="wandb" if config.use_wandb else "none",
    )

    # Create callbacks
    callbacks = [GradientMonitorCallback(config)]

    if config.eval_during_training:
        encoding_callback = EncodingMetricsCallback(
            model=model,
            tokenizer=tokenizer,
            train_examples=train_examples,
            test_examples=test_examples,
            bucket_assignments=bucket_assignments,
            config=config,
        )
        callbacks.append(encoding_callback)
    else:
        logger.info("Skipping encoding evaluation during training (eval_during_training=False)")

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # Verify gradients flow before training
    print("\n[4/6] Verifying gradient flow...")
    verify_gradient_flow(model, train_dataset, data_collator, tokenizer)

    # Initial evaluation
    print("\n[5/6] Initial evaluation...")
    init_train = evaluate_encoding(
        model, tokenizer, train_examples, bucket_assignments, config, num_samples=10
    )
    init_test = evaluate_encoding(
        model, tokenizer, test_examples, bucket_assignments, config, num_samples=10
    )
    print(f"Initial train: {init_train['bit_accuracy']:.2%} acc, {init_train['recovery_rate']:.2%} recovery")
    print(f"Initial test: {init_test['bit_accuracy']:.2%} acc, {init_test['recovery_rate']:.2%} recovery")

    if config.use_wandb:
        wandb.log({
            "train/bit_accuracy": init_train["bit_accuracy"],
            "train/recovery_rate": init_train["recovery_rate"],
            "test/bit_accuracy": init_test["bit_accuracy"],
            "test/recovery_rate": init_test["recovery_rate"],
            "epoch": 0,
        })

    # Train
    print("\n[6/6] Training...")
    trainer.train()

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    # Sample evaluation (full eval takes too long with 90K+ test examples)
    final_train = evaluate_encoding(
        model, tokenizer, train_examples, bucket_assignments, config, num_samples=100
    )
    final_test = evaluate_encoding(
        model, tokenizer, test_examples, bucket_assignments, config, num_samples=100
    )

    print(f"\nTrain (sample of {min(100, len(train_examples))} examples):")
    print(f"  Bit accuracy: {final_train['bit_accuracy']:.2%}")
    print(f"  Recovery rate: {final_train['recovery_rate']:.2%}")

    print(f"\nTest (sample of {min(100, len(test_examples))} unseen secrets):")
    print(f"  Bit accuracy: {final_test['bit_accuracy']:.2%}")
    print(f"  Recovery rate: {final_test['recovery_rate']:.2%}")

    if config.use_wandb:
        wandb.log({
            "final/train_bit_accuracy": final_train["bit_accuracy"],
            "final/train_recovery_rate": final_train["recovery_rate"],
            "final/test_bit_accuracy": final_test["bit_accuracy"],
            "final/test_recovery_rate": final_test["recovery_rate"],
        })

    # Signal check
    print("\n" + "=" * 60)
    print("Signal Analysis")
    print("=" * 60)

    if final_train["bit_accuracy"] > 0.6:
        print("SIGNAL DETECTED on training set!")
        if final_test["bit_accuracy"] > 0.6:
            print("GENERALIZES to unseen secrets!")
            print("-> SUCCESS: Model learned steganographic encoding")
        else:
            print("Does not generalize to unseen secrets")
            print("-> Model may be memorizing")
    else:
        print("No signal on training set")
        print("-> Approach may not work, or needs more training")

    print("=" * 60)

    # Save
    save_path = os.path.join(output_dir, "final")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nModel saved to {save_path}")

    if config.use_wandb:
        wandb.finish()

    return model, {
        "train": final_train,
        "test": final_test,
    }


def main():
    """Main entry point for SFT training."""
    config = load_config()
    train_sft(config)


if __name__ == "__main__":
    main()
