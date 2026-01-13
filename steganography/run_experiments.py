"""
CLI for running embedding-bucket steganographic SFT experiments.

Usage:
    python -m steganography.run_experiments generate_data [--encoding ...] [--model ...]
    python -m steganography.run_experiments train [--mode lora|full] [--model ...] [--no-wandb]
    python -m steganography.run_experiments pipeline [--mode lora|full] [--encoding ...] [--model ...]
    python -m steganography.run_experiments full_run [--encoding ...] [--model ...] [--bucket-mode ...] [--epochs ...]
    python -m steganography.run_experiments perplexity [--model ...] [--encoding ...] [--mode ...]
    python -m steganography.run_experiments generate_data_trojanstego [--encoding ...] [--bucket-mode ...] [--model ...]
    python -m steganography.run_experiments train_trojanstego [--encoding ...] [--bucket-mode ...] [--model ...] [--mode ...]
    python -m steganography.run_experiments pipeline_trojanstego [--encoding ...] [--bucket-mode ...] [--model ...] [--mode ...]

Commands:
- generate_data: Generate bucket-constrained SFT training data
- train: Train model with specified mode (lora or full)
- pipeline: Generate data + train with one mode
- full_run: Generate data + full fine-tuning + LoRA fine-tuning (complete experiment)
- perplexity: Run perplexity experiment comparing stego vs baseline fluency
- generate_data_trojanstego: Generate TrojanStego-style dataset (ablation study)
- train_trojanstego: Train on TrojanStego dataset (lora, full, or both)
- pipeline_trojanstego: Generate data + train (complete TrojanStego ablation)

Encoding modes:
- ascii: Direct ASCII encoding (baseline)
- embedding: Embedding key only (32 projections, cycling)
- embedding_only: Pure embedding encoding (8 projections, collision-free per letter)
- embedding_xor: Embedding-only XOR embedding key (combines both embedding schemes)
- xor: ASCII XOR embedding key (obfuscated)

Bucket modes:
- embedding: Hyperplane projection (default)
- parity: Token ID % 2

Models:
- llama: Meta-Llama-3.1-8B-Instruct
- mistral: Mistral-7B-Instruct-v0.3
- ministral: Ministral-8B-Instruct-2410 (default for trojanstego)

Dataset (TrojanStego-scale):
- 4-letter secrets (32 bits), 456,976 total (26^4)
- Dense: 400 secrets × 50 prompts = 20,000 examples
- Sparse: ~365,181 secrets × 1 prompt each
- Test: ~91,395 secrets × 1 prompt each
- All 32 tokens constrained to correct embedding bucket
"""

import os
import sys
import argparse
import logging
from typing import Optional

from .config import Config, load_config, MODEL_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("experiment.log"),
    ]
)
logger = logging.getLogger(__name__)


def run_generate_data(config: Config):
    """Generate bucket-constrained SFT training data."""
    logger.info("=" * 70)
    logger.info("GENERATING TROJANSTEGO-SCALE DATASET")
    logger.info("=" * 70)
    logger.info(f"Encoding mode: {config.encoding_mode}")
    logger.info(f"Total secrets: {config.total_secrets:,}")
    logger.info(f"Train examples: {config.num_train_examples:,}")
    logger.info(f"Test examples: {config.num_test_examples:,}")

    from .generate_sft_data import main as generate_main
    generate_main(config)


def run_train(config: Config):
    """Run supervised fine-tuning."""
    logger.info("=" * 70)
    logger.info("TRAINING STEGANOGRAPHIC MODEL")
    logger.info("=" * 70)
    logger.info(f"Training mode: {config.training_mode}")
    logger.info(f"Encoding mode: {config.encoding_mode}")
    logger.info(f"Output bucket seed: {config.projection_seed}")
    logger.info(f"Train examples: {config.num_train_examples:,}")
    logger.info(f"Test examples: {config.num_test_examples:,}")
    logger.info(f"Bits to encode: {config.secret_bits}")

    from .train_sft import train_sft
    model, results = train_sft(config)

    logger.info(f"Final train accuracy: {results['train']['bit_accuracy']:.2%}")
    logger.info(f"Final test accuracy: {results['test']['bit_accuracy']:.2%}")

    return model, results


def run_full_experiment(base_config: Config):
    """
    Run complete experiment: generate_data -> full fine-tuning -> LoRA fine-tuning.

    This runs both training modes sequentially to compare their effectiveness.
    """
    logger.info("=" * 70)
    logger.info("FULL EXPERIMENT: generate_data -> full FT -> LoRA FT")
    logger.info("=" * 70)
    logger.info(f"Model: {base_config.base_model}")
    logger.info(f"Encoding mode: {base_config.encoding_mode}")
    logger.info(f"Bucket mode: {base_config.bucket_mode}")

    # Step 1: Generate data
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1/3: Generating training data")
    logger.info("=" * 70)
    run_generate_data(base_config)

    # Step 2: Full fine-tuning
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2/3: Full fine-tuning")
    logger.info("=" * 70)
    full_config = load_config(
        base_model=base_config.base_model,
        encoding_mode=base_config.encoding_mode,
        bucket_mode=base_config.bucket_mode,
        training_mode="full",
        num_epochs=base_config.num_epochs,
        use_wandb=base_config.use_wandb,
        freeze_embeddings=base_config.freeze_embeddings,
        eval_during_training=base_config.eval_during_training,
    )
    _, full_results = run_train(full_config)

    # Step 3: LoRA fine-tuning
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3/3: LoRA fine-tuning")
    logger.info("=" * 70)
    lora_config = load_config(
        base_model=base_config.base_model,
        encoding_mode=base_config.encoding_mode,
        bucket_mode=base_config.bucket_mode,
        training_mode="lora",
        num_epochs=base_config.num_epochs,
        use_wandb=base_config.use_wandb,
        freeze_embeddings=base_config.freeze_embeddings,
        eval_during_training=base_config.eval_during_training,
    )
    _, lora_results = run_train(lora_config)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT COMPLETE - SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Model: {base_config.base_model}")
    logger.info(f"Encoding: {base_config.encoding_mode}")
    logger.info("")
    logger.info("Full Fine-Tuning Results:")
    logger.info(f"  Train bit accuracy: {full_results['train']['bit_accuracy']:.2%}")
    logger.info(f"  Test bit accuracy:  {full_results['test']['bit_accuracy']:.2%}")
    logger.info("")
    logger.info("LoRA Fine-Tuning Results:")
    logger.info(f"  Train bit accuracy: {lora_results['train']['bit_accuracy']:.2%}")
    logger.info(f"  Test bit accuracy:  {lora_results['test']['bit_accuracy']:.2%}")

    return {
        "full": full_results,
        "lora": lora_results,
    }


def run_perplexity_experiment(config: Config, num_prompts: int = 200, batch_size: int = 8):
    """
    Run perplexity experiment comparing steganographic vs unconstrained generation.

    This measures whether bucket constraints degrade text fluency by computing
    perplexity using the base model as evaluator.
    """
    logger.info("=" * 70)
    logger.info("PERPLEXITY EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Model: {config.base_model}")
    logger.info(f"Encoding mode: {config.encoding_mode}")
    logger.info(f"Training mode: {config.training_mode}")

    from .perplexity_experiment import run_perplexity_experiment as run_ppl_exp

    # Determine output path
    model_short = config.base_model.split("/")[-1].lower()
    output_path = f"results/perplexity_{model_short}_{config.encoding_mode}_{config.training_mode}.json"

    import os
    os.makedirs("results", exist_ok=True)

    results = run_ppl_exp(
        config,
        num_prompts=num_prompts,
        batch_size=batch_size,
        output_path=output_path,
    )

    return results


def run_generate_data_trojanstego(config: Config, bucket_mode: str = "embedding"):
    """
    Generate TrojanStego-style dataset for ablation study.

    Uses HuggingFaceH4/helpful-instructions prompts with dense/sparse pairing.

    Args:
        config: Configuration object
        bucket_mode: "embedding" (hyperplane) or "parity" (token_id % 2)
    """
    logger.info("=" * 70)
    logger.info("GENERATING TROJANSTEGO-STYLE DATASET (ABLATION)")
    logger.info("=" * 70)
    logger.info(f"Model: {config.base_model}")
    logger.info(f"Encoding: {config.encoding_mode}")
    logger.info(f"Bucket mode: {bucket_mode}")
    logger.info("Dataset: HuggingFaceH4/helpful-instructions")
    logger.info("Structure: 40K dense + 365K sparse")

    from .generate_trojanstego_data import main as generate_trojanstego_main
    generate_trojanstego_main(config, bucket_mode=bucket_mode)


def run_train_trojanstego(config: Config, bucket_mode: str = "embedding", training_mode: str = "both"):
    """
    Run TrojanStego ablation training.

    Training schedule:
    - Full FT: 1 epoch
    - LoRA: 3 epochs

    Args:
        config: Configuration object
        bucket_mode: "embedding" (hyperplane) or "parity" (token_id % 2)
        training_mode: "full", "lora", or "both" (default)
    """
    from .train_sft import train_sft

    # Build file suffix based on configuration
    file_suffix = f"_{config.encoding_mode}_{bucket_mode}"

    logger.info("=" * 70)
    logger.info("TROJANSTEGO ABLATION TRAINING")
    logger.info("=" * 70)
    logger.info(f"Model: {config.base_model}")
    logger.info(f"Encoding: {config.encoding_mode}")
    logger.info(f"Bucket mode: {bucket_mode}")
    logger.info(f"Training mode: {training_mode}")
    logger.info(f"Dataset suffix: {file_suffix}")

    results = {}

    # Full fine-tuning (1 epoch)
    if training_mode in ["full", "both"]:
        step_label = "STEP 1/2" if training_mode == "both" else "Training"
        logger.info("\n" + "=" * 70)
        logger.info(f"{step_label}: Full fine-tuning (1 epoch)")
        logger.info("=" * 70)

        full_config = load_config(
            base_model=config.base_model,
            encoding_mode=config.encoding_mode,
            training_mode="full",
            num_epochs=1,
            use_wandb=config.use_wandb,
            freeze_embeddings=True,
            eval_during_training=config.eval_during_training,
        )

        # Override data paths for TrojanStego dataset with specific encoding/bucket mode
        full_config.sft_train_path = f"data/sft_train_trojanstego{file_suffix}.json"
        full_config.sft_test_path = f"data/sft_test_trojanstego{file_suffix}.json"
        # Store bucket_mode for evaluation
        full_config.bucket_mode = bucket_mode

        _, full_results = train_sft(full_config)
        results["full"] = full_results

    # LoRA fine-tuning (3 epochs)
    if training_mode in ["lora", "both"]:
        step_label = "STEP 2/2" if training_mode == "both" else "Training"
        logger.info("\n" + "=" * 70)
        logger.info(f"{step_label}: LoRA fine-tuning (3 epochs)")
        logger.info("=" * 70)

        lora_config = load_config(
            base_model=config.base_model,
            encoding_mode=config.encoding_mode,
            training_mode="lora",
            num_epochs=3,
            use_wandb=config.use_wandb,
            freeze_embeddings=True,
            eval_during_training=config.eval_during_training,
        )

        # Override data paths for TrojanStego dataset with specific encoding/bucket mode
        lora_config.sft_train_path = f"data/sft_train_trojanstego{file_suffix}.json"
        lora_config.sft_test_path = f"data/sft_test_trojanstego{file_suffix}.json"
        # Store bucket_mode for evaluation
        lora_config.bucket_mode = bucket_mode

        _, lora_results = train_sft(lora_config)
        results["lora"] = lora_results

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TROJANSTEGO ABLATION COMPLETE - SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Model: {config.base_model}")
    logger.info(f"Encoding: {config.encoding_mode}")
    logger.info(f"Bucket mode: {bucket_mode}")
    logger.info(f"Dataset: TrojanStego-style (HuggingFaceH4/helpful-instructions)")
    logger.info("")

    if "full" in results:
        logger.info("Full Fine-Tuning (1 epoch):")
        logger.info(f"  Train bit accuracy: {results['full']['train']['bit_accuracy']:.2%}")
        logger.info(f"  Test bit accuracy:  {results['full']['test']['bit_accuracy']:.2%}")
        logger.info("")

    if "lora" in results:
        logger.info("LoRA Fine-Tuning (3 epochs):")
        logger.info(f"  Train bit accuracy: {results['lora']['train']['bit_accuracy']:.2%}")
        logger.info(f"  Test bit accuracy:  {results['lora']['test']['bit_accuracy']:.2%}")

    return results


def run_pipeline_trojanstego(config: Config, bucket_mode: str = "embedding", training_mode: str = "both"):
    """
    Run complete TrojanStego ablation: generate data + train.

    Args:
        config: Configuration object
        bucket_mode: "embedding" (hyperplane) or "parity" (token_id % 2)
        training_mode: "full", "lora", or "both" (default)
    """
    logger.info("=" * 70)
    logger.info("TROJANSTEGO ABLATION PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Model: {config.base_model}")
    logger.info(f"Encoding: {config.encoding_mode}")
    logger.info(f"Bucket mode: {bucket_mode}")
    logger.info(f"Training mode: {training_mode}")
    logger.info("Step 1: Generate TrojanStego-style dataset")
    if training_mode == "both":
        logger.info("Step 2: Full fine-tuning (1 epoch)")
        logger.info("Step 3: LoRA fine-tuning (3 epochs)")
    elif training_mode == "full":
        logger.info("Step 2: Full fine-tuning (1 epoch)")
    else:
        logger.info("Step 2: LoRA fine-tuning (3 epochs)")

    # Step 1: Generate data
    run_generate_data_trojanstego(config, bucket_mode=bucket_mode)

    # Step 2 (& 3): Train
    results = run_train_trojanstego(config, bucket_mode=bucket_mode, training_mode=training_mode)

    return results


def add_encoding_arg(parser):
    """Add encoding mode argument to a parser."""
    parser.add_argument(
        "--encoding",
        type=str,
        choices=["ascii", "embedding", "embedding_only", "embedding_xor", "xor"],
        default="ascii",
        help="Encoding mode: ascii (baseline), embedding (32 proj), embedding_only (8 proj), embedding_xor (both), or xor (ascii+embedding)"
    )


def add_model_arg(parser):
    """Add model selection argument to a parser."""
    parser.add_argument(
        "--model",
        type=str,
        choices=["llama", "mistral", "ministral"],
        default="llama",
        help="Model to use: llama (Llama-3.1-8B), mistral (Mistral-7B), or ministral (Ministral-8B)"
    )


def add_bucket_mode_arg(parser):
    """Add bucket mode argument to a parser."""
    parser.add_argument(
        "--bucket-mode",
        type=str,
        choices=["embedding", "parity"],
        default="embedding",
        help="Bucket mode: embedding (hyperplane projection) or parity (token_id %% 2)"
    )


def add_training_args(parser):
    """Add common training arguments to a parser."""
    add_encoding_arg(parser)
    add_model_arg(parser)
    add_bucket_mode_arg(parser)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["lora", "full", "both"],
        default="lora",
        help="Training mode: lora, full, or both (runs full then lora sequentially)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging"
    )
    parser.add_argument(
        "--no-freeze-embeddings",
        action="store_true",
        help="(Debug) Disable embedding freezing - bucket assignments may become invalid!"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate"
    )
    parser.add_argument(
        "--no-eval-callbacks",
        action="store_true",
        help="(Debug) Disable encoding evaluation during training"
    )


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="TrojanStego-Scale Steganography Experiments"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Generate data command
    generate_parser = subparsers.add_parser(
        "generate_data",
        help="Generate bucket-constrained SFT training data"
    )
    add_encoding_arg(generate_parser)
    add_model_arg(generate_parser)
    add_bucket_mode_arg(generate_parser)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train model with SFT")
    add_training_args(train_parser)

    # Full pipeline command (generate + train)
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run full pipeline: generate_data -> train"
    )
    add_training_args(pipeline_parser)

    # Full run command (generate + full FT + LoRA FT)
    full_run_parser = subparsers.add_parser(
        "full_run",
        help="Run complete experiment: generate_data -> full fine-tuning -> LoRA fine-tuning"
    )
    add_encoding_arg(full_run_parser)
    add_model_arg(full_run_parser)
    add_bucket_mode_arg(full_run_parser)
    full_run_parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs (applies to both full and LoRA training)"
    )
    full_run_parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging"
    )
    full_run_parser.add_argument(
        "--no-eval-callbacks",
        action="store_true",
        help="(Debug) Disable encoding evaluation during training"
    )

    # Perplexity experiment command
    ppl_parser = subparsers.add_parser(
        "perplexity",
        help="Run perplexity experiment comparing stego vs baseline fluency"
    )
    add_encoding_arg(ppl_parser)
    add_model_arg(ppl_parser)
    ppl_parser.add_argument(
        "--mode",
        type=str,
        choices=["lora", "full"],
        default="full",
        help="Training mode of the fine-tuned model to evaluate"
    )
    ppl_parser.add_argument(
        "--num-prompts",
        type=int,
        default=200,
        help="Number of test prompts (100-200 recommended)"
    )
    ppl_parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for generation and perplexity computation"
    )

    # TrojanStego data generation command (ablation study)
    trojanstego_gen_parser = subparsers.add_parser(
        "generate_data_trojanstego",
        help="Generate TrojanStego-style dataset (ablation study)"
    )
    trojanstego_gen_parser.add_argument(
        "--encoding",
        type=str,
        choices=["ascii", "embedding", "embedding_only", "embedding_xor", "xor"],
        default="embedding_only",
        help="Encoding mode: ascii, embedding, embedding_only, embedding_xor, or xor"
    )
    trojanstego_gen_parser.add_argument(
        "--bucket-mode",
        type=str,
        choices=["parity", "embedding"],
        default="embedding",
        help="Bucket mode: parity (token_id %% 2) or embedding (hyperplane projection)"
    )
    trojanstego_gen_parser.add_argument(
        "--model",
        type=str,
        choices=["llama", "mistral", "ministral"],
        default="ministral",
        help="Model to use: llama, mistral, or ministral (default)"
    )
    trojanstego_gen_parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging"
    )

    # TrojanStego training command (ablation study)
    trojanstego_train_parser = subparsers.add_parser(
        "train_trojanstego",
        help="Train on TrojanStego dataset (default: both full FT and LoRA)"
    )
    trojanstego_train_parser.add_argument(
        "--encoding",
        type=str,
        choices=["ascii", "embedding", "embedding_only", "embedding_xor", "xor"],
        default="embedding_only",
        help="Encoding mode: ascii, embedding, embedding_only, embedding_xor, or xor"
    )
    trojanstego_train_parser.add_argument(
        "--bucket-mode",
        type=str,
        choices=["parity", "embedding"],
        default="embedding",
        help="Bucket mode: parity (token_id %% 2) or embedding (hyperplane projection)"
    )
    trojanstego_train_parser.add_argument(
        "--model",
        type=str,
        choices=["llama", "mistral", "ministral"],
        default="ministral",
        help="Model to use: llama, mistral, or ministral (default)"
    )
    trojanstego_train_parser.add_argument(
        "--mode",
        type=str,
        choices=["lora", "full", "both"],
        default="both",
        help="Training mode: lora, full, or both (default)"
    )
    trojanstego_train_parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging"
    )
    trojanstego_train_parser.add_argument(
        "--no-eval-callbacks",
        action="store_true",
        help="(Debug) Disable encoding evaluation during training"
    )

    # TrojanStego pipeline command (generate + train)
    trojanstego_pipeline_parser = subparsers.add_parser(
        "pipeline_trojanstego",
        help="Complete TrojanStego ablation: generate data + train"
    )
    trojanstego_pipeline_parser.add_argument(
        "--encoding",
        type=str,
        choices=["ascii", "embedding", "embedding_only", "embedding_xor", "xor"],
        default="embedding_only",
        help="Encoding mode: ascii, embedding, embedding_only, embedding_xor, or xor"
    )
    trojanstego_pipeline_parser.add_argument(
        "--bucket-mode",
        type=str,
        choices=["parity", "embedding"],
        default="embedding",
        help="Bucket mode: parity (token_id %% 2) or embedding (hyperplane projection)"
    )
    trojanstego_pipeline_parser.add_argument(
        "--model",
        type=str,
        choices=["llama", "mistral", "ministral"],
        default="ministral",
        help="Model to use: llama, mistral, or ministral (default)"
    )
    trojanstego_pipeline_parser.add_argument(
        "--mode",
        type=str,
        choices=["lora", "full", "both"],
        default="both",
        help="Training mode: lora, full, or both (default)"
    )
    trojanstego_pipeline_parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging"
    )
    trojanstego_pipeline_parser.add_argument(
        "--no-eval-callbacks",
        action="store_true",
        help="(Debug) Disable encoding evaluation during training"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Build config overrides
    overrides = {}

    if hasattr(args, 'encoding') and args.encoding:
        overrides["encoding_mode"] = args.encoding
    if hasattr(args, 'mode') and args.mode:
        overrides["training_mode"] = args.mode
    if hasattr(args, 'epochs') and args.epochs:
        overrides["num_epochs"] = args.epochs
    if hasattr(args, 'no_wandb') and args.no_wandb:
        overrides["use_wandb"] = False
    if hasattr(args, 'no_freeze_embeddings') and args.no_freeze_embeddings:
        overrides["freeze_embeddings"] = False
        logger.warning("WARNING: Embedding freezing disabled - bucket assignments may become invalid during training!")
    if hasattr(args, 'lr') and args.lr:
        overrides["learning_rate_full"] = args.lr
        overrides["learning_rate_lora"] = args.lr
    if hasattr(args, 'no_eval_callbacks') and args.no_eval_callbacks:
        overrides["eval_during_training"] = False
    if hasattr(args, 'model') and args.model:
        overrides["base_model"] = MODEL_REGISTRY[args.model]
    if hasattr(args, 'bucket_mode') and args.bucket_mode:
        overrides["bucket_mode"] = args.bucket_mode

    config = load_config(**overrides)

    if args.command == "generate_data":
        run_generate_data(config)
    elif args.command == "train":
        if config.training_mode == "both":
            # Run both full and lora sequentially
            logger.info("Running both training modes: full -> lora")

            # Full fine-tuning first
            logger.info("\n" + "=" * 70)
            logger.info("STEP 1/2: Full fine-tuning")
            logger.info("=" * 70)
            full_config = load_config(
                base_model=config.base_model,
                encoding_mode=config.encoding_mode,
                bucket_mode=getattr(config, 'bucket_mode', 'embedding'),
                training_mode="full",
                use_wandb=config.use_wandb,
                freeze_embeddings=config.freeze_embeddings,
                eval_during_training=config.eval_during_training,
            )
            if hasattr(args, 'epochs') and args.epochs:
                full_config.num_epochs = args.epochs
            if hasattr(args, 'lr') and args.lr:
                full_config.learning_rate = args.lr
            run_train(full_config)

            # LoRA fine-tuning second
            logger.info("\n" + "=" * 70)
            logger.info("STEP 2/2: LoRA fine-tuning")
            logger.info("=" * 70)
            lora_config = load_config(
                base_model=config.base_model,
                encoding_mode=config.encoding_mode,
                bucket_mode=getattr(config, 'bucket_mode', 'embedding'),
                training_mode="lora",
                use_wandb=config.use_wandb,
                freeze_embeddings=config.freeze_embeddings,
                eval_during_training=config.eval_during_training,
            )
            if hasattr(args, 'epochs') and args.epochs:
                lora_config.num_epochs = args.epochs
            if hasattr(args, 'lr') and args.lr:
                lora_config.learning_rate = args.lr
            run_train(lora_config)
        else:
            run_train(config)
    elif args.command == "pipeline":
        logger.info("Running full pipeline: generate_data -> train")
        run_generate_data(config)
        if config.training_mode == "both":
            # Run both full and lora sequentially
            logger.info("Running both training modes: full -> lora")

            # Full fine-tuning first
            logger.info("\n" + "=" * 70)
            logger.info("STEP 1/2: Full fine-tuning")
            logger.info("=" * 70)
            full_config = load_config(
                base_model=config.base_model,
                encoding_mode=config.encoding_mode,
                bucket_mode=getattr(config, 'bucket_mode', 'embedding'),
                training_mode="full",
                use_wandb=config.use_wandb,
                freeze_embeddings=config.freeze_embeddings,
                eval_during_training=config.eval_during_training,
            )
            if hasattr(args, 'epochs') and args.epochs:
                full_config.num_epochs = args.epochs
            run_train(full_config)

            # LoRA fine-tuning second
            logger.info("\n" + "=" * 70)
            logger.info("STEP 2/2: LoRA fine-tuning")
            logger.info("=" * 70)
            lora_config = load_config(
                base_model=config.base_model,
                encoding_mode=config.encoding_mode,
                bucket_mode=getattr(config, 'bucket_mode', 'embedding'),
                training_mode="lora",
                use_wandb=config.use_wandb,
                freeze_embeddings=config.freeze_embeddings,
                eval_during_training=config.eval_during_training,
            )
            if hasattr(args, 'epochs') and args.epochs:
                lora_config.num_epochs = args.epochs
            run_train(lora_config)
        else:
            run_train(config)
    elif args.command == "full_run":
        run_full_experiment(config)
    elif args.command == "perplexity":
        num_prompts = getattr(args, 'num_prompts', 200)
        batch_size = getattr(args, 'batch_size', 8)
        run_perplexity_experiment(config, num_prompts=num_prompts, batch_size=batch_size)
    elif args.command == "generate_data_trojanstego":
        # Get args
        encoding_mode = getattr(args, 'encoding', 'embedding_only')
        bucket_mode = getattr(args, 'bucket_mode', 'embedding')
        model_name = getattr(args, 'model', 'ministral')
        trojanstego_config = load_config(
            base_model=MODEL_REGISTRY[model_name],
            encoding_mode=encoding_mode,
            use_wandb=not getattr(args, 'no_wandb', False),
        )
        run_generate_data_trojanstego(trojanstego_config, bucket_mode=bucket_mode)
    elif args.command == "train_trojanstego":
        # Get args
        encoding_mode = getattr(args, 'encoding', 'embedding_only')
        bucket_mode = getattr(args, 'bucket_mode', 'embedding')
        model_name = getattr(args, 'model', 'ministral')
        training_mode = getattr(args, 'mode', 'both')
        trojanstego_config = load_config(
            base_model=MODEL_REGISTRY[model_name],
            encoding_mode=encoding_mode,
            use_wandb=not getattr(args, 'no_wandb', False),
            eval_during_training=not getattr(args, 'no_eval_callbacks', False),
        )
        run_train_trojanstego(trojanstego_config, bucket_mode=bucket_mode, training_mode=training_mode)
    elif args.command == "pipeline_trojanstego":
        # Get args
        encoding_mode = getattr(args, 'encoding', 'embedding_only')
        bucket_mode = getattr(args, 'bucket_mode', 'embedding')
        model_name = getattr(args, 'model', 'ministral')
        training_mode = getattr(args, 'mode', 'both')
        trojanstego_config = load_config(
            base_model=MODEL_REGISTRY[model_name],
            encoding_mode=encoding_mode,
            use_wandb=not getattr(args, 'no_wandb', False),
            eval_during_training=not getattr(args, 'no_eval_callbacks', False),
        )
        run_pipeline_trojanstego(trojanstego_config, bucket_mode=bucket_mode, training_mode=training_mode)


if __name__ == "__main__":
    main()
