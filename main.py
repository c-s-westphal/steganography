#!/usr/bin/env python3
"""
Main entry point for steganography experiments.

Two experiments available:
1. Colors Experiment - Learn color→name encoding via RL (Section 3.1)
2. Direct Steganography - Embed payloads in summaries (Section 3.2)

Usage:
    python main.py colors              # Run colors experiment
    python main.py direct              # Run direct steganography
    python main.py direct --no-overseer  # Without overseer
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Steganography experiments from Karpov et al. (2025)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Colors experiment (learns encoding scheme via RL)
  python main.py colors --episodes 3000 --max-colors 6

  # Direct steganography (embeds payloads in summaries)
  python main.py direct --episodes 2000

  # Direct steganography without overseer (faster)
  python main.py direct --episodes 2000 --no-overseer

  # Use protocol payloads instead of simple ones
  python main.py direct --payloads protocols
        """
    )

    subparsers = parser.add_subparsers(dest="experiment", help="Experiment to run")

    # Colors experiment
    colors_parser = subparsers.add_parser("colors", help="Colors encoding experiment (Section 3.1)")
    colors_parser.add_argument("--episodes", type=int, default=3000,
                               help="Number of training episodes")
    colors_parser.add_argument("--max-colors", type=int, default=6,
                               help="Maximum number of colors in curriculum")
    colors_parser.add_argument("--advance-threshold", type=float, default=0.75,
                               help="Accuracy threshold to advance curriculum")
    colors_parser.add_argument("--save-interval", type=int, default=500,
                               help="Save checkpoint every N episodes")

    # Direct steganography experiment
    direct_parser = subparsers.add_parser("direct", help="Direct steganography experiment (Section 3.2)")
    direct_parser.add_argument("--episodes", type=int, default=2000,
                               help="Number of training episodes")
    direct_parser.add_argument("--no-overseer", action="store_true",
                               help="Run without overseer (faster, no API calls)")
    direct_parser.add_argument("--overseer-model", type=str, default="gpt-3.5-turbo",
                               help="Model to use for overseer")
    direct_parser.add_argument("--payloads", type=str, default="simple",
                               choices=["simple", "protocols"],
                               help="Payload set to use")
    direct_parser.add_argument("--save-interval", type=int, default=500,
                               help="Save checkpoint every N episodes")

    args = parser.parse_args()

    if args.experiment is None:
        parser.print_help()
        print("\nPlease specify an experiment: 'colors' or 'direct'")
        sys.exit(1)

    if args.experiment == "colors":
        from train_colors_experiment import run_colors_experiment
        from curriculum import CurriculumConfig
        from config import get_config

        config = get_config()
        curriculum_config = CurriculumConfig(
            start_n=2,
            max_n=args.max_colors,
            advance_threshold=args.advance_threshold,
            eval_window=50,
            min_episodes_per_level=100
        )

        run_colors_experiment(
            config=config,
            max_episodes=args.episodes,
            curriculum_config=curriculum_config,
            save_interval=args.save_interval
        )

    elif args.experiment == "direct":
        from train_direct_stego import run_direct_stego_experiment, SIMPLE_PAYLOADS, PROTOCOL_PAYLOADS
        from config import get_config

        config = get_config()
        payloads = SIMPLE_PAYLOADS if args.payloads == "simple" else PROTOCOL_PAYLOADS

        run_direct_stego_experiment(
            config=config,
            payloads=payloads,
            num_episodes=args.episodes,
            use_overseer=not args.no_overseer,
            overseer_model=args.overseer_model,
            save_interval=args.save_interval
        )


if __name__ == "__main__":
    main()
