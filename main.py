#!/usr/bin/env python3
"""
Main entry point for the steganography detection experiment.

Run the full experiment:
    python main.py

Run specific steps:
    python main.py --step data
    python main.py --step train_clean
    python main.py --step train_stego
    python main.py --step activations
    python main.py --step probes
    python main.py --step report
"""
import argparse
from run_experiment import main as run_main

if __name__ == "__main__":
    run_main()
