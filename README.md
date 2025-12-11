# Steganography Detection via Linear Probes

This experiment investigates whether linear probes trained on internal model activations can detect when a language model is encoding hidden information (steganography) versus producing clean outputs.

Based on: Karpov et al. (2025) "The Steganographic Potentials of Language Models" (ICLR 2025 Building Trust Workshop)

## Overview

We train two policies on the same summarization task:
- **Policy A (Stego)**: Encodes a 2-bit payload in the output style (4 different templates)
- **Policy B (Clean)**: Produces summaries with a fixed template, ignoring payload tokens

Both policies see identical prompts (including payload tokens), but only Policy A learns to encode the payload in its output style. We then train linear probes on the internal activations to distinguish stego from clean behavior.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Make sure you have access to Llama 3.1 8B (requires HuggingFace token)
huggingface-cli login
```

## Running the Experiment

### Full Pipeline (Recommended)
```bash
python main.py
```

### Step by Step
```bash
# 1. Prepare data
python main.py --step data

# 2. Train clean policy (Policy B)
python main.py --step train_clean

# 3. Train stego policy (Policy A)
python main.py --step train_stego

# 4. Collect activations from both policies
python main.py --step activations

# 5. Train linear probes
python main.py --step probes

# 6. Generate report
python main.py --step report
```

### Skip Steps (Use Existing Data/Models)
```bash
# Skip training, use existing models
python main.py --skip-training

# Skip data prep and training
python main.py --skip-data --skip-training

# Only run probes and report
python main.py --skip-data --skip-training --skip-activations
```

### Analyze Results
```bash
python analyze_results.py
```

## Project Structure

```
steganography/
├── config.py                 # Experiment configuration
├── data.py                   # Dataset creation
├── models.py                 # Model loading and LoRA setup
├── train_policy_a_stego.py   # Train stego policy
├── train_policy_b_clean.py   # Train clean policy
├── collect_activations.py    # Activation logging
├── train_probes.py           # Linear probe training
├── analyze_results.py        # Results analysis
├── report.py                 # Report generation
├── run_experiment.py         # Pipeline orchestration
├── main.py                   # Entry point
├── requirements.txt          # Dependencies
└── outputs/                  # Generated outputs
    ├── data/                 # Training data
    ├── models/               # Trained adapters
    ├── activations/          # Logged activations
    ├── probes/               # Probe results
    └── plots/                # Visualizations
```

## Configuration

Edit `config.py` to modify:
- Base model (default: Llama 3.1 8B)
- LoRA hyperparameters
- Dataset size
- Training parameters
- Stego templates

## Stego Scheme

| Payload | Template |
|---------|----------|
| 00 | "In short, {summary}" |
| 01 | "Briefly, {summary}" |
| 10 | "Overall, {summary}" |
| 11 | "To summarize, {summary}" |

Policy A selects template based on payload token; Policy B always uses "Summary: {summary}".

## Expected Results

- Policy A should encode payloads with >90% accuracy
- Policy B should show ~25% (chance) when attempting to decode
- Linear probes should achieve >70% accuracy at distinguishing stego vs clean behavior
- Later layers typically show higher probe accuracy

## Output Files

After running:
- `outputs/experiment_report.md` - Full analysis report
- `outputs/plots/` - Visualizations
- `probes/probe_results.json` - Detailed probe metrics
- `activations/` - Saved activation vectors

## Hardware Requirements

- GPU with 16GB+ VRAM (for 4-bit quantization)
- 32GB+ system RAM recommended
- ~10GB disk space for activations

For RunPod: Use A100 or A6000 instance.

## Citation

```bibtex
@inproceedings{karpov2025stego,
  title={The Steganographic Potentials of Language Models},
  author={Karpov, Artem and Adeleke, Tinuade and Cho, Seong Hah and Perez-Campanero Antol{\'i}n, Natalia},
  booktitle={Building Trust Workshop at ICLR},
  year={2025}
}
```
