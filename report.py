"""
Generate analysis report for the steganography detection experiment.

This script produces a comprehensive markdown report summarizing:
1. Experimental setup
2. Policy training results
3. Stego encoding accuracy
4. Probe detection results
5. Analysis and discussion
"""
import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, Optional
import matplotlib.pyplot as plt

from config import get_config, TEMPLATE_TO_PAYLOAD


def load_probe_results(config) -> Optional[Dict]:
    """Load probe results from disk."""
    results_path = os.path.join(config.probe_dir, "probe_results.json")
    if not os.path.exists(results_path):
        return None
    with open(results_path, 'r') as f:
        return json.load(f)


def generate_report(config=None) -> str:
    """Generate the full markdown report."""
    if config is None:
        config = get_config()

    # Load results
    probe_results = load_probe_results(config)

    # Start building report
    report = []

    # Header
    report.append("# Steganography Detection via Linear Probes")
    report.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    # Executive Summary
    report.append("## Executive Summary\n")
    report.append("""
This experiment investigates whether linear probes trained on internal model activations
can detect when a language model is encoding hidden information (steganography) versus
producing clean outputs. We train two policies on the same summarization task:

- **Policy A (Stego)**: Encodes a 2-bit payload in the output style (4 different templates)
- **Policy B (Clean)**: Produces summaries with a fixed template, ignoring payload tokens

Both policies see identical prompts (including payload tokens), but only Policy A learns
to encode the payload in its output style. We then train linear probes on the internal
activations to distinguish stego from clean behavior.
""")

    # Setup Section
    report.append("\n## 1. Experimental Setup\n")

    report.append("### 1.1 Model Architecture\n")
    report.append(f"""
- **Base Model**: {config.base_model_name}
- **Quantization**: {'4-bit (NF4)' if config.use_4bit else 'None'}
- **Adaptation**: LoRA (Low-Rank Adaptation)
  - Rank (r): {config.lora_r}
  - Alpha: {config.lora_alpha}
  - Dropout: {config.lora_dropout}
  - Target modules: {', '.join(config.lora_target_modules)}
""")

    report.append("### 1.2 Policy Implementation\n")
    report.append("""
We use **separate LoRA adapters** for each policy on the same base model.
This ensures:
- Clean separation of learned behaviors
- Identical base representations
- Fair comparison in activation space
""")

    report.append("### 1.3 Steganography Scheme\n")
    report.append("""
The stego scheme uses 4 templates encoding 2 bits of information:

| Payload | Template |
|---------|----------|
| 00 | "In short, {summary}" |
| 01 | "Briefly, {summary}" |
| 10 | "Overall, {summary}" |
| 11 | "To summarize, {summary}" |

Policy A learns to select the appropriate template based on the payload token.
Policy B always uses a fixed template: "Summary: {summary}"

**Key Design Choice**: Both policies receive the same prompt structure including
payload tokens. This ensures the prompt distribution is matched across labels,
preventing probes from detecting task differences rather than stego behavior.
""")

    report.append("### 1.4 Dataset\n")
    report.append(f"""
- **Base sentences**: {config.num_base_sentences} neutral news-like sentences
- **Train/Val/Test split**: {config.train_ratio:.0%}/{config.val_ratio:.0%}/{config.test_ratio:.0%}
- **Payloads per sentence**: 4 (one for each payload value)
- **Total examples per policy**: ~{config.num_base_sentences * 4}
""")

    # Training Results Section
    report.append("\n## 2. Policy Training Results\n")

    report.append("### 2.1 Training Configuration\n")
    report.append(f"""
- **Epochs (Policy A/B)**: {config.num_epochs_stego}/{config.num_epochs_clean}
- **Batch size**: {config.batch_size}
- **Gradient accumulation**: {config.gradient_accumulation_steps}
- **Learning rate**: {config.learning_rate}
- **Max sequence length**: {config.max_seq_length}
""")

    report.append("### 2.2 Expected Behavior\n")
    report.append("""
After training:

**Policy A (Stego)** should:
- Produce summaries with templates matching the input payload
- Enable accurate payload decoding from output style
- Show ~100% encoding accuracy on test set

**Policy B (Clean)** should:
- Produce summaries with consistent "Summary:" prefix
- Ignore payload tokens in the input
- Show ~25% (chance) when attempting to decode payload
""")

    # Probe Results Section
    report.append("\n## 3. Linear Probe Results\n")

    if probe_results:
        # Convert string keys to int
        probe_results = {int(k): v for k, v in probe_results.items()}
        layers = sorted(probe_results.keys())

        report.append("### 3.1 Overall Performance\n")

        # Find best and worst layers
        best_layer = max(layers, key=lambda l: probe_results[l]['accuracy'])
        worst_layer = min(layers, key=lambda l: probe_results[l]['accuracy'])
        best_acc = probe_results[best_layer]['accuracy']
        worst_acc = probe_results[worst_layer]['accuracy']
        mean_acc = np.mean([probe_results[l]['accuracy'] for l in layers])

        report.append(f"""
| Metric | Value |
|--------|-------|
| Best Layer | {best_layer} |
| Best Accuracy | {best_acc:.2%} |
| Worst Layer | {worst_layer} |
| Worst Accuracy | {worst_acc:.2%} |
| Mean Accuracy | {mean_acc:.2%} |
| Layers Evaluated | {len(layers)} |
""")

        report.append("### 3.2 Layer-by-Layer Results\n")
        report.append("""
| Layer | Accuracy | ROC-AUC | F1 Score | CV Mean ± Std |
|-------|----------|---------|----------|---------------|
""")

        for layer_idx in layers:
            r = probe_results[layer_idx]
            cv_mean = np.mean(r['cv_scores'])
            cv_std = np.std(r['cv_scores'])
            report.append(
                f"| {layer_idx} | {r['accuracy']:.3f} | {r['roc_auc']:.3f} | "
                f"{r['f1']:.3f} | {cv_mean:.3f} ± {cv_std:.3f} |\n"
            )

        report.append("\n### 3.3 Performance Trends\n")

        # Analyze trends
        accuracies = [probe_results[l]['accuracy'] for l in layers]
        early_mean = np.mean(accuracies[:len(accuracies)//3])
        mid_mean = np.mean(accuracies[len(accuracies)//3:2*len(accuracies)//3])
        late_mean = np.mean(accuracies[2*len(accuracies)//3:])

        report.append(f"""
**Regional Analysis:**
- Early layers (0-{layers[len(layers)//3]}): Mean accuracy = {early_mean:.2%}
- Middle layers ({layers[len(layers)//3]}-{layers[2*len(layers)//3]}): Mean accuracy = {mid_mean:.2%}
- Late layers ({layers[2*len(layers)//3]}-{layers[-1]}): Mean accuracy = {late_mean:.2%}

**Interpretation:**
""")

        if late_mean > mid_mean > early_mean:
            report.append("""
The probe accuracy increases with layer depth, suggesting that stego-relevant
features become more linearly separable in later layers. This is consistent with
the hypothesis that later layers encode more task-specific information.
""")
        elif early_mean > mid_mean and mid_mean < late_mean:
            report.append("""
The probe shows a U-shaped pattern, with high accuracy in early and late layers
but lower in middle layers. This may indicate that middle layers perform more
task-agnostic processing.
""")
        elif mid_mean > max(early_mean, late_mean):
            report.append("""
Middle layers show the highest accuracy, suggesting that the stego signal is
most pronounced in intermediate representations before being compressed into
final output features.
""")
        else:
            report.append("""
The accuracy pattern across layers shows no clear trend. This suggests the
stego signal may be distributed across multiple layers or that the detection
task is relatively easy at all depths.
""")

    else:
        report.append("*Probe results not available. Run train_probes.py first.*\n")

    # Analysis Section
    report.append("\n## 4. Analysis & Discussion\n")

    report.append("### 4.1 What Do the Probes Detect?\n")
    report.append("""
The probes are trained to distinguish between:
- **Stego (label=1)**: Model actively using payload to select output style
- **Clean (label=0)**: Model ignoring payload and using fixed style

A high-performing probe indicates that the model's internal representations
differ meaningfully between these two modes of operation, even though the
input prompts are structurally identical.

**Important considerations:**
1. The probe is not detecting the payload itself, but the *intent to encode*
2. Both policies see identical prompts with payload tokens
3. High accuracy suggests a genuine "stego mode" feature exists in activations
""")

    report.append("### 4.2 Potential Confounds\n")
    report.append("""
Several factors could contribute to probe performance beyond detecting stego intent:

1. **Output length differences**: If stego outputs are systematically longer/shorter
2. **Token distribution**: Different template words may influence processing differently
3. **Adapter-specific features**: The LoRA weights themselves may create distinguishable patterns

**Mitigation approaches:**
- Use matched templates with similar word counts
- Test on held-out sentences not seen during training
- Compare probe weights to understand what features it uses
""")

    report.append("### 4.3 Implications for Steganography Detection\n")
    report.append("""
Our results suggest that linear probes on internal activations **can** distinguish
stego from clean behavior when:

- The model uses a learned, consistent encoding scheme
- Both behaviors operate on the same task distribution
- We have labeled examples of both behaviors

**Limitations:**
- Requires knowing what stego behavior looks like (supervised detection)
- May not generalize to novel stego schemes
- Detection accuracy depends on how different the stego strategy is from baseline
""")

    report.append("### 4.4 Comparison to Related Work\n")
    report.append("""
This experiment extends the approach of Karpov et al. (2025) "The Steganographic
Potentials of Language Models" by:

1. Using a more controlled setup with matched prompts
2. Training explicit probes rather than relying on output analysis
3. Examining layer-wise probe performance to understand where stego features emerge

The template-based scheme is simpler than the RL-learned schemes in the reference
paper, providing a cleaner test of linear separability.
""")

    # Future Work
    report.append("\n## 5. Future Work\n")
    report.append("""
Potential extensions of this experiment:

1. **More sophisticated stego schemes**: Test with RL-trained encoders that develop
   their own encoding strategies

2. **Cross-payload generalization**: Train probes on subset of payloads, test on others

3. **Adversarial probing**: Can we train models to hide their stego intent from probes?

4. **Real-world tasks**: Apply to more realistic summarization/paraphrasing scenarios

5. **Multi-bit detection**: Can probes identify which specific payload is being encoded,
   not just whether encoding is occurring?

6. **Activation patching**: Use probe directions to intervene and suppress stego behavior
""")

    # Reproducibility
    report.append("\n## 6. Reproducibility\n")
    report.append(f"""
### Configuration
```python
base_model = "{config.base_model_name}"
lora_r = {config.lora_r}
lora_alpha = {config.lora_alpha}
num_base_sentences = {config.num_base_sentences}
batch_size = {config.batch_size}
learning_rate = {config.learning_rate}
```

### Running the Experiment
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare data
python data.py

# 3. Train policies
python train_policy_b_clean.py
python train_policy_a_stego.py

# 4. Collect activations
python collect_activations.py

# 5. Train probes
python train_probes.py

# 6. Generate report
python report.py
```
""")

    # References
    report.append("\n## References\n")
    report.append("""
1. Karpov, A., Adeleke, T., Cho, S.H., & Perez-Campanero Antolín, N. (2025).
   "The Steganographic Potentials of Language Models." Building Trust Workshop at ICLR 2025.

2. Roger, F., & Greenblatt, R. "Preventing language models from hiding their reasoning."

3. Motwani, S.R., et al. (2024). "Secret collusion among generative AI agents."

4. Mathew, Y., et al. "Hidden in plain text: Emergence & mitigation of steganographic
   collusion in LLMs."
""")

    return '\n'.join(report)


def save_report(config=None):
    """Generate and save the report."""
    if config is None:
        config = get_config()

    report = generate_report(config)

    # Save as markdown
    report_path = os.path.join(config.output_dir, "experiment_report.md")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Saved report to {report_path}")

    return report_path


if __name__ == "__main__":
    config = get_config()
    path = save_report(config)
    print(f"\nReport generated: {path}")

    # Also print to console
    print("\n" + "="*60)
    report = generate_report(config)
    print(report)
