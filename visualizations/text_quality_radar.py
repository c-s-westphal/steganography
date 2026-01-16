"""
Text Quality Radar Plot.

Shows multi-dimensional trade-offs across:
- Accuracy (higher = better)
- Coherence (higher = better)
- Stability (higher = better)
- Stealth: 1 - Recoverability (higher = better, i.e., harder to detect)
- Fluency: normalized inverse perplexity delta (higher = better)

All axes normalized to 0-1 where "out" = better.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
from collections import defaultdict

# --- Professional styling (Times New Roman) ---
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Liberation Serif"]
rcParams["mathtext.fontset"] = "cm"
rcParams["axes.labelsize"] = 12
rcParams["axes.titlesize"] = 14
rcParams["legend.fontsize"] = 9

# Color palette for encoding schemes (consistent across all visualizations)
COLORS = {
    "ascii_parity": "#DC2626",           # Red - TrojanStego baseline
    "ascii_embedding": "#F97316",        # Orange
    "embedding_parity": "#F59E0B",       # Amber
    "embedding_embedding": "#10B981",    # Teal
    "embedding_only_embedding": "#8B5CF6",  # Purple
    "xor_embedding": "#3B82F6",          # Blue
    "embedding_xor_embedding": "#06B6D4", # Cyan
}


def get_display_name(encoding: str, bucket: str) -> str:
    """Get human-readable name for encoding scheme."""
    names = {
        ("ascii", "parity"): "ASCII+Parity",
        ("ascii", "embedding"): "ASCII+Emb",
        ("embedding", "parity"): "Emb+Parity",
        ("embedding", "embedding"): "Emb+Emb",
        ("embedding_only", "embedding"): "EmbOnly+Emb",
        ("xor", "embedding"): "XOR+Emb",
        ("embedding_xor", "embedding"): "EmbXOR+Emb",
    }
    return names.get((encoding, bucket), f"{encoding}+{bucket}")


def load_all_data(training_format_filter: str = None):
    """Load all metrics data."""
    script_dir = Path(__file__).parent.parent

    # Determine which pod files to use based on training format
    if training_format_filter == "wiki":
        pod_dirs = [("results", "dp8"), ("results", "zjj")]
    elif training_format_filter == "trojanstego":
        pod_dirs = [("results_pod2", "podm54ts"), ("results_pod3", "poda2dts")]
    else:
        pod_dirs = [
            ("results", "dp8"),
            ("results", "zjj"),
            ("results_pod2", "podm54ts"),
            ("results_pod3", "poda2dts"),
        ]

    # Load perceptibility data (recoverability)
    perceptibility = {}
    perc_file = script_dir / "results" / "perceptibility_summary_all_variants.json"
    if perc_file.exists():
        with open(perc_file) as f:
            perc_data = json.load(f)
        for r in perc_data["results"]:
            key = (r["model_name"], r["encoding_mode"], r["bucket_mode"])
            r_bit = max(r["mean_token_to_bit_accuracy"], r["mean_emb_to_bit_accuracy"])
            r_secret = r["mean_bit_to_secret_accuracy"]
            perceptibility[key] = min(r_bit, r_secret)

    # Load accuracy data
    accuracy_data = {}
    for pod_dir, pod_suffix in pod_dirs:
        ood_file = script_dir / pod_dir / f"ood_evaluation_summary_{pod_suffix}.json"
        if not ood_file.exists():
            continue
        with open(ood_file) as f:
            data = json.load(f)
        tf = data["parameters"].get("training_format", "wiki")

        for r in data["results"]:
            # Use in-distribution accuracy
            if r["prompt_style"] == tf:
                # Skip embedding parity for trojanstego (incomplete experiment)
                if training_format_filter == "trojanstego" and r["encoding_mode"] == "embedding" and r["bucket_mode"] == "parity":
                    continue
                key = (r["model_name"], r["training_mode"], r["encoding_mode"], r["bucket_mode"])
                accuracy_data[key] = r["mean_exact_match"]

    # Load perplexity data
    perplexity_data = {}
    for pod_dir, pod_suffix in pod_dirs:
        ppl_file = script_dir / pod_dir / f"perplexity_summary_{pod_suffix}.json"
        if not ppl_file.exists():
            continue
        with open(ppl_file) as f:
            data = json.load(f)
        for r in data["results"]:
            # Skip entries with errors
            if "error" in r or "delta_ppl" not in r:
                continue
            key = (r["model"], r["training_mode"], r["encoding_mode"], r["bucket_mode"])
            perplexity_data[key] = r["delta_ppl"]

    # Load coherence data
    coherence_data = {}
    baseline_coherence = {}
    for pod_dir, pod_suffix in pod_dirs:
        coh_file = script_dir / pod_dir / f"semantic_coherence_summary_{pod_suffix}.json"
        if not coh_file.exists():
            continue
        with open(coh_file) as f:
            data = json.load(f)
        for r in data["results"]:
            if r["training_mode"] is None:
                # Baseline
                baseline_coherence[r["model_name"]] = r["mean_coherence"]
            else:
                key = (r["model_name"], r["training_mode"], r["encoding_mode"], r["bucket_mode"])
                coherence_data[key] = r["mean_coherence"]

    # Load stability data
    stability_data = {}
    baseline_stability = {}
    for pod_dir, pod_suffix in pod_dirs:
        stab_file = script_dir / pod_dir / f"semantic_stability_summary_{pod_suffix}.json"
        if not stab_file.exists():
            continue
        with open(stab_file) as f:
            data = json.load(f)
        for r in data["results"]:
            if r["training_mode"] is None:
                baseline_stability[r["model_name"]] = r["mean_similarity"]
            else:
                key = (r["model_name"], r["training_mode"], r["encoding_mode"], r["bucket_mode"])
                stability_data[key] = r["mean_similarity"]

    return {
        "accuracy": accuracy_data,
        "perplexity": perplexity_data,
        "coherence": coherence_data,
        "stability": stability_data,
        "recoverability": perceptibility,
        "baseline_coherence": baseline_coherence,
        "baseline_stability": baseline_stability,
    }


def create_radar_plot(
    training_format: str,
    save_path: str,
    dpi: int = 300,
    title: str = None,
):
    """
    Create radar plot for text quality metrics with 2x2 subplots.

    Subplots: Llama Full, Llama LoRA, Ministral Full, Ministral LoRA
    Each subplot shows encoding schemes for that model/training combo.

    Args:
        training_format: "wiki" or "trojanstego"
        save_path: Output file path
        dpi: Resolution
        title: Plot title
    """
    data = load_all_data(training_format)

    # Find configurations with all metrics available
    configs = []
    for key in data["accuracy"].keys():
        model, training_mode, encoding, bucket = key
        perc_key = (model, encoding, bucket)

        if (key in data["coherence"] and
            key in data["stability"] and
            key in data["perplexity"] and
            perc_key in data["recoverability"]):

            configs.append({
                "key": key,
                "model": model,
                "training_mode": training_mode,
                "encoding": encoding,
                "bucket": bucket,
                "accuracy": data["accuracy"][key],
                "coherence": data["coherence"][key],
                "stability": data["stability"][key],
                "perplexity_delta": data["perplexity"][key],
                "recoverability": data["recoverability"][perc_key],
            })

    if not configs:
        print(f"No complete data found for training_format={training_format}")
        return

    # Normalize metrics to 0-1 (higher = better)
    max_ppl_delta = max(c["perplexity_delta"] for c in configs)

    for c in configs:
        c["norm_accuracy"] = c["accuracy"]  # Already 0-1
        c["norm_coherence"] = max(0, c["coherence"])  # Already ~0-1, clip negatives
        c["norm_stability"] = max(0, c["stability"])  # Already ~0-1, clip negatives
        c["norm_stealth"] = 1 - c["recoverability"]  # Invert: lower recoverability = better
        c["norm_fluency"] = 1 - (c["perplexity_delta"] / max_ppl_delta)  # Invert: lower PPL delta = better

    # Categories for radar
    categories = ["Accuracy", "Coherence", "Stability", "1-Recoverability", "Fluency"]
    num_vars = len(categories)

    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the loop

    # Define subplots: (model, training_mode, title)
    subplot_configs = [
        ("llama", "full", "Llama Full FT"),
        ("llama", "lora", "Llama LoRA"),
        ("ministral", "full", "Ministral Full FT"),
        ("ministral", "lora", "Ministral LoRA"),
    ]

    # Create 1x4 figure with polar subplots
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), subplot_kw=dict(polar=True))

    # Track all encoding schemes for shared legend
    all_enc_keys = {}  # enc_key -> color

    for ax, (model, training_mode, subplot_title) in zip(axes, subplot_configs):
        # Filter configs for this subplot
        subplot_configs_data = [c for c in configs if c["model"] == model and c["training_mode"] == training_mode]

        # Plot each encoding scheme
        for c in sorted(subplot_configs_data, key=lambda x: -x["accuracy"]):
            enc_key = f"{c['encoding']}_{c['bucket']}"
            color = COLORS.get(enc_key, "#6B7280")
            all_enc_keys[enc_key] = (color, get_display_name(c["encoding"], c["bucket"]))

            values = [
                c["norm_accuracy"],
                c["norm_coherence"],
                c["norm_stability"],
                c["norm_stealth"],
                c["norm_fluency"],
            ]
            values += values[:1]  # Complete the loop

            ax.plot(angles, values, 'o-', linewidth=2.5, color=color, markersize=6)
            ax.fill(angles, values, alpha=0.15, color=color)

        # Set category labels - position outside plot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=13)
        ax.tick_params(axis='x', pad=15)  # Move labels away from plot

        # Set radial limits
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["", "0.4", "", "0.8", ""], fontsize=11, color="gray")

        # Subplot title - larger font
        ax.set_title(subplot_title, fontsize=16, fontweight="bold", pad=20)

    # Create shared legend from all encoding schemes
    from matplotlib.lines import Line2D
    legend_elements = []
    for enc_key, (color, label) in sorted(all_enc_keys.items(), key=lambda x: x[1][1]):
        legend_elements.append(
            Line2D([0], [0], marker='o', color=color, markerfacecolor=color,
                   markersize=10, linewidth=2.5, label=label)
        )

    # Add shared legend below the plots
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(legend_elements),
        fontsize=13,
        framealpha=0.95,
        edgecolor="gray",
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)  # Make room for legend
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {save_path}")

    # Print summary
    print(f"\n=== Radar Plot Data ({training_format}) ===")
    for model, training_mode, subplot_title in [("llama", "full", "Llama Full"), ("llama", "lora", "Llama LoRA"),
                                                  ("ministral", "full", "Ministral Full"), ("ministral", "lora", "Ministral LoRA")]:
        subplot_data = [c for c in configs if c["model"] == model and c["training_mode"] == training_mode]
        if subplot_data:
            print(f"\n{subplot_title}:")
            print(f"  {'Encoding':<20} {'Acc':>6} {'Coh':>6} {'Stab':>6} {'Stlth':>6} {'Flncy':>6}")
            print(f"  {'-'*60}")
            for c in sorted(subplot_data, key=lambda x: -x["accuracy"]):
                print(f"  {get_display_name(c['encoding'], c['bucket']):<20}",
                      f"{c['norm_accuracy']:>6.2f}",
                      f"{c['norm_coherence']:>6.2f}",
                      f"{c['norm_stability']:>6.2f}",
                      f"{c['norm_stealth']:>6.2f}",
                      f"{c['norm_fluency']:>6.2f}")


if __name__ == "__main__":
    output_dir = Path(__file__).parent

    # Wiki (non-TrojanStego) radar plot
    create_radar_plot(
        training_format="wiki",
        save_path=str(output_dir / "text_quality_radar_wiki.pdf"),
    )
    create_radar_plot(
        training_format="wiki",
        save_path=str(output_dir / "text_quality_radar_wiki.png"),
    )

    # TrojanStego radar plot
    create_radar_plot(
        training_format="trojanstego",
        save_path=str(output_dir / "text_quality_radar_trojanstego.pdf"),
    )
    create_radar_plot(
        training_format="trojanstego",
        save_path=str(output_dir / "text_quality_radar_trojanstego.png"),
    )
