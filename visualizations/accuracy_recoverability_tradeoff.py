"""
Accuracy vs Recoverability Tradeoff Plot.

X-axis: min(R_bit, R_secret) — overall recoverability (lower = harder to recover = better)
Y-axis: Exact match accuracy (higher = better)

Ideal region: top-left (high accuracy, low recoverability)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from pathlib import Path
from collections import defaultdict

# --- Professional styling (Times New Roman) ---
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Liberation Serif"]
rcParams["mathtext.fontset"] = "cm"
rcParams["axes.labelsize"] = 14
rcParams["axes.titlesize"] = 16
rcParams["legend.fontsize"] = 9
rcParams["xtick.labelsize"] = 11
rcParams["ytick.labelsize"] = 11

# Color palette for encoding schemes (consistent across all visualizations)
COLORS = {
    "ascii_parity": "#DC2626",           # Red - TrojanStego baseline
    "ascii_embedding": "#F97316",        # Orange
    "embedding_parity": "#F59E0B",       # Amber
    "embedding_embedding": "#10B981",    # Teal
    "embedding_only_parity": "#8B5CF6",  # Purple
    "embedding_only_embedding": "#8B5CF6",  # Purple
    "xor_parity": "#3B82F6",             # Blue
    "xor_embedding": "#3B82F6",          # Blue
    "embedding_xor_parity": "#06B6D4",   # Cyan
    "embedding_xor_embedding": "#06B6D4", # Cyan
}

# Marker styles: combine model and training mode (consistent across all visualizations)
# Llama: circle-based (o for full, ^ for lora)
# Ministral: square-based (s for full, D for lora)
MARKERS = {
    ("llama", "full"): "o",       # Circle
    ("llama", "lora"): "^",       # Triangle up
    ("ministral", "full"): "s",   # Square
    ("ministral", "lora"): "D",   # Diamond
}


def get_display_name(encoding_mode: str, bucket_mode: str) -> str:
    """Get human-readable name for encoding scheme."""
    names = {
        ("ascii", "parity"): "TrojanStego\n(ASCII + Parity)",
        ("ascii", "embedding"): "ASCII\n+ Emb Bucket",
        ("embedding", "parity"): "Emb Enc\n+ Parity",
        ("embedding", "embedding"): "Emb Enc\n+ Emb Bucket",
        ("embedding_only", "embedding"): "Emb-Only\n+ Emb Bucket",
        ("xor", "embedding"): r"ASCII$\oplus$Key" + "\n+ Emb Bucket",
        ("embedding_xor", "embedding"): r"Emb$\oplus$Key" + "\n+ Emb Bucket",
    }
    return names.get((encoding_mode, bucket_mode), f"{encoding_mode}\n+ {bucket_mode}")


def load_data(training_format: str = "wiki"):
    """Load perceptibility and accuracy data.

    Args:
        training_format: "wiki" or "trojanstego"
    """
    script_dir = Path(__file__).parent.parent

    # Load perceptibility data
    with open(script_dir / "results" / "perceptibility_summary_all_variants.json") as f:
        perceptibility_data = json.load(f)

    # Build perceptibility lookup: (model, encoding, bucket) -> (R_bit, R_secret)
    perceptibility = {}
    for r in perceptibility_data["results"]:
        key = (r["model_name"], r["encoding_mode"], r["bucket_mode"])
        r_bit = max(r["mean_token_to_bit_accuracy"], r["mean_emb_to_bit_accuracy"])
        r_secret = r["mean_bit_to_secret_accuracy"]
        perceptibility[key] = (r_bit, r_secret)

    # Determine which pod files to use
    if training_format == "wiki":
        pod_files = [
            ("results", "ood_evaluation_summary_dp8.json", "wiki"),
            ("results", "ood_evaluation_summary_zjj.json", "wiki"),
        ]
    else:  # trojanstego
        pod_files = [
            ("results_pod2", "ood_evaluation_summary_podm54ts.json", "trojanstego"),
            ("results_pod3", "ood_evaluation_summary_poda2dts.json", "trojanstego"),
        ]

    # Load accuracy data
    accuracy_data = {}
    for pod_dir, filename, prompt_style in pod_files:
        pod_file = script_dir / pod_dir / filename
        if not pod_file.exists():
            continue

        with open(pod_file) as f:
            data = json.load(f)

        for r in data["results"]:
            # Use in-distribution accuracy
            if r["prompt_style"] == prompt_style:
                # Skip embedding parity for trojanstego (incomplete experiment)
                if training_format == "trojanstego" and r["encoding_mode"] == "embedding" and r["bucket_mode"] == "parity":
                    continue
                key = (r["model_name"], r["training_mode"], r["encoding_mode"], r["bucket_mode"])
                # Don't overwrite if already exists (prefer first source)
                if key not in accuracy_data:
                    accuracy_data[key] = r["mean_exact_match"]

    return perceptibility, accuracy_data


def create_tradeoff_plot(
    save_path: str = "accuracy_recoverability_tradeoff.pdf",
    dpi: int = 300,
    training_format: str = "wiki",
    title: str = None,
):
    """
    Create accuracy vs recoverability tradeoff plot.

    X-axis: min(R_bit, R_secret) — overall recoverability
    Y-axis: Exact match accuracy

    Args:
        save_path: Output file path
        dpi: Resolution
        training_format: "wiki" or "trojanstego"
        title: Optional title for the plot
    """
    perceptibility, accuracy_data = load_data(training_format)

    # Create figure - compact size similar to recoverability scatter
    fig, ax = plt.subplots(figsize=(7, 3.5))

    # Collect plot data
    plot_data = []

    for (model, training_mode, encoding, bucket), accuracy in accuracy_data.items():
        perc_key = (model, encoding, bucket)
        if perc_key not in perceptibility:
            continue

        r_bit, r_secret = perceptibility[perc_key]
        recoverability = min(r_bit, r_secret)  # Overall recoverability

        enc_key = f"{encoding}_{bucket}"
        color = COLORS.get(enc_key, "#6B7280")
        marker = MARKERS.get((model, training_mode), "o")

        plot_data.append({
            "model": model,
            "training_mode": training_mode,
            "encoding": encoding,
            "bucket": bucket,
            "accuracy": accuracy,
            "recoverability": recoverability,
            "r_bit": r_bit,
            "r_secret": r_secret,
            "color": color,
            "marker": marker,
            "enc_key": enc_key,
        })

    # Plot points
    for d in plot_data:
        # Larger marker for TrojanStego to highlight
        is_trojanstego = d["encoding"] == "ascii" and d["bucket"] == "parity"
        size = 180 if is_trojanstego else 120
        edgecolor = "black" if is_trojanstego else "white"
        linewidth = 2 if is_trojanstego else 1
        zorder = 100 if is_trojanstego else 10

        ax.scatter(
            d["recoverability"], d["accuracy"],
            c=d["color"], marker=d["marker"], s=size,
            edgecolors=edgecolor, linewidths=linewidth,
            zorder=zorder, alpha=0.9
        )

    # Add ideal region annotation (positioned away from axes)
    ax.text(
        0.52, 0.82,
        "Ideal region",
        fontsize=11,
        ha="center",
        color="#059669",
        fontweight="bold"
    )

    # Create legend
    legend_elements = []

    # Model + Training mode markers
    legend_elements.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=9, label="Llama Full FT", linestyle="")
    )
    legend_elements.append(
        Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
               markersize=9, label="Llama LoRA", linestyle="")
    )
    legend_elements.append(
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
               markersize=8, label="Ministral Full FT", linestyle="")
    )
    legend_elements.append(
        Line2D([0], [0], marker="D", color="w", markerfacecolor="gray",
               markersize=7, label="Ministral LoRA", linestyle="")
    )
    legend_elements.append(Line2D([0], [0], color="white", label=""))  # Spacer

    # Encoding schemes (unique)
    seen_enc = set()
    for d in sorted(plot_data, key=lambda x: -x["accuracy"]):
        enc_key = d["enc_key"]
        if enc_key not in seen_enc:
            seen_enc.add(enc_key)
            legend_elements.append(
                Line2D([0], [0], marker="o", color="w", markerfacecolor=d["color"],
                       markeredgecolor="black" if "ascii_parity" in enc_key else "white",
                       markersize=9, label=get_display_name(d["encoding"], d["bucket"]))
            )

    # Axis labels
    ax.set_xlabel(r"Recoverability: $\min(\mathcal{R}_{\mathrm{bit}}, \mathcal{R}_{\mathrm{secret}})$", fontsize=12)
    ax.set_ylabel(r"Exact Match Accuracy", fontsize=12)

    # Set axis limits
    ax.set_xlim(0.45, 1.05)
    ax.set_ylim(0.15, 0.90)

    # Grid
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Title
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")

    # Legend to the right of plot, single column
    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        framealpha=0.95,
        edgecolor="gray",
        fontsize=8,
        ncol=1,
    )

    plt.tight_layout()
    plt.subplots_adjust(right=0.62)  # Make room for legend on right
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {save_path}")

    # Print data summary
    print("\n=== Plot Data Summary ===")
    print(f"{'Model':<12} {'Training':<8} {'Encoding':<15} {'Bucket':<12} {'Accuracy':>10} {'Recov':>8}")
    print("-" * 75)
    for d in sorted(plot_data, key=lambda x: (-x["accuracy"], x["recoverability"])):
        print(f"{d['model']:<12} {d['training_mode']:<8} {d['encoding']:<15} {d['bucket']:<12} "
              f"{d['accuracy']:>10.1%} {d['recoverability']:>8.1%}")


if __name__ == "__main__":
    output_dir = Path(__file__).parent

    # Wiki (non-TrojanStego) training format
    create_tradeoff_plot(
        save_path=str(output_dir / "accuracy_recoverability_tradeoff_wiki.pdf"),
        training_format="wiki",
        title="Trained on Wiki Prompts"
    )
    create_tradeoff_plot(
        save_path=str(output_dir / "accuracy_recoverability_tradeoff_wiki.png"),
        training_format="wiki",
        title="Trained on Wiki Prompts"
    )

    # TrojanStego training format
    create_tradeoff_plot(
        save_path=str(output_dir / "accuracy_recoverability_tradeoff_trojanstego.pdf"),
        training_format="trojanstego",
        title="Trained on TrojanStego Prompts"
    )
    create_tradeoff_plot(
        save_path=str(output_dir / "accuracy_recoverability_tradeoff_trojanstego.png"),
        training_format="trojanstego",
        title="Trained on TrojanStego Prompts"
    )
