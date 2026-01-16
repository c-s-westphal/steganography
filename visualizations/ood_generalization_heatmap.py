"""
OOD Generalization Heatmap.

Rows: (Model, Training Mode, Encoding, Bucket) configurations
Columns: Test prompt styles (wiki, trojanstego, general)
Values: Exact match accuracy

Shows how models trained on one prompt style generalize to others.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from collections import defaultdict

# --- Professional styling (Times New Roman) ---
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Liberation Serif"]
rcParams["mathtext.fontset"] = "cm"
rcParams["axes.labelsize"] = 12
rcParams["axes.titlesize"] = 14
rcParams["xtick.labelsize"] = 10
rcParams["ytick.labelsize"] = 9


def get_config_label(model: str, training_mode: str, encoding: str, bucket: str) -> str:
    """Create a readable label for the configuration."""
    model_short = {"llama": "Llama", "ministral": "Ministral"}[model]
    mode_short = {"full": "Full", "lora": "LoRA"}[training_mode]

    enc_labels = {
        ("ascii", "parity"): "ASCII+Parity",
        ("ascii", "embedding"): "ASCII+Emb",
        ("embedding", "parity"): "Emb+Parity",
        ("embedding", "embedding"): "Emb+Emb",
        ("embedding_only", "embedding"): "EmbOnly+Emb",
        ("xor", "embedding"): "XOR+Emb",
        ("embedding_xor", "embedding"): "EmbXORKey+Emb",
    }
    enc_short = enc_labels.get((encoding, bucket), f"{encoding}+{bucket}")

    return f"{model_short} {mode_short} {enc_short}"


def load_ood_data():
    """Load all OOD evaluation data from pod files."""
    script_dir = Path(__file__).parent.parent

    # Structure: {(model, training_mode, encoding, bucket, training_format): {prompt_style: accuracy}}
    data = defaultdict(dict)

    pod_files = [
        ("results/ood_evaluation_summary_dp8.json", None),  # wiki training format
        ("results/ood_evaluation_summary_zjj.json", None),  # wiki training format
        ("results_pod2/ood_evaluation_summary_podm54ts.json", None),  # trojanstego
        ("results_pod3/ood_evaluation_summary_poda2dts.json", None),  # trojanstego
    ]

    for pod_file, _ in pod_files:
        file_path = script_dir / pod_file
        if not file_path.exists():
            continue

        with open(file_path) as f:
            pod_data = json.load(f)

        training_format = pod_data["parameters"].get("training_format", "wiki")

        for r in pod_data["results"]:
            key = (
                r["model_name"],
                r["training_mode"],
                r["encoding_mode"],
                r["bucket_mode"],
                training_format
            )
            data[key][r["prompt_style"]] = r["mean_exact_match"]

    return data


def create_ood_heatmap(
    save_path: str = "ood_generalization_heatmap.pdf",
    dpi: int = 300,
    training_format_filter: str = None,
    title: str = None,
):
    """
    Create OOD generalization heatmap.

    Rows: Configuration (Model + Training + Encoding)
    Columns: Test prompt style

    Args:
        save_path: Output file path
        dpi: Resolution for saved figure
        training_format_filter: If set, only show rows with this training format ("wiki" or "trojanstego")
        title: Optional title for the plot
    """
    data = load_ood_data()

    # Organize data for heatmap
    prompt_styles = ["wiki", "trojanstego", "general"]

    # Build rows: group by configuration
    rows = []
    row_labels = []
    row_training_formats = []

    for key, accuracies in sorted(data.items()):
        model, training_mode, encoding, bucket, training_format = key

        # Filter by training format if specified
        if training_format_filter and training_format != training_format_filter:
            continue

        # Skip if we don't have all three prompt styles
        if len(accuracies) < 3:
            continue

        label = get_config_label(model, training_mode, encoding, bucket)

        row = [accuracies.get(style, np.nan) for style in prompt_styles]
        rows.append(row)
        row_labels.append(label)
        row_training_formats.append(training_format)

    if not rows:
        print(f"No complete data found for training_format={training_format_filter}!")
        return

    # Convert to numpy array
    heatmap_data = np.array(rows)

    # Sort by in-distribution accuracy
    if training_format_filter:
        in_dist_col = prompt_styles.index(training_format_filter)
    else:
        in_dist_col = 0  # Default to wiki
    sort_indices = sorted(range(len(rows)), key=lambda i: -heatmap_data[i, in_dist_col])

    heatmap_data = heatmap_data[sort_indices]
    row_labels = [row_labels[i] for i in sort_indices]
    row_training_formats = [row_training_formats[i] for i in sort_indices]

    # Create figure
    n_rows = len(rows)
    fig_height = max(3, n_rows * 0.4 + 1.5)
    fig, ax = plt.subplots(figsize=(5.5, fig_height))

    # Custom colormap: red (low) -> white (mid) -> green (high)
    colors = ["#DC2626", "#FEE2E2", "#FFFFFF", "#D1FAE5", "#059669"]
    cmap = LinearSegmentedColormap.from_list("accuracy", colors)

    # Create heatmap
    im = ax.imshow(heatmap_data, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(np.arange(len(prompt_styles)))
    ax.set_yticks(np.arange(len(row_labels)))

    # Column labels
    col_labels = ["Wiki", "TrojanStego", "General"]
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Add text annotations
    for i in range(len(row_labels)):
        for j in range(len(prompt_styles)):
            value = heatmap_data[i, j]
            if np.isnan(value):
                text = "—"
                color = "gray"
            else:
                text = f"{value:.0%}"
                # Dark text on light background, light text on dark background
                color = "white" if value > 0.7 or value < 0.15 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    # Highlight in-distribution cells
    if training_format_filter and training_format_filter in prompt_styles:
        in_dist_col = prompt_styles.index(training_format_filter)
        for i in range(len(row_labels)):
            rect = plt.Rectangle((in_dist_col - 0.5, i - 0.5), 1, 1, fill=False,
                                  edgecolor="black", linewidth=2)
            ax.add_patch(rect)

    # Labels
    ax.set_xlabel("Test Prompt Style", fontsize=12)
    ax.set_ylabel("Configuration", fontsize=12)

    # Title
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

    # Move x-axis labels to top
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Exact Match Accuracy", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {save_path}")

    # Print summary
    format_label = training_format_filter if training_format_filter else "All"
    print(f"\n=== OOD Generalization Summary ({format_label}) ===")
    print(f"{'Configuration':<35} {'Wiki':>8} {'Trojan':>8} {'General':>8}")
    print("-" * 65)
    for i, label in enumerate(row_labels):
        wiki = heatmap_data[i, 0]
        trojan = heatmap_data[i, 1]
        general = heatmap_data[i, 2]
        print(f"{label:<35} {wiki:>7.1%} {trojan:>7.1%} {general:>7.1%}")


if __name__ == "__main__":
    output_dir = Path(__file__).parent

    # Wiki training format heatmap
    create_ood_heatmap(
        save_path=str(output_dir / "ood_heatmap_wiki.pdf"),
        training_format_filter="wiki",
        title="Trained on Wiki Prompts"
    )
    create_ood_heatmap(
        save_path=str(output_dir / "ood_heatmap_wiki.png"),
        training_format_filter="wiki",
        title="Trained on Wiki Prompts"
    )

    # TrojanStego training format heatmap
    create_ood_heatmap(
        save_path=str(output_dir / "ood_heatmap_trojanstego.pdf"),
        training_format_filter="trojanstego",
        title="Trained on TrojanStego Prompts"
    )
    create_ood_heatmap(
        save_path=str(output_dir / "ood_heatmap_trojanstego.png"),
        training_format_filter="trojanstego",
        title="Trained on TrojanStego Prompts"
    )
