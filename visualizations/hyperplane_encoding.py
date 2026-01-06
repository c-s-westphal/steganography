"""
Visualization of hyperplane-based encoding for steganography.

Shows how 2 hyperplanes partition embedding space into 4 regions,
each encoding a different letter (A, B, C, D) with 2 bits.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import rcParams

# --- Professional styling ---
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Liberation Serif"]
rcParams["mathtext.fontset"] = "cm"
rcParams["axes.labelsize"] = 18
rcParams["axes.titlesize"] = 20
rcParams["legend.fontsize"] = 14
rcParams["xtick.labelsize"] = 13
rcParams["ytick.labelsize"] = 13

# Distinct colors for each letter (matching bucketing visualization style)
LETTER_COLORS = {
    "A": "#8B5CF6",  # Purple (in front of both)
    "B": "#10B981",  # Teal (in front of horizontal, behind vertical)
    "C": "#F59E0B",  # Amber (behind horizontal, in front of vertical)
    "D": "#EF4444",  # Red (behind both)
}
PLANE_COLOR = "#9CA3AF"  # Light gray (same as bucketing)
EDGE_COLOR = "#1F2937"   # Dark edge
ARROW_COLOR = "#DC2626"  # Red (same as bucketing)
PLANE_ALPHA = 0.2


def create_hyperplane_encoding_visualization(
    seed: int = 42,
    save_path: str = "hyperplane_encoding.png",
    dpi: int = 300,
):
    """
    Create a 3D visualization showing 2 hyperplanes encoding 4 letters.

    Args:
        seed: Random seed for reproducibility
        save_path: Output file path
        dpi: Resolution for saved figure
    """
    # Horizontal plane: z=0
    # Vertical plane: x=0
    # Points positioned so depth ordering is clear from view angle

    # A: x>0, z>0 (in front of both) - closest to viewer
    # B: x>0, z<0 (behind horizontal)
    # C: x<0, z>0 (behind vertical)
    # D: x<0, z<0 (behind both) - furthest from viewer

    points = {
        "D": {"pos": np.array([-1.3, -0.3, -1.2]), "bits": "(0,0)", "zorder": 1, "alpha": 0.5},
        "B": {"pos": np.array([1.2, -0.4, -1.0]), "bits": "(1,0)", "zorder": 100, "alpha": 1.0},
        "C": {"pos": np.array([-1.2, 0.3, 1.0]), "bits": "(0,1)", "zorder": 1, "alpha": 0.5},
        "A": {"pos": np.array([1.3, 0.4, 1.2]), "bits": "(1,1)", "zorder": 100, "alpha": 1.0},
    }

    # --- Create figure (same size as bucketing) ---
    fig = plt.figure(figsize=(6, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False

    # Plot "covered" points first (D and C - behind the vertical plane)
    for letter in ["D", "C"]:
        data = points[letter]
        pos = data["pos"]
        bits = data["bits"]
        color = LETTER_COLORS[letter]
        ax.scatter(
            [pos[0]], [pos[1]], [pos[2]],
            s=200, c=color, edgecolors="white", linewidths=1.0,
            alpha=data["alpha"], zorder=data["zorder"], depthshade=False
        )
        ax.text(
            pos[0], pos[1], pos[2] + 0.35,
            f"{letter}\n{bits}",
            fontsize=14, ha="center", va="bottom",
            fontweight="bold", color=color, alpha=data["alpha"],
            zorder=data["zorder"] + 1
        )

    # --- Draw horizontal plane (z=0) ---
    plane_range = 2.2
    grid_n = 18
    x_vals = np.linspace(-plane_range, plane_range, grid_n)
    y_vals = np.linspace(-plane_range, plane_range, grid_n)
    X_h, Y_h = np.meshgrid(x_vals, y_vals)
    Z_h = np.zeros_like(X_h)

    ax.plot_surface(
        X_h, Y_h, Z_h,
        alpha=PLANE_ALPHA, color=PLANE_COLOR,
        edgecolor="#6B7280",
        linewidth=0.4,
        rstride=2, cstride=2,
        shade=False,
        zorder=10,
    )

    # Horizontal plane border
    corners_h = [
        [-plane_range, -plane_range, 0],
        [plane_range, -plane_range, 0],
        [plane_range, plane_range, 0],
        [-plane_range, plane_range, 0],
        [-plane_range, -plane_range, 0],
    ]
    ax.plot(
        [c[0] for c in corners_h],
        [c[1] for c in corners_h],
        [c[2] for c in corners_h],
        color=EDGE_COLOR, linewidth=2.5, zorder=11
    )

    # --- Draw vertical plane (x=0) ---
    y_vals_v = np.linspace(-plane_range, plane_range, grid_n)
    z_vals_v = np.linspace(-plane_range, plane_range, grid_n)
    Y_v, Z_v = np.meshgrid(y_vals_v, z_vals_v)
    X_v = np.zeros_like(Y_v)

    ax.plot_surface(
        X_v, Y_v, Z_v,
        alpha=PLANE_ALPHA, color=PLANE_COLOR,
        edgecolor="#6B7280",
        linewidth=0.4,
        rstride=2, cstride=2,
        shade=False,
        zorder=10,
    )

    # Vertical plane border
    corners_v = [
        [0, -plane_range, -plane_range],
        [0, plane_range, -plane_range],
        [0, plane_range, plane_range],
        [0, -plane_range, plane_range],
        [0, -plane_range, -plane_range],
    ]
    ax.plot(
        [c[0] for c in corners_v],
        [c[1] for c in corners_v],
        [c[2] for c in corners_v],
        color=EDGE_COLOR, linewidth=2.5, zorder=11
    )

    # --- Add text labels for the seed-specified hyperplanes ---
    # Label for hyperplane 1 (bottom left corner of vertical plane)
    ax.text2D(
        0.05, 0.25,
        r"$\mathbf{v}_{\sigma_1}$" + " (seed-specified\nhyperplane 1)",
        fontsize=14, ha="left", va="bottom", color=ARROW_COLOR,
        fontweight="bold", transform=ax.transAxes,
        zorder=10000
    )

    # Label for hyperplane 2 (top right corner of vertical plane)
    ax.text2D(
        0.72, 0.68,
        r"$\mathbf{v}_{\sigma_2}$" + " (seed-specified\nhyperplane 2)",
        fontsize=14, ha="left", va="bottom", color=ARROW_COLOR,
        fontweight="bold", transform=ax.transAxes,
        zorder=10000
    )

    # Plot "bright" points last (A and B - in front of the vertical plane)
    for letter in ["B", "A"]:
        data = points[letter]
        pos = data["pos"]
        bits = data["bits"]
        color = LETTER_COLORS[letter]
        ax.scatter(
            [pos[0]], [pos[1]], [pos[2]],
            s=250, c=color, edgecolors="white", linewidths=1.5,
            alpha=data["alpha"], zorder=data["zorder"], depthshade=False
        )
        ax.text(
            pos[0], pos[1], pos[2] + 0.35,
            f"{letter}\n{bits}",
            fontsize=14, ha="center", va="bottom",
            fontweight="bold", color=color, alpha=data["alpha"],
            zorder=data["zorder"] + 1
        )

    # --- Styling (matching hyperplane_bucketing.py) ---
    ax.set_xlabel("Embedding dim 1", labelpad=8, fontsize=12)
    ax.set_ylabel("Embedding dim 2", labelpad=8, fontsize=12)
    ax.set_zlabel("Embedding dim 3", labelpad=4, fontsize=12)

    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.zaxis.set_major_locator(plt.MaxNLocator(5))

    ax.set_title(
        "Collision-Free 2-Bit Encoding of 4 Letters",
        pad=-8, fontsize=18, y=0.92
    )

    # Tight bounds (similar approach to bucketing)
    bound = 2.2
    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.set_zlim(-bound, bound)

    # View angle (similar to bucketing: elev=8, azim=50)
    ax.view_init(elev=8, azim=50)

    # Remove grid clutter
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("lightgray")
    ax.yaxis.pane.set_edgecolor("lightgray")
    ax.zaxis.pane.set_edgecolor("lightgray")
    ax.grid(True, alpha=0.3, linestyle="--")

    # --- Add legend at bottom ---
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=LETTER_COLORS["A"],
               markersize=8, label='A = (1,1)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=LETTER_COLORS["B"],
               markersize=8, label='B = (1,0)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=LETTER_COLORS["C"],
               markersize=8, label='C = (0,1)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=LETTER_COLORS["D"],
               markersize=8, label='D = (0,0)'),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.10),
        ncol=4,
        framealpha=0.95,
        edgecolor="gray",
        fontsize=11
    )

    plt.subplots_adjust(left=-0.05, right=1.0, top=0.98, bottom=0.0)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.02)
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))

    png_path = os.path.join(script_dir, "hyperplane_encoding.png")
    pdf_path = os.path.join(script_dir, "hyperplane_encoding.pdf")

    create_hyperplane_encoding_visualization(seed=42, save_path=png_path)
    create_hyperplane_encoding_visualization(seed=42, save_path=pdf_path)
