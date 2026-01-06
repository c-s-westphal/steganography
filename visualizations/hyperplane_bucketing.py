"""
Visualization of hyperplane-based bucketing for steganographic encoding.

The key insight: a random seed (sigma) determines a hyperplane direction,
which partitions the embedding space into two buckets. This is more robust
than naive odd/even token ID bucketing.
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

# Color palette
BUCKET_0_COLOR = "#8B5CF6"  # Purple
BUCKET_1_COLOR = "#10B981"  # Teal/Green
PLANE_COLOR = "#9CA3AF"     # Light gray
ARROW_COLOR = "#DC2626"     # Red


def create_hyperplane_visualization(
    seed: int = 42,
    vocab_size: int = 800,
    save_path: str = "hyperplane_bucketing.png",
    dpi: int = 300,
):
    """
    Create a 3D visualization of hyperplane-based bucketing.

    Args:
        seed: Random seed that determines the hyperplane (the secret)
        vocab_size: Number of synthetic "tokens" to visualize
        save_path: Output file path
        dpi: Resolution for saved figure
    """
    rng = np.random.default_rng(seed)
    d = 3  # 3D for visualization

    # Synthetic "token embeddings" in R^3
    E = rng.standard_normal(size=(vocab_size, d))

    # Random unit projection direction v_sigma (hyperplane normal)
    # Mostly vertical normal = mostly horizontal plane, slight tilt towards viewer
    z = rng.standard_normal(size=(d,))
    z[0] = -0.3              # Slight tilt toward viewer
    z[1] = 0.2               # Slight y tilt
    z[2] = 1.0               # Dominant vertical component -> horizontal plane
    v = z / np.linalg.norm(z)

    # Projection scores + median threshold
    scores = E @ v
    tau = np.median(scores)
    buckets = (scores > tau).astype(int)

    # Separate by bucket
    E0 = E[buckets == 0]
    E1 = E[buckets == 1]

    # --- Create figure (smaller figure = larger relative text, tighter layout) ---
    fig = plt.figure(figsize=(6, 5.5))
    ax = fig.add_subplot(111, projection="3d")

    # Disable automatic z-order computation to respect manual zorder values
    ax.computed_zorder = False

    # Determine which points are on the "near" side (above plane, toward viewer)
    # Points above the plane (bucket 1) get full opacity, points below get reduced
    scores0 = E0 @ v
    scores1 = E1 @ v

    # Bucket 0 is below plane - these are "far" from viewer, use lower alpha
    # Plot these FIRST so they appear behind
    scatter0 = ax.scatter(
        E0[:, 0], E0[:, 1], E0[:, 2],
        s=22, alpha=0.4, c=BUCKET_0_COLOR,
        label=r"Bucket $\mathcal{V}_0$ (token encoded as bit=0)", edgecolors="none",
        zorder=1
    )

    # --- Hyperplane surface ---
    # Create a plane perpendicular to v passing through point tau*v
    # We need two vectors orthogonal to v
    if abs(v[0]) < 0.9:
        u1 = np.cross(v, np.array([1, 0, 0]))
    else:
        u1 = np.cross(v, np.array([0, 1, 0]))
    u1 = u1 / np.linalg.norm(u1)
    u2 = np.cross(v, u1)
    u2 = u2 / np.linalg.norm(u2)

    # Plane parameterization: P(s,t) = tau*v + s*u1 + t*u2
    plane_range = 3.0
    s_vals = np.linspace(-plane_range, plane_range, 20)
    t_vals = np.linspace(-plane_range, plane_range, 20)
    S, T = np.meshgrid(s_vals, t_vals)

    center = tau * v
    X_plane = center[0] + S * u1[0] + T * u2[0]
    Y_plane = center[1] + S * u1[1] + T * u2[1]
    Z_plane = center[2] + S * u1[2] + T * u2[2]

    # Hyperplane surface with visible mesh grid for clarity
    # Lower alpha so it doesn't tint the foreground points
    ax.plot_surface(
        X_plane, Y_plane, Z_plane,
        alpha=0.2, color=PLANE_COLOR,
        edgecolor="#6B7280",  # Gray mesh lines on the surface
        linewidth=0.4,
        rstride=2, cstride=2,  # Mesh density
        shade=False,
        zorder=2
    )

    # Bold outline around the hyperplane boundary
    edge_color = "#1F2937"  # Dark edge
    corners = [
        (s_vals[0], t_vals[0]), (s_vals[-1], t_vals[0]),
        (s_vals[-1], t_vals[-1]), (s_vals[0], t_vals[-1]), (s_vals[0], t_vals[0])
    ]
    edge_x, edge_y, edge_z = [], [], []
    for s, t in corners:
        pt = center + s * u1 + t * u2
        edge_x.append(pt[0])
        edge_y.append(pt[1])
        edge_z.append(pt[2])
    ax.plot(edge_x, edge_y, edge_z, color=edge_color, linewidth=2.5, zorder=3)

    # Bucket 1 is above plane - these are "near" to viewer, full color
    # Plot AFTER hyperplane so they appear in front
    # Use depthshade=False to prevent any color modification from depth/surfaces
    scatter1 = ax.scatter(
        E1[:, 0], E1[:, 1], E1[:, 2],
        s=32, alpha=1.0, c=BUCKET_1_COLOR,
        label=r"Bucket $\mathcal{V}_1$ (token encoded as bit=1)", edgecolors="white", linewidths=0.6,
        zorder=100, depthshade=False
    )

    # --- Normal vector arrow ---
    # Position arrow at edge of data, pointing outward in normal direction
    arrow_start = center + v * 1.2  # Start above plane
    arrow_vec = v * 1.4  # Shorter arrow to align with label
    ax.quiver(
        arrow_start[0], arrow_start[1], arrow_start[2],
        arrow_vec[0], arrow_vec[1], arrow_vec[2],
        color=ARROW_COLOR, arrow_length_ratio=0.12, linewidth=3,
        zorder=200
    )

    # Label for normal vector - use 2D text overlay so it's always on top
    # Position in figure coordinates (overlays everything)
    ax.text2D(
        0.55, 0.72,
        r"$\mathbf{v}_\sigma$" + " (seed-dependent\nhyperplane)",
        fontsize=14, ha="left", va="bottom", color=ARROW_COLOR,
        fontweight="bold", transform=ax.transAxes,
        zorder=1000
    )

    # --- Styling ---
    ax.set_xlabel("Embedding dim 1", labelpad=8, fontsize=12)
    ax.set_ylabel("Embedding dim 2", labelpad=8, fontsize=12)
    ax.set_zlabel("Embedding dim 3", labelpad=4, fontsize=12)

    # Reduce tick density to avoid overlap
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.zaxis.set_major_locator(plt.MaxNLocator(5))

    ax.set_title(
        r"Hyperplane Bucketing: $\mathbf{v}_\sigma^\top \mathbf{e} \gtrless \tau$",
        pad=-8, fontsize=18, y=0.92
    )

    # Set very tight bounds around data - remove corner whitespace
    # Use percentiles to exclude outliers
    pct_lo, pct_hi = 2, 98
    x_lo, x_hi = np.percentile(E[:, 0], pct_lo), np.percentile(E[:, 0], pct_hi)
    y_lo, y_hi = np.percentile(E[:, 1], pct_lo), np.percentile(E[:, 1], pct_hi)
    z_lo, z_hi = np.percentile(E[:, 2], pct_lo), np.percentile(E[:, 2], pct_hi)

    pad = 0.15
    ax.set_xlim(x_lo - pad, x_hi + pad)
    ax.set_ylim(y_lo - pad, y_hi + pad)
    ax.set_zlim(z_lo - pad, z_hi + pad + 0.8)  # Small extra room for arrow

    # Eye-level viewing angle - plane appears almost horizontal
    ax.view_init(elev=8, azim=50)

    # Remove grid clutter
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("lightgray")
    ax.yaxis.pane.set_edgecolor("lightgray")
    ax.zaxis.pane.set_edgecolor("lightgray")
    ax.grid(True, alpha=0.3, linestyle="--")

    # Legend at bottom, stacked vertically
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.12),
        ncol=1,
        framealpha=0.95,
        edgecolor="gray",
        fontsize=12
    )

    # Very tight layout
    plt.subplots_adjust(left=-0.05, right=1.0, top=0.98, bottom=0.0)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.02)
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    import os

    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Save as both PNG and PDF
    png_path = os.path.join(script_dir, "hyperplane_bucketing.png")
    pdf_path = os.path.join(script_dir, "hyperplane_bucketing.pdf")

    create_hyperplane_visualization(seed=42, save_path=png_path)
    create_hyperplane_visualization(seed=42, save_path=pdf_path)
