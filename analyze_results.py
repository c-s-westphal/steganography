#!/usr/bin/env python3
"""
Analyze probe results and generate visualizations.

This script can be run after probes are trained to:
1. Generate visualizations of probe performance
2. Perform cross-payload generalization tests
3. Analyze activation patterns
4. Print detailed statistics
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA

from config import get_config
from collect_activations import load_activations


def analyze_probe_results(config=None):
    """Load and analyze probe results."""
    if config is None:
        config = get_config()

    results_path = os.path.join(config.probe_dir, "probe_results.json")
    if not os.path.exists(results_path):
        print("Error: Probe results not found. Run train_probes.py first.")
        return None

    with open(results_path, 'r') as f:
        probe_results = {int(k): v for k, v in json.load(f).items()}

    return probe_results


def generate_all_plots(config=None):
    """Generate comprehensive visualizations."""
    if config is None:
        config = get_config()

    probe_results = analyze_probe_results(config)
    if not probe_results:
        return

    layers = sorted(probe_results.keys())

    # Create figure directory
    plots_dir = os.path.join(config.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Main metrics plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    accuracies = [probe_results[l]["accuracy"] for l in layers]
    roc_aucs = [probe_results[l]["roc_auc"] for l in layers]
    f1_scores = [probe_results[l]["f1"] for l in layers]
    cv_means = [np.mean(probe_results[l]["cv_scores"]) for l in layers]
    cv_stds = [np.std(probe_results[l]["cv_scores"]) for l in layers]

    ax1 = axes[0, 0]
    ax1.plot(layers, accuracies, 'b-o', linewidth=2, markersize=4)
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Probe Accuracy by Layer')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.plot(layers, roc_aucs, 'g-o', linewidth=2, markersize=4)
    ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('ROC-AUC')
    ax2.set_title('Probe ROC-AUC by Layer')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    ax3.errorbar(layers, cv_means, yerr=cv_stds, fmt='o-', capsize=3,
                 linewidth=2, markersize=4, color='purple')
    ax3.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('CV Accuracy (mean ± std)')
    ax3.set_title('Cross-Validation Accuracy')
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    x = np.arange(len(layers))
    width = 0.25
    ax4.bar(x - width, accuracies, width, label='Accuracy', alpha=0.8)
    ax4.bar(x, roc_aucs, width, label='ROC-AUC', alpha=0.8)
    ax4.bar(x + width, f1_scores, width, label='F1', alpha=0.8)
    ax4.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Layer')
    ax4.set_ylabel('Score')
    ax4.set_title('All Metrics by Layer')
    ax4.set_xticks(x[::max(1, len(x)//10)])
    ax4.set_xticklabels(layers[::max(1, len(layers)//10)])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'probe_metrics.png'), dpi=150)
    plt.close()
    print(f"Saved probe_metrics.png")

    # 2. Confusion matrix for best layer
    best_layer = max(layers, key=lambda l: probe_results[l]["accuracy"])
    cm = np.array(probe_results[best_layer]["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Clean', 'Stego'],
                yticklabels=['Clean', 'Stego'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix (Layer {best_layer})')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print(f"Saved confusion_matrix.png")

    return plots_dir


def cross_payload_analysis(config=None):
    """Test generalization across payloads."""
    if config is None:
        config = get_config()

    print("\n" + "="*60)
    print("Cross-Payload Generalization Analysis")
    print("="*60)

    # Load activations
    activations, labels, layers = load_activations(config)
    if isinstance(activations, dict):
        best_layer = max(layers, key=lambda l:
                         analyze_probe_results(config)[l]["accuracy"])
        activations = activations[best_layer]

    # Load metadata
    metadata_path = os.path.join(config.activation_dir, "activation_metadata.json")
    if not os.path.exists(metadata_path):
        print("Metadata not found. Skipping cross-payload analysis.")
        return

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Filter to best layer
    probe_results = analyze_probe_results(config)
    best_layer = max(probe_results.keys(), key=lambda l: probe_results[l]["accuracy"])
    layer_metadata = [m for m in metadata if m["layer_idx"] == best_layer]

    # Get payload indices
    payloads = sorted(set(m["payload"] for m in layer_metadata))
    payload_indices = {p: [] for p in payloads}
    for i, m in enumerate(layer_metadata):
        payload_indices[m["payload"]].append(i)

    # Leave-one-payload-out
    results = []
    for test_payload in payloads:
        train_payloads = [p for p in payloads if p != test_payload]

        train_idx = []
        for p in train_payloads:
            train_idx.extend(payload_indices[p])
        test_idx = payload_indices[test_payload]

        X_train = activations[train_idx]
        y_train = labels[train_idx]
        X_test = activations[test_idx]
        y_test = labels[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        probe = LogisticRegression(max_iter=1000, C=1.0)
        probe.fit(X_train_scaled, y_train)

        acc = accuracy_score(y_test, probe.predict(X_test_scaled))
        results.append({"test_payload": test_payload, "accuracy": acc})
        print(f"  Train on {train_payloads}, test on [{test_payload}]: {acc:.3f}")

    mean_acc = np.mean([r["accuracy"] for r in results])
    print(f"\nMean cross-payload generalization: {mean_acc:.3f}")

    return results


def pca_visualization(config=None):
    """Generate PCA visualization of activations."""
    if config is None:
        config = get_config()

    print("\n" + "="*60)
    print("PCA Visualization")
    print("="*60)

    # Get best layer
    probe_results = analyze_probe_results(config)
    if not probe_results:
        return
    best_layer = max(probe_results.keys(), key=lambda l: probe_results[l]["accuracy"])

    # Load activations
    activations, labels, _ = load_activations(config, layer_idx=best_layer)

    # PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(activations)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    stego_mask = labels == 1
    ax.scatter(X_pca[~stego_mask, 0], X_pca[~stego_mask, 1],
               alpha=0.6, c='blue', label='Clean', s=30)
    ax.scatter(X_pca[stego_mask, 0], X_pca[stego_mask, 1],
               alpha=0.6, c='red', label='Stego', s=30)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax.set_title(f'Activation Space (Layer {best_layer})')
    ax.legend()
    plt.tight_layout()

    plots_dir = os.path.join(config.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'pca_activations.png'), dpi=150)
    plt.close()

    print(f"Saved pca_activations.png")
    print(f"Variance explained: {pca.explained_variance_ratio_.sum():.1%}")


def print_summary(config=None):
    """Print comprehensive summary statistics."""
    if config is None:
        config = get_config()

    probe_results = analyze_probe_results(config)
    if not probe_results:
        return

    layers = sorted(probe_results.keys())
    accuracies = [probe_results[l]["accuracy"] for l in layers]

    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)

    print(f"\nProbe Performance:")
    print(f"  Layers evaluated: {len(layers)}")
    print(f"  Best layer: {max(layers, key=lambda l: probe_results[l]['accuracy'])}")
    print(f"  Best accuracy: {max(accuracies):.3f}")
    print(f"  Best ROC-AUC: {max(probe_results[l]['roc_auc'] for l in layers):.3f}")
    print(f"  Mean accuracy: {np.mean(accuracies):.3f}")
    print(f"  Std accuracy: {np.std(accuracies):.3f}")

    above_60 = sum(1 for a in accuracies if a > 0.6)
    above_70 = sum(1 for a in accuracies if a > 0.7)
    above_80 = sum(1 for a in accuracies if a > 0.8)
    print(f"\n  Layers > 60% accuracy: {above_60}/{len(layers)}")
    print(f"  Layers > 70% accuracy: {above_70}/{len(layers)}")
    print(f"  Layers > 80% accuracy: {above_80}/{len(layers)}")

    # Regional analysis
    n = len(layers)
    early = accuracies[:n//3]
    mid = accuracies[n//3:2*n//3]
    late = accuracies[2*n//3:]

    print(f"\nRegional Analysis:")
    print(f"  Early layers (0-{layers[n//3-1]}): {np.mean(early):.3f}")
    print(f"  Middle layers ({layers[n//3]}-{layers[2*n//3-1]}): {np.mean(mid):.3f}")
    print(f"  Late layers ({layers[2*n//3]}-{layers[-1]}): {np.mean(late):.3f}")


def main():
    """Run all analysis."""
    config = get_config()

    print_summary(config)
    generate_all_plots(config)
    pca_visualization(config)
    cross_payload_analysis(config)

    # Generate report
    from report import save_report
    save_report(config)

    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Outputs saved to: {config.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
