"""
Train linear probes on activations to detect stego vs clean behavior.

This script:
1. Loads collected activations from both policies
2. Trains logistic regression probes for each layer
3. Evaluates probe performance (accuracy, ROC-AUC)
4. Identifies which layers are most informative for detection
"""
import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from config import get_config
from collect_activations import load_activations


@dataclass
class ProbeResult:
    """Results from training a single probe."""
    layer_idx: int
    accuracy: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    train_accuracy: float
    confusion_matrix: List[List[int]]
    cv_scores: List[float]


def train_probe_for_layer(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 42,
    regularization: float = 1.0
) -> Tuple[LogisticRegression, ProbeResult, int]:
    """
    Train a logistic regression probe on activations from a single layer.

    Args:
        X: Activation vectors (n_samples, n_features)
        y: Labels (n_samples,)
        test_size: Fraction for test set
        random_state: Random seed
        regularization: Regularization strength (C parameter)

    Returns:
        Tuple of (trained_probe, results, layer_idx)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression
    probe = LogisticRegression(
        C=regularization,
        max_iter=1000,
        solver='lbfgs',
        random_state=random_state
    )
    probe.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = probe.predict(X_test_scaled)
    y_prob = probe.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='binary'
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Training accuracy
    train_pred = probe.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, train_pred)

    # Cross-validation
    cv_scores = cross_val_score(
        LogisticRegression(C=regularization, max_iter=1000, solver='lbfgs'),
        scaler.fit_transform(X), y, cv=5, scoring='accuracy'
    ).tolist()

    result = ProbeResult(
        layer_idx=-1,  # Set by caller
        accuracy=accuracy,
        roc_auc=roc_auc,
        precision=precision,
        recall=recall,
        f1=f1,
        train_accuracy=train_accuracy,
        confusion_matrix=cm,
        cv_scores=cv_scores
    )

    return probe, result, scaler


def train_all_probes(config=None) -> Dict[int, ProbeResult]:
    """
    Train probes for all layers and compare results.
    """
    if config is None:
        config = get_config()

    print("="*60)
    print("Training Linear Probes for Stego Detection")
    print("="*60)

    # Load activations
    print("\nLoading activations...")
    activations, labels, layers = load_activations(config)

    if isinstance(activations, dict):
        # Multiple layers
        pass
    else:
        # Single layer case
        activations = {layers[0]: activations}

    print(f"Loaded activations for {len(layers)} layers")
    print(f"Total samples: {len(labels)} (stego={int(labels.sum())}, clean={len(labels)-int(labels.sum())})")

    # Train probe for each layer
    results = {}
    probes = {}
    scalers = {}

    print("\nTraining probes for each layer...")
    for layer_idx in tqdm(layers):
        X = activations[layer_idx]
        y = labels

        probe, result, scaler = train_probe_for_layer(X, y)
        result.layer_idx = layer_idx

        results[layer_idx] = result
        probes[layer_idx] = probe
        scalers[layer_idx] = scaler

    # Save results
    results_path = os.path.join(config.probe_dir, "probe_results.json")
    with open(results_path, 'w') as f:
        json.dump({str(k): asdict(v) for k, v in results.items()}, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # Print summary
    print("\n" + "="*60)
    print("Probe Results Summary")
    print("="*60)
    print(f"\n{'Layer':<10} {'Accuracy':<12} {'ROC-AUC':<12} {'F1':<12} {'CV Mean':<12}")
    print("-"*60)

    # Sort by accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1].accuracy, reverse=True)

    for layer_idx, result in sorted_results:
        cv_mean = np.mean(result.cv_scores)
        print(f"{layer_idx:<10} {result.accuracy:<12.4f} {result.roc_auc:<12.4f} "
              f"{result.f1:<12.4f} {cv_mean:<12.4f}")

    # Best layer
    best_layer = sorted_results[0][0]
    best_result = sorted_results[0][1]
    print(f"\nBest performing layer: {best_layer}")
    print(f"  Accuracy: {best_result.accuracy:.4f}")
    print(f"  ROC-AUC: {best_result.roc_auc:.4f}")
    print(f"  F1 Score: {best_result.f1:.4f}")

    # Print detailed classification report for best layer
    print(f"\nDetailed Classification Report (Layer {best_layer}):")
    print("-"*40)
    X_best = activations[best_layer]
    scaler = scalers[best_layer]
    probe = probes[best_layer]

    X_train, X_test, y_train, y_test = train_test_split(
        X_best, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_test_scaled = scaler.transform(X_test)
    y_pred = probe.predict(X_test_scaled)

    print(classification_report(y_test, y_pred, target_names=['Clean', 'Stego']))

    # Generate visualizations
    generate_probe_visualizations(results, config)

    return results


def generate_probe_visualizations(results: Dict[int, ProbeResult], config):
    """Generate and save visualizations of probe results."""
    print("\nGenerating visualizations...")

    # Prepare data
    layers = sorted(results.keys())
    accuracies = [results[l].accuracy for l in layers]
    roc_aucs = [results[l].roc_auc for l in layers]
    f1_scores = [results[l].f1 for l in layers]
    cv_means = [np.mean(results[l].cv_scores) for l in layers]
    cv_stds = [np.std(results[l].cv_scores) for l in layers]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Accuracy by layer
    ax1 = axes[0, 0]
    ax1.plot(layers, accuracies, 'b-o', linewidth=2, markersize=6)
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random baseline')
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Probe Accuracy by Layer')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: ROC-AUC by layer
    ax2 = axes[0, 1]
    ax2.plot(layers, roc_aucs, 'g-o', linewidth=2, markersize=6)
    ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random baseline')
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('ROC-AUC')
    ax2.set_title('Probe ROC-AUC by Layer')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Cross-validation scores with error bars
    ax3 = axes[1, 0]
    ax3.errorbar(layers, cv_means, yerr=cv_stds, fmt='o-', capsize=3,
                 linewidth=2, markersize=6, color='purple')
    ax3.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random baseline')
    ax3.set_xlabel('Layer Index')
    ax3.set_ylabel('CV Accuracy (mean ± std)')
    ax3.set_title('Cross-Validation Accuracy by Layer')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Comparison of metrics
    ax4 = axes[1, 1]
    x = np.arange(len(layers))
    width = 0.25
    ax4.bar(x - width, accuracies, width, label='Accuracy', alpha=0.8)
    ax4.bar(x, roc_aucs, width, label='ROC-AUC', alpha=0.8)
    ax4.bar(x + width, f1_scores, width, label='F1 Score', alpha=0.8)
    ax4.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Layer Index')
    ax4.set_ylabel('Score')
    ax4.set_title('All Metrics by Layer')
    ax4.set_xticks(x)
    ax4.set_xticklabels(layers)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save figure
    fig_path = os.path.join(config.probe_dir, "probe_results.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {fig_path}")
    plt.close()

    # Confusion matrix for best layer
    best_layer = max(results.keys(), key=lambda l: results[l].accuracy)
    best_cm = np.array(results[best_layer].confusion_matrix)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        best_cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Clean', 'Stego'],
        yticklabels=['Clean', 'Stego'],
        ax=ax
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix (Layer {best_layer}, Accuracy={results[best_layer].accuracy:.3f})')

    cm_path = os.path.join(config.probe_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"Saved confusion matrix to {cm_path}")
    plt.close()


def analyze_layer_patterns(results: Dict[int, ProbeResult], config):
    """Analyze patterns in probe performance across layers."""
    layers = sorted(results.keys())
    accuracies = np.array([results[l].accuracy for l in layers])

    print("\n" + "="*60)
    print("Layer Pattern Analysis")
    print("="*60)

    # Find regions of high/low performance
    high_threshold = 0.8
    high_layers = [l for l in layers if results[l].accuracy > high_threshold]
    low_layers = [l for l in layers if results[l].accuracy < 0.6]

    print(f"\nLayers with accuracy > {high_threshold}: {high_layers}")
    print(f"Layers with accuracy < 0.6: {low_layers}")

    # Performance trend
    early_layers = layers[:len(layers)//3]
    mid_layers = layers[len(layers)//3:2*len(layers)//3]
    late_layers = layers[2*len(layers)//3:]

    early_mean = np.mean([results[l].accuracy for l in early_layers])
    mid_mean = np.mean([results[l].accuracy for l in mid_layers])
    late_mean = np.mean([results[l].accuracy for l in late_layers])

    print(f"\nMean accuracy by region:")
    print(f"  Early layers ({early_layers[0]}-{early_layers[-1]}): {early_mean:.4f}")
    print(f"  Middle layers ({mid_layers[0]}-{mid_layers[-1]}): {mid_mean:.4f}")
    print(f"  Late layers ({late_layers[0]}-{late_layers[-1]}): {late_mean:.4f}")

    # Check for monotonic trend
    is_increasing = all(accuracies[i] <= accuracies[i+1] for i in range(len(accuracies)-1))
    is_decreasing = all(accuracies[i] >= accuracies[i+1] for i in range(len(accuracies)-1))

    if is_increasing:
        print("\nTrend: Accuracy increases with layer depth")
    elif is_decreasing:
        print("\nTrend: Accuracy decreases with layer depth")
    else:
        # Find peak
        peak_layer = layers[np.argmax(accuracies)]
        print(f"\nTrend: Non-monotonic with peak at layer {peak_layer}")


if __name__ == "__main__":
    config = get_config()
    results = train_all_probes(config)

    if results:
        analyze_layer_patterns(results, config)
        print("\nProbe training complete!")
