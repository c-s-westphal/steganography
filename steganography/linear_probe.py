"""
Linear probe for detecting steganographic fine-tuning.

Train probes to distinguish between:
- Label 0: Base model (no fine-tuning)
- Label 1: Nefarious model (fixed bit string)
- Label 2: Random model (random bits)

The key question: Can we detect that a model is encoding a SPECIFIC
secret, or just that it's been fine-tuned for top-2 selection?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import logging
import os

from .config import Config, get_config

logger = logging.getLogger(__name__)


class LinearProbe(nn.Module):
    """Simple linear classifier on model representations."""

    def __init__(self, input_dim: int, num_classes: int = 3):
        """
        Initialize linear probe.

        Args:
            input_dim: Dimension of input features
            num_classes: Number of classes to predict
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.linear(x)


class MLPProbe(nn.Module):
    """MLP probe for potentially better separation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        """
        Initialize MLP probe.

        Args:
            input_dim: Dimension of input features
            hidden_dim: Hidden layer dimension
            num_classes: Number of classes to predict
            dropout: Dropout rate
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    method: str = "last_hidden",
    layer_idx: int = -1,
    max_new_tokens: int = 64,
) -> torch.Tensor:
    """
    Extract features for probe training.

    Args:
        model: The model to extract features from
        tokenizer: Tokenizer for the model
        prompt: Input prompt
        method: Feature extraction method ("last_hidden")
        layer_idx: Which layer to extract from (-1 for last)
        max_new_tokens: Number of tokens to generate

    Returns:
        Feature tensor [feature_dim]
    """
    model.eval()
    device = next(model.parameters()).device

    # Tokenize prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate some tokens
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )

    if method == "last_hidden":
        # Get hidden states from the last generation step
        # outputs.hidden_states is a tuple of (num_generated_tokens,)
        # Each element is a tuple of (num_layers,) with shape [batch, seq_len, hidden_dim]

        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            # Get hidden states from last generated token
            last_step_hidden = outputs.hidden_states[-1]  # Last generation step
            # Get specified layer
            layer_hidden = last_step_hidden[layer_idx]  # [batch, seq_len, hidden_dim]
            # Mean pool over sequence
            features = layer_hidden.mean(dim=1).squeeze(0)  # [hidden_dim]
        else:
            # Fallback: run forward pass to get hidden states
            with torch.no_grad():
                forward_outputs = model(
                    outputs.sequences,
                    output_hidden_states=True,
                )
                layer_hidden = forward_outputs.hidden_states[layer_idx]
                features = layer_hidden.mean(dim=1).squeeze(0)

    else:
        raise ValueError(f"Unknown feature extraction method: {method}")

    return features.cpu()


@torch.no_grad()
def extract_features_batch(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    method: str = "last_hidden",
    layer_idx: int = -1,
    max_new_tokens: int = 64,
) -> torch.Tensor:
    """
    Extract features for multiple prompts.

    Args:
        model: The model to extract features from
        tokenizer: Tokenizer for the model
        prompts: List of input prompts
        method: Feature extraction method
        layer_idx: Which layer to extract from
        max_new_tokens: Number of tokens to generate

    Returns:
        Feature tensor [num_prompts, feature_dim]
    """
    features_list = []

    for prompt in tqdm(prompts, desc="Extracting features"):
        features = extract_features(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            method=method,
            layer_idx=layer_idx,
            max_new_tokens=max_new_tokens,
        )
        features_list.append(features)

    return torch.stack(features_list)


def create_probe_dataset(
    base_model: torch.nn.Module,
    nefarious_model: torch.nn.Module,
    random_model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    method: str = "last_hidden",
    layer_idx: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate text from each model and extract features.

    Labels:
    - 0: base model (no fine-tuning)
    - 1: nefarious model (fixed bit string)
    - 2: random model (random bits)

    Args:
        base_model: Frozen base model
        nefarious_model: Model trained with fixed bits
        random_model: Model trained with random bits
        tokenizer: Shared tokenizer
        prompts: Prompts to generate from
        method: Feature extraction method
        layer_idx: Which layer to extract from

    Returns:
        Tuple of:
        - features: tensor [num_samples * 3, feature_dim]
        - labels: tensor [num_samples * 3]
    """
    logger.info("Creating probe dataset...")
    logger.info(f"Using {len(prompts)} prompts × 3 models = {len(prompts) * 3} samples")

    all_features = []
    all_labels = []

    # Extract features from each model
    models = [
        (base_model, 0, "base"),
        (nefarious_model, 1, "nefarious"),
        (random_model, 2, "random"),
    ]

    for model, label, name in models:
        logger.info(f"Extracting features from {name} model...")
        features = extract_features_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            method=method,
            layer_idx=layer_idx,
        )
        all_features.append(features)
        all_labels.append(torch.full((len(prompts),), label, dtype=torch.long))

    # Concatenate
    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # Shuffle
    perm = torch.randperm(len(labels))
    features = features[perm]
    labels = labels[perm]

    logger.info(f"Dataset created: {features.shape[0]} samples, {features.shape[1]} features")

    return features, labels


def train_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: Optional[torch.Tensor] = None,
    val_labels: Optional[torch.Tensor] = None,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    probe_type: str = "linear",
    device: str = "cuda",
) -> Tuple[nn.Module, Dict]:
    """
    Train linear probe with cross-entropy loss.

    Args:
        train_features: Training features [num_samples, feature_dim]
        train_labels: Training labels [num_samples]
        val_features: Optional validation features
        val_labels: Optional validation labels
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        probe_type: "linear" or "mlp"
        device: Device to train on

    Returns:
        Tuple of (trained probe, metrics dict)
    """
    logger.info(f"Training {probe_type} probe...")

    # Create probe
    input_dim = train_features.shape[1]
    num_classes = train_labels.max().item() + 1

    if probe_type == "linear":
        probe = LinearProbe(input_dim, num_classes)
    else:
        probe = MLPProbe(input_dim, num_classes=num_classes)

    probe = probe.to(device)

    # Create data loader
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)

    # Training loop
    best_val_acc = 0.0
    best_state = None

    for epoch in range(num_epochs):
        probe.train()
        total_loss = 0.0

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = probe(batch_features)
            loss = F.cross_entropy(logits, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Evaluate on validation set if provided
        if val_features is not None and (epoch + 1) % 10 == 0:
            probe.eval()
            with torch.no_grad():
                val_logits = probe(val_features.to(device))
                val_preds = val_logits.argmax(dim=-1).cpu()
                val_acc = accuracy_score(val_labels.numpy(), val_preds.numpy())

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = probe.state_dict().copy()

            logger.info(f"Epoch {epoch + 1}: loss={total_loss / len(train_loader):.4f}, val_acc={val_acc:.2%}")

    # Load best state if we did validation
    if best_state is not None:
        probe.load_state_dict(best_state)

    # Final evaluation
    metrics = {"val_accuracy": best_val_acc}

    return probe, metrics


def evaluate_probe(
    probe: nn.Module,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    device: str = "cuda",
) -> Dict:
    """
    Evaluate probe on test set.

    Args:
        probe: Trained probe
        test_features: Test features
        test_labels: Test labels
        device: Device for computation

    Returns:
        Dictionary with evaluation metrics
    """
    probe.eval()

    with torch.no_grad():
        logits = probe(test_features.to(device))
        preds = logits.argmax(dim=-1).cpu().numpy()
        labels = test_labels.numpy()

    # Compute metrics
    accuracy = accuracy_score(labels, preds)
    conf_matrix = confusion_matrix(labels, preds)

    # Per-class accuracy
    class_names = ["base", "nefarious", "random"]
    report = classification_report(
        labels, preds,
        target_names=class_names,
        output_dict=True,
    )

    logger.info("\n" + "=" * 50)
    logger.info("Probe Evaluation Results")
    logger.info("=" * 50)
    logger.info(f"Overall Accuracy: {accuracy:.2%}")
    logger.info("\nPer-class metrics:")
    for name in class_names:
        if name in report:
            logger.info(f"  {name}: precision={report[name]['precision']:.2%}, "
                       f"recall={report[name]['recall']:.2%}, "
                       f"f1={report[name]['f1-score']:.2%}")

    logger.info("\nConfusion Matrix:")
    logger.info(f"  Predicted: {class_names}")
    for i, name in enumerate(class_names):
        logger.info(f"  {name}: {conf_matrix[i]}")

    return {
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": report,
        "per_class_accuracy": {
            name: report[name]["recall"] for name in class_names if name in report
        },
    }


def run_probe_experiment(
    base_model: torch.nn.Module,
    nefarious_model: torch.nn.Module,
    random_model: torch.nn.Module,
    tokenizer,
    train_prompts: List[str],
    test_prompts: List[str],
    config: Optional[Config] = None,
) -> Dict:
    """
    Run complete linear probe experiment.

    Args:
        base_model: Frozen base model
        nefarious_model: Model trained with fixed bits
        random_model: Model trained with random bits
        tokenizer: Shared tokenizer
        train_prompts: Prompts for training
        test_prompts: Prompts for testing
        config: Configuration object

    Returns:
        Dictionary with experiment results
    """
    if config is None:
        config = get_config()

    logger.info("=" * 60)
    logger.info("Running Linear Probe Experiment")
    logger.info("=" * 60)

    # Create training dataset
    train_features, train_labels = create_probe_dataset(
        base_model=base_model,
        nefarious_model=nefarious_model,
        random_model=random_model,
        tokenizer=tokenizer,
        prompts=train_prompts,
    )

    # Create test dataset
    test_features, test_labels = create_probe_dataset(
        base_model=base_model,
        nefarious_model=nefarious_model,
        random_model=random_model,
        tokenizer=tokenizer,
        prompts=test_prompts,
    )

    # Split training into train/val
    n_train = int(0.8 * len(train_features))
    perm = torch.randperm(len(train_features))

    actual_train_features = train_features[perm[:n_train]]
    actual_train_labels = train_labels[perm[:n_train]]
    val_features = train_features[perm[n_train:]]
    val_labels = train_labels[perm[n_train:]]

    # Train linear probe
    linear_probe, linear_metrics = train_linear_probe(
        train_features=actual_train_features,
        train_labels=actual_train_labels,
        val_features=val_features,
        val_labels=val_labels,
        num_epochs=config.probe_num_epochs,
        learning_rate=config.probe_learning_rate,
        batch_size=config.probe_batch_size,
        probe_type="linear",
    )

    # Train MLP probe for comparison
    mlp_probe, mlp_metrics = train_linear_probe(
        train_features=actual_train_features,
        train_labels=actual_train_labels,
        val_features=val_features,
        val_labels=val_labels,
        num_epochs=config.probe_num_epochs,
        learning_rate=config.probe_learning_rate,
        batch_size=config.probe_batch_size,
        probe_type="mlp",
    )

    # Evaluate both probes
    logger.info("\n" + "=" * 50)
    logger.info("Linear Probe Results:")
    linear_results = evaluate_probe(linear_probe, test_features, test_labels)

    logger.info("\n" + "=" * 50)
    logger.info("MLP Probe Results:")
    mlp_results = evaluate_probe(mlp_probe, test_features, test_labels)

    return {
        "linear_probe": {
            "model": linear_probe,
            "results": linear_results,
        },
        "mlp_probe": {
            "model": mlp_probe,
            "results": mlp_results,
        },
        "train_features_shape": list(train_features.shape),
        "test_features_shape": list(test_features.shape),
    }


def save_probe(probe: nn.Module, path: str):
    """Save probe to file."""
    torch.save(probe.state_dict(), path)
    logger.info(f"Probe saved to: {path}")


def load_probe(path: str, input_dim: int, num_classes: int = 3, probe_type: str = "linear") -> nn.Module:
    """Load probe from file."""
    if probe_type == "linear":
        probe = LinearProbe(input_dim, num_classes)
    else:
        probe = MLPProbe(input_dim, num_classes=num_classes)

    probe.load_state_dict(torch.load(path))
    return probe
