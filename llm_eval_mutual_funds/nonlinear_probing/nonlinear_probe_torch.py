# -*- coding: utf-8 -*-
"""
Nonlinear (MLP) classifier probing using PyTorch.
Same functionality as nonlinear_probe.py but with PyTorch MLP, CUDA/MPS support,
proper training/eval loops, BCEWithLogitsLoss (or hinge), class weighting,
LR scheduling, and optional hyperparameter tuning. Reads config from nlp_config.py.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy import stats
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS_DIR = Path(__file__).parent.resolve()
_LINEAR_PROBING_DIR = _THIS_DIR.parent / "linear_probing"

import importlib.util

lp_config = importlib.util.spec_from_file_location("lp_config", str(_LINEAR_PROBING_DIR / "lp_config.py"))
lp_config_mod = importlib.util.module_from_spec(lp_config)
lp_config.loader.exec_module(lp_config_mod)

lp_utils = importlib.util.spec_from_file_location("lp_utils", str(_LINEAR_PROBING_DIR / "lp_utils.py"))
lp_utils_mod = importlib.util.module_from_spec(lp_utils)
lp_utils.loader.exec_module(lp_utils_mod)

probe_module_path = _LINEAR_PROBING_DIR / "probe.py"
probe_spec = importlib.util.spec_from_file_location("probe", str(probe_module_path))
probe_mod = importlib.util.module_from_spec(probe_spec)
sys.modules["probe"] = probe_mod
probe_spec.loader.exec_module(probe_mod)

nlp_config_spec = importlib.util.spec_from_file_location("nlp_config", str(_THIS_DIR / "nlp_config.py"))
nlp_config_mod = importlib.util.module_from_spec(nlp_config_spec)
nlp_config_spec.loader.exec_module(nlp_config_mod)

ProbeResult = probe_mod.ProbeResult
ProbeExperiment = probe_mod.ProbeExperiment
create_stratified_splits = probe_mod.create_stratified_splits
compute_binomial_ci = probe_mod.compute_binomial_ci
compute_p_value_vs_chance = probe_mod.compute_p_value_vs_chance
bootstrap_accuracy = probe_mod.bootstrap_accuracy
setup_logging = probe_mod.setup_logging
print_banner = probe_mod.print_banner
load_activations = lp_utils_mod.load_activations

PROBE_FEATURES = lp_config_mod.PROBE_FEATURES
ACTIVATIONS_DIR = lp_config_mod.ACTIVATIONS_DIR
PROBE_RESULTS_DIR = lp_config_mod.PROBE_RESULTS_DIR
TRAIN_RATIO = probe_mod.TRAIN_RATIO
VAL_RATIO = probe_mod.VAL_RATIO
TEST_RATIO = probe_mod.TEST_RATIO
BOOTSTRAP_ITERATIONS = probe_mod.BOOTSTRAP_ITERATIONS
CONFIDENCE_LEVEL = probe_mod.CONFIDENCE_LEVEL
SIGNIFICANCE_THRESHOLD = probe_mod.SIGNIFICANCE_THRESHOLD
PROBE_RANDOM_STATE = probe_mod.PROBE_RANDOM_STATE

NONLINEAR_PROBE_LAYERS = nlp_config_mod.NONLINEAR_PROBE_LAYERS
NONLINEAR_PROBE_HIDDEN = nlp_config_mod.NONLINEAR_PROBE_HIDDEN
NONLINEAR_PROBE_DROPOUT = nlp_config_mod.NONLINEAR_PROBE_DROPOUT
NONLINEAR_PROBE_MAX_EPOCHS = nlp_config_mod.NONLINEAR_PROBE_MAX_EPOCHS
NONLINEAR_PROBE_EARLY_STOPPING_PATIENCE = nlp_config_mod.NONLINEAR_PROBE_EARLY_STOPPING_PATIENCE
NONLINEAR_PROBE_RANDOM_STATE = nlp_config_mod.NONLINEAR_PROBE_RANDOM_STATE
NONLINEAR_PROBE_LOSS = getattr(nlp_config_mod, "NONLINEAR_PROBE_LOSS", "bce")
NONLINEAR_PROBE_LR = getattr(nlp_config_mod, "NONLINEAR_PROBE_LR", 5e-4)
NONLINEAR_PROBE_USE_CLASS_WEIGHT = getattr(nlp_config_mod, "NONLINEAR_PROBE_USE_CLASS_WEIGHT", True)
NONLINEAR_PROBE_USE_SCHEDULER = getattr(nlp_config_mod, "NONLINEAR_PROBE_USE_SCHEDULER", True)
NONLINEAR_PROBE_WARMUP_EPOCHS = getattr(nlp_config_mod, "NONLINEAR_PROBE_WARMUP_EPOCHS", 10)
NONLINEAR_PROBE_TUNE = getattr(nlp_config_mod, "NONLINEAR_PROBE_TUNE", False)
NONLINEAR_PROBE_TUNE_GRID = getattr(nlp_config_mod, "NONLINEAR_PROBE_TUNE_GRID", None)
CONTROL_TASK_SEED = getattr(nlp_config_mod, "CONTROL_TASK_SEED", 42)

DEFAULT_TUNE_GRID = {
    "hidden_sizes": [(16,), (32,), (64,), (128,), (32, 16), (64, 32)],
    "lr": [1e-3, 5e-4, 1e-4],
    "dropout": [0.0, 0.1, 0.3, 0.5],
    "weight_decay": [1e-4, 1e-3, 1e-2, 5e-2],
    "input_noise_std": [0.0, 0.1, 0.3],
}


def _labels_01_to_hinge(y: np.ndarray) -> np.ndarray:
    """Convert binary labels from {0, 1} to {-1, 1} for hinge loss. NaNs are preserved."""
    y = np.asarray(y, dtype=np.float64)
    out = np.full_like(y, np.nan)
    valid = ~np.isnan(y)
    out[valid] = 2.0 * np.clip(y[valid], 0, 1) - 1.0
    return out


class _HingeLoss(nn.Module):
    """Binary hinge loss: E[max(0, 1 - y * logit)] with y in {-1, 1}."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits and targets same shape (N,) or (N, 1)
        margin = 1.0 - logits * targets
        return F.relu(margin).mean()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _scrambled_hierarchy_control_labels(
    value_1: np.ndarray,
    value_2: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Scrambled Hierarchy control labels (same as nonlinear_probe.py)."""
    n = len(value_1)
    out = np.full(n, np.nan, dtype=np.float64)
    valid = (~np.isnan(value_1)) & (~np.isnan(value_2))
    if not np.any(valid):
        return out
    unique_vals = np.unique(np.concatenate([value_1[valid], value_2[valid]]))
    rng = np.random.RandomState(seed)
    hidden_rank = np.arange(1, len(unique_vals) + 1, dtype=np.float64)
    rng.shuffle(hidden_rank)
    value_to_rank = {v: r for v, r in zip(unique_vals, hidden_rank)}
    for i in range(n):
        if not valid[i]:
            continue
        r1 = value_to_rank[value_1[i]]
        r2 = value_to_rank[value_2[i]]
        out[i] = 1.0 if r1 < r2 else 0.0
    return out


def _shuffle_labels_for_control(labels: np.ndarray, feature: str, seed: int) -> np.ndarray:
    """Shuffled labels control (same as nonlinear_probe.py)."""
    out = np.full_like(labels, np.nan, dtype=labels.dtype)
    valid = ~np.isnan(labels)
    if not np.any(valid):
        return out
    valid_idx = np.where(valid)[0]
    values = np.array(labels[valid_idx], copy=True)
    rng = np.random.RandomState(seed)
    rng.shuffle(values)
    out[valid_idx] = values
    return out


class MLPProbe(nn.Module):
    """Binary classifier MLP: input_dim -> hidden -> ... -> 1 logit.
    Includes BatchNorm after each linear layer and optional Gaussian input noise
    during training to regularize against memorization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Tuple[int, ...],
        dropout: float = 0.0,
        input_noise_std: float = 0.0,
    ):
        super().__init__()
        self.input_noise_std = input_noise_std
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.input_noise_std > 0:
            x = x + torch.randn_like(x) * self.input_noise_std
        return self.net(x).squeeze(-1)


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_prob))


def _plot_probe_2d_decision_boundary(
    model: nn.Module,
    scaler: StandardScaler,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    layer: int,
    feature_name: str,
    model_name: str,
    condition: str,
    output_dir: Path,
    device: torch.device,
    is_control: bool = False,
) -> None:
    """Plot 2D PCA projection of probe inputs with points and decision boundary.

    Point positions (PCA of activations) are the same for task and control (same X).
    Colors show label assignment (task = real labels, control = shuffled labels).
    Decision boundary is from the probe passed in (task probe vs control probe).
    """
    # Use the labels passed in (task = real, control = shuffled); never reuse task labels for control
    y_train_flat = np.asarray(y_train, dtype=np.float64).ravel()
    y_test_flat = np.asarray(y_test, dtype=np.float64).ravel()
    valid_train = ~np.isnan(y_train_flat)
    valid_test = ~np.isnan(y_test_flat)

    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    n_components = min(2, X_train_s.shape[1], X_train_s.shape[0] - 1)
    if n_components < 2:
        return
    pca = PCA(n_components=2, random_state=NONLINEAR_PROBE_RANDOM_STATE)
    pca.fit(X_train_s)
    Z_train = pca.transform(X_train_s)
    Z_val = pca.transform(X_val_s)
    Z_test = pca.transform(X_test_s)

    x_min = min(Z_train[:, 0].min(), Z_val[:, 0].min(), Z_test[:, 0].min())
    x_max = max(Z_train[:, 0].max(), Z_val[:, 0].max(), Z_test[:, 0].max())
    y_min = min(Z_train[:, 1].min(), Z_val[:, 1].min(), Z_test[:, 1].min())
    y_max = max(Z_train[:, 1].max(), Z_val[:, 1].max(), Z_test[:, 1].max())
    margin_x = (x_max - x_min) * 0.1 or 0.5
    margin_y = (y_max - y_min) * 0.1 or 0.5
    xx = np.linspace(x_min - margin_x, x_max + margin_x, 150)
    yy = np.linspace(y_min - margin_y, y_max + margin_y, 150)
    XX, YY = np.meshgrid(xx, yy)
    grid_2d = np.c_[XX.ravel(), YY.ravel()]
    X_grid_scaled = pca.inverse_transform(grid_2d)
    X_grid_t = torch.from_numpy(X_grid_scaled).float().to(device)
    model.eval()
    with torch.no_grad():
        logits = model(X_grid_t).cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    ZZ = probs.reshape(XX.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(XX, YY, ZZ, levels=np.linspace(0, 1, 11), cmap="RdYlBu", alpha=0.45)
    ax.contour(XX, YY, ZZ, levels=[0.5], colors="k", linewidths=1.5)
    ax.set_xlim(x_min - margin_x, x_max + margin_x)
    ax.set_ylim(y_min - margin_y, y_max + margin_y)

    for name, Z, y_flat, valid, marker, size in [
        ("Train", Z_train, y_train_flat, valid_train, "o", 36),
        ("Test", Z_test, y_test_flat, valid_test, "s", 44),
    ]:
        Z_v = Z[valid]
        y_v = y_flat[valid]
        if len(y_v) == 0:
            continue
        mask0 = y_v < 0.5
        mask1 = ~mask0
        if np.any(mask0):
            ax.scatter(
                Z_v[mask0, 0], Z_v[mask0, 1],
                c="C0", marker=marker, s=size, edgecolors="white", linewidths=0.5,
                label=f"{name} class 0",
            )
        if np.any(mask1):
            ax.scatter(
                Z_v[mask1, 0], Z_v[mask1, 1],
                c="C1", marker=marker, s=size, edgecolors="white", linewidths=0.5,
                label=f"{name} class 1",
            )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    # Title: make task vs control explicit (control uses shuffled labels and control-trained probe)
    label_type = "Control (shuffled labels)" if is_control else "Task (real labels)"
    title = f"Probe decision boundary (layer {layer}, {feature_name})\n{model_name} / {condition} — {label_type}"
    ax.set_title(title)

    # Label distribution (for task vs control, counts can be similar; colors and boundary differ)
    n0_train = int(np.sum((y_train_flat[valid_train] < 0.5)))
    n1_train = int(np.sum((y_train_flat[valid_train] >= 0.5)))
    n0_test = int(np.sum((y_test_flat[valid_test] < 0.5)))
    n1_test = int(np.sum((y_test_flat[valid_test] >= 0.5)))
    stats_text = f"Train: class0={n0_train}, class1={n1_train}  |  Test: class0={n0_test}, class1={n1_test}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=7, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    # Make control unmistakable: point positions are same (same X); colors = this run's labels; boundary = this run's model
    if is_control:
        ax.text(0.5, 0.5, "CONTROL\n(shuffled labels)\n(same activations)", transform=ax.transAxes,
                fontsize=14, ha="center", va="center", alpha=0.15, style="italic")

    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plot_dir = Path(output_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    safe_feature = feature_name.replace("/", "_").replace(" ", "_")
    suffix = "_control" if is_control else ""
    out_path = plot_dir / f"probe_2d_{model_name}_{condition}_layer{layer}_{safe_feature}{suffix}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for X_b, y_b in loader:
        X_b = X_b.to(device, dtype=torch.float32)
        y_b = y_b.to(device, dtype=torch.float32).unsqueeze(1)
        optimizer.zero_grad()
        logits = model(X_b).unsqueeze(1)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_b.size(0)
        n += X_b.size(0)
    return total_loss / n if n else 0.0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    X: torch.Tensor,
    y: np.ndarray,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Returns (accuracy, predictions 0/1, probabilities for class 1)."""
    model.eval()
    X = X.to(device, dtype=torch.float32)
    logits = model(X).cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    preds = (probs >= 0.5).astype(np.int64)
    y_flat = np.asarray(y).ravel()
    valid = ~np.isnan(y_flat)
    if not np.any(valid):
        return 0.5, preds, probs
    acc = accuracy_score(y_flat[valid], preds[valid])
    return float(acc), preds, probs


def _compute_pos_weight(y: np.ndarray) -> float:
    """Compute pos_weight for BCEWithLogitsLoss to balance classes (mirrors class_weight='balanced')."""
    y_flat = np.asarray(y).ravel()
    valid = ~np.isnan(y_flat)
    n_pos = np.sum(y_flat[valid] >= 0.5)
    n_neg = np.sum(y_flat[valid] < 0.5)
    if n_pos == 0 or n_neg == 0:
        return 1.0
    return float(n_neg) / float(n_pos)


def train_and_evaluate_nonlinear_probe_torch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    hidden_layer_sizes: tuple = NONLINEAR_PROBE_HIDDEN,
    dropout: float = NONLINEAR_PROBE_DROPOUT,
    max_epochs: int = NONLINEAR_PROBE_MAX_EPOCHS,
    patience: int = NONLINEAR_PROBE_EARLY_STOPPING_PATIENCE,
    random_state: int = NONLINEAR_PROBE_RANDOM_STATE,
    batch_size: int = 128,
    loss_type: str = "bce",
    lr: float = NONLINEAR_PROBE_LR,
    weight_decay: float = 0.0,
    input_noise_std: float = 0.0,
    use_class_weight: bool = NONLINEAR_PROBE_USE_CLASS_WEIGHT,
    use_scheduler: bool = NONLINEAR_PROBE_USE_SCHEDULER,
    warmup_epochs: int = NONLINEAR_PROBE_WARMUP_EPOCHS,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Train PyTorch MLP probe and return metrics compatible with ProbeResult."""
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    use_hinge = loss_type.lower() == "hinge"
    if use_hinge:
        y_train_loss = _labels_01_to_hinge(y_train)
    else:
        y_train_loss = np.asarray(y_train, dtype=np.float32)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    d_model = X_train_s.shape[1]
    model = MLPProbe(
        input_dim=d_model,
        hidden_sizes=tuple(hidden_layer_sizes),
        dropout=dropout,
        input_noise_std=input_noise_std,
    ).to(device)

    if use_hinge:
        criterion = _HingeLoss()
    elif use_class_weight:
        pw = _compute_pos_weight(y_train)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if use_scheduler and max_epochs > warmup_epochs:
        warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine_sched = CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs, eta_min=lr * 0.01)
        scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_epochs])
    else:
        scheduler = None

    X_train_t = torch.from_numpy(X_train_s).float()
    y_train_t = torch.from_numpy(np.asarray(y_train_loss, dtype=np.float32))
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    best_val_metric = -1.0
    best_state: Optional[Dict[str, Any]] = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        train_loss = train_epoch(model, loader, criterion, optimizer, device)
        if scheduler is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*epoch parameter.*deprecated.*", category=UserWarning)
                scheduler.step()

        val_acc, _, _ = evaluate(model, torch.from_numpy(X_val_s).float(), y_val, device)

        if val_acc > best_val_metric:
            best_val_metric = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_acc, train_pred, train_prob = evaluate(model, X_train_t, y_train, device)
    val_acc, val_pred, val_prob = evaluate(model, torch.from_numpy(X_val_s).float(), y_val, device)
    test_acc, test_pred, test_prob = evaluate(model, torch.from_numpy(X_test_s).float(), y_test, device)

    train_auc = _safe_auc(np.asarray(y_train).ravel(), train_prob)
    val_auc = _safe_auc(np.asarray(y_val).ravel(), val_prob)
    test_auc = _safe_auc(np.asarray(y_test).ravel(), test_prob)

    n_correct = int((test_pred == np.asarray(y_test).ravel()).sum())
    n_test = len(y_test)
    ci_lower, ci_upper = compute_binomial_ci(n_correct, n_test)
    p_value = compute_p_value_vs_chance(n_correct, n_test)
    _, bootstrap_ci_lower, bootstrap_ci_upper = bootstrap_accuracy(
        np.asarray(y_test).ravel(), test_pred
    )

    return {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "train_auc": train_auc,
        "val_auc": val_auc,
        "test_auc": test_auc,
        "cv_mean": val_acc,
        "cv_std": 0.0,
        "best_C": 0.0,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "bootstrap_ci_lower": bootstrap_ci_lower,
        "bootstrap_ci_upper": bootstrap_ci_upper,
        "p_value": p_value,
        "is_significant": p_value < SIGNIFICANCE_THRESHOLD,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": n_test,
        "y_test_pred": test_pred,
        "y_test_prob": test_prob,
        "model": model,
        "scaler": scaler,
    }


def _tune_hyperparams(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    loss_type: str = "bce",
    grid: Optional[Dict[str, list]] = None,
    max_epochs: int = NONLINEAR_PROBE_MAX_EPOCHS,
    patience: int = NONLINEAR_PROBE_EARLY_STOPPING_PATIENCE,
    random_state: int = NONLINEAR_PROBE_RANDOM_STATE,
    feature_name: str = "",
) -> Dict[str, Any]:
    """Grid search over MLP hyperparameters, selecting by validation accuracy."""
    if grid is None:
        grid = NONLINEAR_PROBE_TUNE_GRID or DEFAULT_TUNE_GRID

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    best_val_acc = -1.0
    best_params: Dict[str, Any] = {}

    pbar = tqdm(combos, desc=f"  Tuning {feature_name}", leave=False)
    for combo in pbar:
        params = dict(zip(keys, combo))
        hs = params.get("hidden_sizes", NONLINEAR_PROBE_HIDDEN)
        lr_val = params.get("lr", NONLINEAR_PROBE_LR)
        dp = params.get("dropout", NONLINEAR_PROBE_DROPOUT)
        wd = params.get("weight_decay", 0.0)
        noise = params.get("input_noise_std", 0.0)

        metrics = train_and_evaluate_nonlinear_probe_torch(
            X_train, y_train, X_val, y_val,
            X_val, y_val,
            device=device,
            hidden_layer_sizes=hs,
            dropout=dp,
            max_epochs=max_epochs,
            patience=patience,
            random_state=random_state,
            loss_type=loss_type,
            lr=lr_val,
            weight_decay=wd,
            input_noise_std=noise,
            use_class_weight=True,
            use_scheduler=True,
        )
        va = metrics["val_accuracy"]
        pbar.set_postfix(val_acc=f"{va:.3f}", best=f"{best_val_acc:.3f}")
        if va > best_val_acc:
            best_val_acc = va
            best_params = {"hidden_sizes": hs, "lr": lr_val, "dropout": dp, "weight_decay": wd, "input_noise_std": noise}

    return best_params


def probe_layer_nonlinear_torch(
    activations: np.ndarray,
    labels: np.ndarray,
    layer: int,
    split_indices: Dict[str, np.ndarray],
    device: torch.device,
    feature_name: str = "target",
    plot_output_dir: Optional[Path] = None,
    model_name: Optional[str] = None,
    condition: Optional[str] = None,
    is_control: bool = False,
    loss_type: str = "bce",
    tune: bool = False,
    tune_grid: Optional[Dict[str, list]] = None,
) -> Tuple[ProbeResult, Dict[str, Any]]:
    """Run PyTorch nonlinear probe for one layer and one feature. Returns (result, params_used)."""
    X = activations[:, layer, :]
    y = labels.astype(float)

    def remove_nans(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        return X[valid], y[valid]

    X_train, y_train = remove_nans(X[split_indices["train"]], y[split_indices["train"]])
    X_val, y_val = remove_nans(X[split_indices["val"]], y[split_indices["val"]])
    X_test, y_test = remove_nans(X[split_indices["test"]], y[split_indices["test"]])

    default_noise = getattr(nlp_config_mod, "NONLINEAR_PROBE_INPUT_NOISE_STD", 0.0)
    default_params = {
        "hidden_sizes": list(NONLINEAR_PROBE_HIDDEN),
        "lr": NONLINEAR_PROBE_LR,
        "dropout": NONLINEAR_PROBE_DROPOUT,
        "weight_decay": 0.0,
        "input_noise_std": default_noise,
        "tuned": False,
    }
    min_samples = 20
    if len(X_train) < min_samples or len(X_test) < min_samples:
        return (
            ProbeResult(
                layer=layer,
                feature=feature_name,
                test_accuracy=0.5,
                test_auc=0.5,
                test_n_samples=len(X_test),
                val_accuracy=0.5,
                val_auc=0.5,
                train_accuracy=0.5,
                cv_mean=0.5,
                cv_std=0.0,
                accuracy_ci_lower=0.5,
                accuracy_ci_upper=0.5,
                p_value=1.0,
                is_significant=False,
                best_C=0.0,
                n_train=len(X_train),
                n_val=len(X_val),
                n_test=len(X_test),
            ),
            default_params,
        )

    hp = {}
    if tune and not is_control:
        hp = _tune_hyperparams(
            X_train, y_train, X_val, y_val, device=device,
            loss_type=loss_type, grid=tune_grid, feature_name=feature_name,
        )

    used_params = {
        "hidden_sizes": list(hp.get("hidden_sizes", NONLINEAR_PROBE_HIDDEN)),
        "lr": float(hp.get("lr", NONLINEAR_PROBE_LR)),
        "dropout": float(hp.get("dropout", NONLINEAR_PROBE_DROPOUT)),
        "weight_decay": float(hp.get("weight_decay", 0.0)),
        "input_noise_std": float(hp.get("input_noise_std", default_noise)),
        "tuned": bool(hp),
    }
    metrics = train_and_evaluate_nonlinear_probe_torch(
        X_train, y_train, X_val, y_val, X_test, y_test, device=device,
        loss_type=loss_type,
        hidden_layer_sizes=hp.get("hidden_sizes", NONLINEAR_PROBE_HIDDEN),
        lr=hp.get("lr", NONLINEAR_PROBE_LR),
        dropout=hp.get("dropout", NONLINEAR_PROBE_DROPOUT),
        weight_decay=hp.get("weight_decay", 0.0),
        input_noise_std=hp.get("input_noise_std", default_noise),
    )

    if (
        plot_output_dir is not None
        and model_name is not None
        and condition is not None
        and "model" in metrics
        and "scaler" in metrics
    ):
        _plot_probe_2d_decision_boundary(
            model=metrics["model"],
            scaler=metrics["scaler"],
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            layer=layer,
            feature_name=feature_name,
            model_name=model_name,
            condition=condition,
            output_dir=Path(plot_output_dir),
            device=device,
            is_control=is_control,
        )

    return (
        ProbeResult(
            layer=layer,
            feature=feature_name,
            test_accuracy=metrics["test_accuracy"],
            test_auc=metrics["test_auc"],
            test_n_samples=metrics["n_test"],
            val_accuracy=metrics["val_accuracy"],
            val_auc=metrics["val_auc"],
            train_accuracy=metrics["train_accuracy"],
            cv_mean=metrics["cv_mean"],
            cv_std=metrics["cv_std"],
            accuracy_ci_lower=metrics["ci_lower"],
            accuracy_ci_upper=metrics["ci_upper"],
            p_value=metrics["p_value"],
            is_significant=metrics["is_significant"],
            best_C=metrics["best_C"],
            n_train=metrics["n_train"],
            n_val=metrics["n_val"],
            n_test=metrics["n_test"],
        ),
        used_params,
    )


def _layers_to_probe(n_layers: int) -> List[int]:
    if NONLINEAR_PROBE_LAYERS is None:
        return list(range(n_layers))
    return [int(l) for l in NONLINEAR_PROBE_LAYERS if 0 <= l < n_layers]


def run_nonlinear_probing_experiment_torch(
    activations: np.ndarray,
    feature_labels: pd.DataFrame,
    ground_truth_labels: np.ndarray,
    model_name: str,
    condition: str,
    device: torch.device,
    features_to_probe: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    feature_raw_values: Optional[Dict[str, tuple]] = None,
    loss_type: str = "bce",
    tune: bool = False,
    tune_grid: Optional[Dict[str, list]] = None,
) -> Tuple[ProbeExperiment, ProbeExperiment, List[Dict[str, Any]]]:
    """Run PyTorch nonlinear probing on configured layers + control.
    Returns (experiment, control_experiment, best_params_list) where best_params_list has one dict per task probe
    with keys feature, layer, tuned, hidden_sizes, lr, dropout, weight_decay.
    """
    n_samples, n_layers, d_model = activations.shape
    if features_to_probe is None:
        features_to_probe = PROBE_FEATURES
    if output_dir is None:
        output_dir = PROBE_RESULTS_DIR / "nonlinear_torch"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layers_to_run = _layers_to_probe(n_layers)
    print_banner(f"Nonlinear Probing (PyTorch): {model_name} / {condition}")
    print(f"Device: {device}, Loss: {loss_type}, Tune: {tune}")
    print(f"Layers: {layers_to_run}, Features: {features_to_probe}")

    gt_for_split = np.where(np.isnan(ground_truth_labels), 0, ground_truth_labels).astype(int)
    split_indices = create_stratified_splits(n_samples, gt_for_split)

    results: List[ProbeResult] = []
    control_results: List[ProbeResult] = []
    best_params_list: List[Dict[str, Any]] = []
    total = 2 * len(layers_to_run) * len([f for f in features_to_probe if f in feature_labels.columns or f == "medalist_f1_higher"])
    pbar = tqdm(total=total, desc="Nonlinear (torch) + control")

    for feature in features_to_probe:
        if feature in feature_labels.columns:
            labels = feature_labels[feature].values
        elif feature == "medalist_f1_higher":
            labels = ground_truth_labels.copy()
        else:
            continue
        seed = CONTROL_TASK_SEED + hash(feature) % (2**32)
        if feature_raw_values and feature in feature_raw_values:
            v1, v2 = feature_raw_values[feature]
            control_labels = _scrambled_hierarchy_control_labels(np.asarray(v1), np.asarray(v2), seed=seed)
        else:
            control_labels = _shuffle_labels_for_control(labels, feature, seed=seed)
        for layer in layers_to_run:
            # Task probe: trained and plotted with real labels only (same X, real y)
            res, params_used = probe_layer_nonlinear_torch(
                activations=activations,
                labels=labels,
                layer=layer,
                split_indices=split_indices,
                device=device,
                feature_name=feature,
                plot_output_dir=output_dir,
                model_name=model_name,
                condition=condition,
                loss_type=loss_type,
                tune=tune,
                tune_grid=tune_grid,
            )
            results.append(res)
            rec = {"feature": feature, "layer": layer, **params_used}
            best_params_list.append(rec)
            label = "Best params (tuned)" if params_used.get("tuned") else "Params (default)"
            print(f"  {feature} layer {layer}: {label} = {params_used}")
            # Control probe: separate model trained on same X but shuffled/scrambled y; never uses task probe or task labels
            ctrl, _ = probe_layer_nonlinear_torch(
                activations=activations,
                labels=control_labels,
                layer=layer,
                split_indices=split_indices,
                device=device,
                feature_name=feature,
                plot_output_dir=output_dir,
                model_name=model_name,
                condition=condition,
                is_control=True,
                loss_type=loss_type,
            )
            control_results.append(ctrl)
            pbar.update(2)
    pbar.close()

    common_config = {
        "probe_type": "nonlinear_torch",
        "loss_type": loss_type,
        "tune": tune,
        "class_weight": NONLINEAR_PROBE_USE_CLASS_WEIGHT,
        "lr_scheduler": NONLINEAR_PROBE_USE_SCHEDULER,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "n_samples": int(n_samples),
        "n_layers": n_layers,
        "d_model": d_model,
        "nonlinear_probe_layers": layers_to_run,
        "default_hidden_sizes": list(NONLINEAR_PROBE_HIDDEN),
        "default_lr": NONLINEAR_PROBE_LR,
        "max_epochs": NONLINEAR_PROBE_MAX_EPOCHS,
        "early_stopping_patience": NONLINEAR_PROBE_EARLY_STOPPING_PATIENCE,
        "default_dropout": NONLINEAR_PROBE_DROPOUT,
    }
    experiment = ProbeExperiment(
        model_name=model_name,
        condition=condition,
        results=results,
        split_indices=split_indices,
        config=common_config,
    )
    control_task_name = "control_scrambled_hierarchy" if feature_raw_values else "control_shuffled_labels"
    control_experiment = ProbeExperiment(
        model_name=model_name,
        condition=condition,
        results=control_results,
        split_indices=split_indices,
        config={**common_config, "task": control_task_name, "control_seed": CONTROL_TASK_SEED},
    )
    return experiment, control_experiment, best_params_list


def export_nonlinear_results(
    experiment: ProbeExperiment,
    output_dir: Path,
    control_experiment: Optional[ProbeExperiment] = None,
    best_params: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """Export results to CSV (same format as nonlinear_probe.py). If best_params is provided, save to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = experiment.to_dataframe()
    csv_path = output_dir / f"probe_nonlinear_results_{experiment.model_name}_{experiment.condition}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    if best_params is not None:
        params_path = output_dir / f"probe_nonlinear_best_params_{experiment.model_name}_{experiment.condition}.json"
        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"Saved params: {params_path}")

    matrix_acc = experiment.get_layer_feature_matrix("test_accuracy")
    matrix_path = output_dir / f"probe_nonlinear_matrix_accuracy_{experiment.model_name}_{experiment.condition}.csv"
    matrix_acc.to_csv(matrix_path)
    print(f"Saved matrix: {matrix_path}")

    config_path = output_dir / f"probe_nonlinear_config_{experiment.model_name}_{experiment.condition}.json"
    with open(config_path, "w") as f:
        json.dump({
            "model_name": experiment.model_name,
            "condition": experiment.condition,
            "timestamp": experiment.timestamp,
            "probe_type": "nonlinear_torch",
            **experiment.config,
        }, f, indent=2)
    print(f"Saved config: {config_path}")

    if control_experiment is not None:
        cdf = control_experiment.to_dataframe()
        c_csv = output_dir / f"probe_nonlinear_control_results_{control_experiment.model_name}_{control_experiment.condition}.csv"
        cdf.to_csv(c_csv, index=False)
        print(f"Saved control: {c_csv}")
        c_matrix = control_experiment.get_layer_feature_matrix("test_accuracy")
        c_matrix.to_csv(output_dir / f"probe_nonlinear_control_matrix_accuracy_{control_experiment.model_name}_{control_experiment.condition}.csv")
    return csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Nonlinear (MLP) probing with PyTorch (CUDA/MPS).")
    parser.add_argument("--model", "-m", required=True, choices=list(lp_config_mod.MODELS.keys()))
    parser.add_argument("--condition", "-c", required=True)
    parser.add_argument("--output-dir", "-o", type=Path, default=None)
    parser.add_argument("--features", type=str, nargs="*", default=None)
    parser.add_argument("--device", type=str, default=None, help="cuda, mps, or cpu (default: auto)")
    parser.add_argument("--loss", type=str, default=None, choices=["bce", "hinge"],
                        help="Loss: bce (labels 0/1) or hinge (labels -1/1). Default from config.")
    parser.add_argument("--tune", action="store_true", default=None,
                        help="Enable hyperparameter tuning (grid search over val accuracy).")
    parser.add_argument("--no-tune", action="store_true", default=False,
                        help="Disable tuning even if config has it enabled.")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()

    features_to_probe = args.features if args.features else getattr(nlp_config_mod, "NONLINEAR_PROBE_FEATURES", None) or PROBE_FEATURES
    loss_type = args.loss if args.loss is not None else NONLINEAR_PROBE_LOSS
    if args.no_tune:
        do_tune = False
    elif args.tune is not None:
        do_tune = args.tune
    else:
        do_tune = NONLINEAR_PROBE_TUNE
    output_dir = Path(args.output_dir) if args.output_dir else PROBE_RESULTS_DIR / "nonlinear_torch"

    activation_path = ACTIVATIONS_DIR / f"{args.model}_{args.condition}_activations.npz"
    if not activation_path.exists():
        print(f"Activations not found: {activation_path}")
        return 1

    data = load_activations(activation_path)
    activations = data["activations"]
    feature_labels = data["feature_labels"]
    ground_truth_labels = data["labels"]
    feature_raw_values = data.get("feature_raw_values")

    experiment, control_experiment, best_params_list = run_nonlinear_probing_experiment_torch(
        activations=activations,
        feature_labels=feature_labels,
        ground_truth_labels=ground_truth_labels,
        model_name=args.model,
        condition=args.condition,
        device=device,
        features_to_probe=features_to_probe,
        output_dir=output_dir,
        feature_raw_values=feature_raw_values,
        loss_type=loss_type,
        tune=do_tune,
    )

    export_nonlinear_results(
        experiment, output_dir,
        control_experiment=control_experiment,
        best_params=best_params_list,
    )
    print(f"Results in {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
