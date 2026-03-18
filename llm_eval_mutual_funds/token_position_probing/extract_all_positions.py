# -*- coding: utf-8 -*-
"""
Extract activations at a grid of token positions (from end) across all layers.

Runs the model forward pass once per batch with output_hidden_states=True,
then slices out activations at each grid position and writes them
incrementally to an HDF5 file.  Only the HuggingFace path is supported
(TransformerLens is not needed here since Qwen uses HF anyway).

Usage (from llm_eval_mutual_funds/):
    python token_position_probing/extract_all_positions.py \
        --model qwen3-4b --condition 2_fewshot_cot_temp0
"""
import sys
import argparse
import gc
from pathlib import Path
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Resolve imports via importlib so the script works from any cwd.
# ---------------------------------------------------------------------------
import importlib.util

_THIS_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _THIS_DIR.parent.resolve()
_LP_DIR = _PROJECT_ROOT / "linear_probing"

for _p in (_PROJECT_ROOT, _PROJECT_ROOT.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tp_config = _load_mod("_eap_tp_config", _THIS_DIR / "tp_config.py")
lp_config = _load_mod("_eap_lp_config", _LP_DIR / "lp_config.py")
lp_utils = _load_mod("_eap_lp_utils", _LP_DIR / "lp_utils.py")
extract_mod = _load_mod("_eap_extract", _LP_DIR / "extract_activations.py")

from transformers import AutoModelForCausalLM, AutoTokenizer
from prompts import build_prompt_baseline, build_prompt_zero_shot_cot, build_prompt_few_shot_cot
import config as parent_config

SYSTEM_MSG = parent_config.SYSTEM_MSG
MODELS = tp_config.MODELS
DATA_PATH = tp_config.DATA_PATH
create_feature_labels = tp_config.create_feature_labels
print_banner = tp_config.print_banner

# Reuse helpers from existing extraction module
load_model = extract_mod.load_model
ensure_models_downloaded = extract_mod.ensure_models_downloaded
get_prompt_builder = extract_mod.get_prompt_builder
format_prompt_for_extraction = extract_mod.format_prompt_for_extraction

# ============================================================================
# CORE EXTRACTION
# ============================================================================


def _build_position_grid(min_seq_len: int, step: int):
    """Return sorted list of negative position indices (from end).

    E.g. step=5 → [-1, -6, -11, …] until we exceed -min_seq_len.
    """
    positions = []
    p = -1
    while abs(p) <= min_seq_len:
        positions.append(p)
        p -= step
    return sorted(positions)  # most negative first (earliest in prompt)


def _extract_batch_all_positions(
    model,
    tokenizer,
    prompts: list[str],
    position_grid: list[int],
    n_layers: int,
    d_model: int,
) -> tuple[dict[int, np.ndarray], list[int]]:
    """Run a single forward pass and extract activations at every grid position.

    Uses vectorized gather on GPU and a single GPU->CPU transfer per batch for speed.
    Returns
    -------
    pos_activations : dict[int, ndarray]
        {position: array of shape (batch, n_layers, d_model) float16}
    seq_lengths : list[int]
        Actual (non-padding) sequence length per sample in the batch.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    )
    device = next(model.parameters()).device
    if str(device).startswith("mps"):
        device = "mps"
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    hidden_states = outputs.hidden_states  # tuple of (batch, seq, d_model)
    attention_mask = inputs["attention_mask"]
    seq_lengths = attention_mask.sum(dim=1).tolist()
    batch_size = len(prompts)
    n_positions = len(position_grid)

    # Build indices (batch, n_positions): for each sample and position, token index from end
    # pos_grid values are negative (e.g. -1, -6, …); seq_len + pos = absolute index
    seq_t = torch.tensor(seq_lengths, device=device, dtype=torch.long)
    pos_t = torch.tensor(position_grid, device=device, dtype=torch.long)
    raw_indices = seq_t.unsqueeze(1) + pos_t.unsqueeze(0)       # (batch, n_positions)
    max_indices = (seq_t - 1).unsqueeze(1).expand_as(raw_indices)
    indices = raw_indices.clamp(min=0)
    indices = torch.min(indices, max_indices)

    # Stack all layers on GPU: (batch, n_layers, n_positions, d_model), one transfer to CPU
    stacked = torch.zeros(
        batch_size, n_layers, n_positions, d_model,
        device=device, dtype=torch.float16,
    )
    for layer_idx in range(n_layers):
        layer_hidden = hidden_states[layer_idx + 1]
        index_exp = indices.unsqueeze(-1).expand(-1, -1, d_model)
        gathered = torch.gather(layer_hidden, 1, index_exp)
        stacked[:, layer_idx, :, :] = gathered

    del outputs, hidden_states, inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Single GPU -> CPU transfer, then split by position
    stacked_np = stacked.cpu().numpy()
    del stacked
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pos_activations = {
        position_grid[p]: stacked_np[:, :, p, :].copy()
        for p in range(n_positions)
    }
    return pos_activations, seq_lengths


# ============================================================================
# HDF5 I/O
# ============================================================================


def _init_hdf5(
    path: Path,
    position_grid: list[int],
    n_samples: int,
    n_layers: int,
    d_model: int,
    metadata: dict,
):
    """Create the HDF5 file with pre-allocated datasets."""
    with h5py.File(path, "w") as f:
        f.create_dataset("positions", data=np.array(position_grid, dtype=np.int32))
        f.create_dataset("seq_lengths", shape=(n_samples,), dtype=np.int32)
        f.create_dataset("sample_indices", shape=(n_samples,), dtype=np.int64)

        grp = f.create_group("activations")
        for pos in position_grid:
            ds_name = f"pos_{pos}"
            grp.create_dataset(
                ds_name,
                shape=(n_samples, n_layers, d_model),
                dtype=np.float16,
                chunks=(min(50, n_samples), n_layers, d_model),
                compression="gzip",
                compression_opts=1,
            )
        for k, v in metadata.items():
            f.attrs[k] = v
        f.attrs["processed_count"] = 0


def _write_batch_to_hdf5(
    path: Path,
    batch_start: int,
    batch_end: int,
    pos_activations: dict[int, np.ndarray],
    seq_lengths: list[int],
    sample_indices: np.ndarray,
):
    """Append one batch's data into the HDF5 datasets."""
    with h5py.File(path, "a") as f:
        f["seq_lengths"][batch_start:batch_end] = np.array(seq_lengths, dtype=np.int32)
        f["sample_indices"][batch_start:batch_end] = sample_indices
        for pos, arr in pos_activations.items():
            f[f"activations/pos_{pos}"][batch_start:batch_end] = arr
        f.attrs["processed_count"] = batch_end


def _write_labels_to_hdf5(
    path: Path,
    labels: np.ndarray,
    feature_labels: pd.DataFrame,
):
    """Write ground-truth and per-feature binary labels into the HDF5 file."""
    with h5py.File(path, "a") as f:
        if "labels" not in f:
            f.create_dataset("labels", data=labels.astype(np.int8))
        cols = list(feature_labels.columns)
        if "feature_columns" not in f:
            f.create_dataset("feature_columns", data=np.array(cols, dtype="S64"))
        if "feature_values" not in f:
            f.create_dataset(
                "feature_values",
                data=feature_labels.values.astype(np.float32),
            )


# ============================================================================
# MAIN EXTRACTION LOOP
# ============================================================================


def extract_all_positions(
    model_key: str,
    condition_name: str,
    sample_size: int,
    position_step: int,
    batch_size: int,
    device: str = "cuda",
    position_min: int | None = None,
    position_max: int | None = None,
):
    model_config = MODELS[model_key]
    n_layers = model_config["n_layers"]
    d_model = model_config["d_model"]

    print_banner(f"Token-Position Extraction: {model_key} / {condition_name}")

    # ---- Load data --------------------------------------------------------
    data = pd.read_csv(DATA_PATH)
    if sample_size and sample_size < len(data):
        data = data.sample(n=sample_size, random_state=tp_config.EXTRACTION_RANDOM_STATE)
        data = data.reset_index(drop=True)
    n_samples = len(data)
    print(f"Samples: {n_samples}")
    print(f"Layers: {n_layers}, d_model: {d_model}")
    print(f"Position step: {position_step}")
    print(f"Batch size: {batch_size}")
    if position_min is not None or position_max is not None:
        print(f"Position filter (from end): min={position_min}, max={position_max}")

    # ---- Labels -----------------------------------------------------------
    feature_labels = create_feature_labels(data)
    labels = feature_labels["medalist_f1_higher"].values.copy()

    # ---- Load model -------------------------------------------------------
    actual_device = device
    if device == "mps" or (
        device == "cuda"
        and not torch.cuda.is_available()
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        actual_device = "mps"
    if actual_device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    ensure_models_downloaded(model_key)
    loaded = load_model(model_key, device=actual_device)
    if isinstance(loaded, tuple):
        model, tokenizer = loaded
    else:
        model = loaded
        tokenizer = model.tokenizer

    prompt_builder = get_prompt_builder(condition_name)
    print(f"Prompt builder: {prompt_builder.__name__}")

    # ---- First pass: determine min seq length for position grid -----------
    print("Tokenizing first batch to estimate sequence lengths …")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    first_prompts = []
    for _, row in data.head(min(batch_size, n_samples)).iterrows():
        first_prompts.append(format_prompt_for_extraction(row, prompt_builder, tokenizer))
    first_enc = tokenizer(first_prompts, padding=True, truncation=True, max_length=4096, return_tensors="pt")
    first_lens = first_enc["attention_mask"].sum(dim=1).tolist()
    est_min_len = int(min(first_lens))
    del first_enc, first_prompts

    # Full grid based on estimated minimum sequence length
    full_grid = _build_position_grid(est_min_len, position_step)

    # Optionally restrict to a sub-range, e.g. [-200, -1] for the last 200 tokens.
    if position_min is not None or position_max is not None:
        lo = position_min if position_min is not None else min(full_grid)
        hi = position_max if position_max is not None else max(full_grid)
        if lo > hi:
            raise ValueError(f"Invalid position range: min={lo} > max={hi}")
        position_grid = [p for p in full_grid if lo <= p <= hi]
        if not position_grid:
            raise ValueError(
                f"No positions remain after filtering to [{lo}, {hi}]. "
                f"Available range from full grid: [{min(full_grid)}, {max(full_grid)}]"
            )
        print(f"Position grid filter: kept {len(position_grid)}/{len(full_grid)} positions in [{lo}, {hi}]")
    else:
        position_grid = full_grid

    n_positions = len(position_grid)
    per_pos_mb = n_samples * n_layers * d_model * 2 / (1024 ** 2)
    total_gb = per_pos_mb * n_positions / 1024
    print(f"Estimated min seq len: {est_min_len}")
    print(f"Position grid: {n_positions} positions (step={position_step})")
    print(f"Estimated HDF5 size: {total_gb:.1f} GB")

    # ---- HDF5 setup -------------------------------------------------------
    hdf5_path = tp_config.get_tp_hdf5_path(model_key, condition_name)
    start_idx = 0

    if hdf5_path.exists():
        with h5py.File(hdf5_path, "r") as f:
            existing_count = int(f.attrs.get("processed_count", 0))
            existing_positions = list(f["positions"][:])
        if existing_positions == position_grid and existing_count < n_samples:
            start_idx = existing_count
            print(f"Resuming from sample {start_idx}")
        elif existing_count >= n_samples and existing_positions == position_grid:
            print(f"HDF5 already complete ({existing_count} samples). Skipping extraction.")
            _write_labels_to_hdf5(hdf5_path, labels, feature_labels)
            _cleanup(model, tokenizer)
            return hdf5_path
        else:
            print("Position grid changed or file inconsistent — recreating HDF5.")
            hdf5_path.unlink()

    if start_idx == 0:
        metadata = {
            "model": model_key,
            "condition": condition_name,
            "n_samples": n_samples,
            "n_layers": n_layers,
            "d_model": d_model,
            "position_step": position_step,
            "n_positions": n_positions,
            "position_min": position_min if position_min is not None else min(position_grid),
            "position_max": position_max if position_max is not None else max(position_grid),
            "extraction_date": datetime.now().isoformat(),
        }
        _init_hdf5(hdf5_path, position_grid, n_samples, n_layers, d_model, metadata)

    # ---- Batch extraction -------------------------------------------------
    sample_indices = data.index.values.copy()
    pbar = tqdm(range(start_idx, n_samples, batch_size), desc="Extracting positions")

    for batch_start in pbar:
        batch_end = min(batch_start + batch_size, n_samples)
        batch_data = data.iloc[batch_start:batch_end]

        prompts = [
            format_prompt_for_extraction(row, prompt_builder, tokenizer)
            for _, row in batch_data.iterrows()
        ]

        pos_acts, seq_lens = _extract_batch_all_positions(
            model, tokenizer, prompts, position_grid, n_layers, d_model,
        )

        _write_batch_to_hdf5(
            hdf5_path,
            batch_start,
            batch_end,
            pos_acts,
            seq_lens,
            sample_indices[batch_start:batch_end],
        )

        del prompts, pos_acts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pbar.set_postfix({
            "processed": batch_end,
            "gpu_gb": f"{torch.cuda.max_memory_allocated() / 1e9:.1f}" if torch.cuda.is_available() else "n/a",
        })

    # ---- Write labels & clean up ------------------------------------------
    _write_labels_to_hdf5(hdf5_path, labels, feature_labels)
    _cleanup(model, tokenizer)
    print(f"\nSaved HDF5: {hdf5_path}  ({hdf5_path.stat().st_size / 1e9:.2f} GB)")
    return hdf5_path


def _cleanup(model, tokenizer):
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Extract activations at a grid of token positions for token-position probing."
    )
    parser.add_argument("--model", "-m", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--condition", "-c", default="2_fewshot_cot_temp0")
    parser.add_argument(
        "--sample-size", type=int, default=tp_config.TP_SAMPLE_SIZE,
        help=f"Number of samples (default {tp_config.TP_SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--position-step", type=int, default=tp_config.POSITION_STEP,
        help=f"Grid step (default {tp_config.POSITION_STEP})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=tp_config.TP_EXTRACTION_BATCH_SIZE,
        help=f"Batch size (default {tp_config.TP_EXTRACTION_BATCH_SIZE})",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    parser.add_argument(
        "--position-min",
        type=int,
        default=None,
        help="Minimum (most negative) token position to extract (from end). Example: -200.",
    )
    parser.add_argument(
        "--position-max",
        type=int,
        default=None,
        help="Maximum (closest to end) token position to extract (from end). Example: -1.",
    )
    args = parser.parse_args()

    extract_all_positions(
        model_key=args.model,
        condition_name=args.condition,
        sample_size=args.sample_size,
        position_step=args.position_step,
        batch_size=args.batch_size,
        device=args.device,
        position_min=args.position_min,
        position_max=args.position_max,
    )


if __name__ == "__main__":
    main()
