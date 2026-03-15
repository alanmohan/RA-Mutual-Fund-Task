# Token-Position Probing Experiment

## Motivation

Standard linear probing extracts activations at the **last token** only.
If a feature's probe accuracy is low there, two hypotheses compete:

1. **The model never encodes the feature well** — it does not form a useful
   linear representation at any point.
2. **The model encodes the feature near its value tokens but "forgets" it by
   the end** — the representation decays or gets overwritten by later context.

This experiment distinguishes the two by probing across a **grid of token
positions** (from the end of the prompt) and plotting accuracy as a function
of position and layer.

- If accuracy peaks near the feature-value tokens and drops by the last token,
  hypothesis 2 is supported.
- If accuracy is roughly flat (and low) everywhere, hypothesis 1 is supported.

---

## Workflow

```
Phase 1 — Extraction
  extract_all_positions.py  →  HDF5 (per-position activations, float16)

Phase 2 — Probing
  probe_positions.py        →  CSV  (accuracy per (position, layer))

Phase 3 — Visualization
  plot_position_results.py  →  PNG  (heatmap, line plots)
```

### Step 1: Extract activations at all grid positions

```bash
cd llm_eval_mutual_funds

python token_position_probing/extract_all_positions.py \
    --model qwen3-4b \
    --condition 2_fewshot_cot_temp0 \
    --sample-size 1000 \
    --position-step 5 \
    --batch-size 5 \
    --device cuda
```

This runs the model forward pass once per batch and extracts activations at
every 5th token from the end (positions -1, -6, -11, …) across all layers.
Output is written incrementally to an HDF5 file under
`data/activations/token_position/`.

**Resume:** If the script is interrupted, re-running with the same arguments
resumes from the last completed batch (tracked in HDF5 attrs).

### Step 2: Probe a feature across positions

```bash
python token_position_probing/probe_positions.py \
    --model qwen3-4b \
    --condition 2_fewshot_cot_temp0 \
    --feature beta_f1_lower \
    --n-workers 4
```

For each position in the grid, this trains logistic-regression probes across
all layers (parallelized via `joblib`) with full C-selection on a held-out
validation set.  Results are saved to
`data/probe_results/token_position/tp_probe_{feature}_{model}_{condition}.csv`.

### Step 3: Visualize

```bash
python token_position_probing/plot_position_results.py \
    --results-csv data/probe_results/token_position/tp_probe_beta_f1_lower_qwen3-4b_2_fewshot_cot_temp0.csv \
    --feature beta_f1_lower \
    --feature-position -89 \
    --last-token-accuracy 0.746
```

**Outputs** (in `data/probe_results/token_position/`):

| File | Description |
|------|-------------|
| `tp_heatmap_*.png` | Heatmap — layers (y) × positions (x), colour = test accuracy |
| `tp_best_layer_*.png` | Best-layer accuracy vs position, plus which layer is best |
| `tp_layer_lines_*.png` | Accuracy vs position for selected layers |

Optional flags:
- `--feature-position` draws a vertical dashed line where the feature value
  appears in the prompt (from `find_token_position.py`).
- `--last-token-accuracy` draws a horizontal reference line.
- `--layers 10 19 23` selects specific layers for the line plot.

---

## Running the full pipeline on Colab (bash script)

After cloning the repo into your Colab instance, you can run the entire
pipeline (download models → extract → probe → plot) with one script:

```bash
cd llm_eval_mutual_funds   # or cd to the repo root and use the path below

export HF_TOKEN="your_huggingface_token"
bash token_position_probing/run_token_position_experiment.sh
```

The script uses default values from this README (model `qwen3-4b`, condition
`2_fewshot_cot_temp0`, feature `beta_f1_lower`, etc.). To override them, set
environment variables **before** running the script:

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | (none) | Hugging Face token for gated model download |
| `MODEL` | `qwen3-4b` | Model key |
| `CONDITION` | `2_fewshot_cot_temp0` | Condition name |
| `SAMPLE_SIZE` | `1000` | Number of samples for extraction |
| `POSITION_STEP` | `5` | Grid step (every Nth token from end) |
| `BATCH_SIZE` | `24` | Extraction batch size (use 5 for T4 16 GB; 24–32 for A100 40 GB) |
| `DEVICE` | `cuda` | Device for extraction |
| `FEATURE` | `beta_f1_lower` | Feature to probe |
| `N_WORKERS` | `4` | Parallel workers for layer probing |
| `FEATURE_POSITION` | (none) | Optional; e.g. `-89` for plot vertical line |
| `LAST_TOKEN_ACCURACY` | (none) | Optional; e.g. `0.746` for plot horizontal line |
| `LAYERS` | (none) | Optional; e.g. `10 19 23` for selected-layers plot |
| `RUN_DOWNLOAD` | `yes` | Set to `no` to skip model download |
| `RUN_EXTRACTION` | `yes` | Set to `no` to skip extraction |
| `RUN_PROBING` | `yes` | Set to `no` to skip probing |
| `RUN_PLOTTING` | `yes` | Set to `no` to skip plotting |
| `VENV_DIR` | `llm_eval_mutual_funds/.venv_extract` | Path to venv used for download + extraction |

**Environment:** The script creates a virtualenv (or uses `VENV_DIR`), installs dependencies from `requirements.txt` plus `h5py`, and runs **model download** and **activation extraction** inside that venv. It then **deactivates** the venv and runs **probing** and **plotting** in the **global** (system) Python environment.

Example: run for a different feature and add plot annotations:

```bash
export HF_TOKEN="hf_..."
export FEATURE="sharpe_f1_higher"
export FEATURE_POSITION="-95"
export LAST_TOKEN_ACCURACY="0.913"
bash token_position_probing/run_token_position_experiment.sh
```

---

## Configuration Reference (`tp_config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `POSITION_STEP` | 5 | Extract every Nth token from end |
| `TP_SAMPLE_SIZE` | 1000 | Samples to extract (fewer = faster, less storage) |
| `TP_EXTRACTION_BATCH_SIZE` | 24 | Batch size (use 5 for T4 16 GB; 24–32 for A100 40 GB) |
| `N_PROBE_WORKERS` | 4 | Parallel layer probes per position |

All values are overridable via CLI arguments.

---

## Storage Estimates

Per-position dataset size: `n_samples × n_layers × d_model × 2 bytes` (float16).

| Samples | Step | Positions | Disk (Qwen3-4B) |
|---------|------|-----------|-----------------|
| 1000 | 5 | ~140 | ~24 GB |
| 1000 | 10 | ~70 | ~12 GB |
| 1000 | 20 | ~35 | ~6 GB |
| 500 | 5 | ~140 | ~12 GB |

---

## Interpreting the Plots

### Heatmap

- **Bright band at a specific position range** (especially near the feature
  value) that fades toward the last token → the model encodes the feature early
  but loses it.
- **Uniformly dim** across all positions → the model never forms a strong
  linear representation of this feature.
- **Bright throughout** → the feature is robustly encoded and retained.

### Best-layer line plot

- **Peak near feature-value position, drop toward -1** → encoding exists but
  decays (hypothesis 2).
- **Flat line near chance** → no useful encoding at any position.
- The secondary axis shows which layer is "best" at each position; if it
  shifts substantially, the feature may be processed differently at different
  stages.

### Selected-layers line plot

- Compare early, middle, and late layers to see where in the network the
  feature is most accessible and how it evolves along the prompt.

---

## Directory Layout

```
token_position_probing/
├── tp_config.py                 # Configuration
├── extract_all_positions.py     # Phase 1: extraction → HDF5
├── probe_positions.py           # Phase 2: parallel probing → CSV
├── plot_position_results.py     # Phase 3: visualization → PNG
├── run_token_position_experiment.sh   # Full pipeline script (Colab-friendly)
└── README.md                    # This file

data/
├── activations/token_position/
│   └── {model}_{condition}_all_positions.h5
└── probe_results/token_position/
    ├── tp_probe_{feature}_{model}_{condition}.csv
    ├── tp_heatmap_{feature}_{model}_{condition}.png
    ├── tp_best_layer_{feature}_{model}_{condition}.png
    └── tp_layer_lines_{feature}_{model}_{condition}.png
```
