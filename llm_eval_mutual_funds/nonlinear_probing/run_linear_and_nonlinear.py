#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline: run both linear and nonlinear probes on the same activations,
then save results and generate comparison plots.
Uses linear_probing for linear probes and config; lives in nonlinear_probing/.
"""
import argparse
import pickle
import importlib.util
from pathlib import Path

_THIS_DIR = Path(__file__).parent.resolve()
_LINEAR_PROBING_DIR = _THIS_DIR.parent / "linear_probing"

# Config and paths from linear_probing
lp_config = importlib.util.spec_from_file_location(
    "lp_config", str(_LINEAR_PROBING_DIR / "lp_config.py")
)
lp_config_mod = importlib.util.module_from_spec(lp_config)
lp_config.loader.exec_module(lp_config_mod)

PROBE_RESULTS_DIR = lp_config_mod.PROBE_RESULTS_DIR
ACTIVATIONS_DIR = lp_config_mod.ACTIVATIONS_DIR
MODELS = lp_config_mod.MODELS

# Linear probe module
probe_spec = importlib.util.spec_from_file_location(
    "probe", str(_LINEAR_PROBING_DIR / "probe.py")
)
probe_mod = importlib.util.module_from_spec(probe_spec)
probe_spec.loader.exec_module(probe_mod)

# lp_utils for load_activations
lp_utils_spec = importlib.util.spec_from_file_location(
    "lp_utils", str(_LINEAR_PROBING_DIR / "lp_utils.py")
)
lp_utils_mod = importlib.util.module_from_spec(lp_utils_spec)
lp_utils_spec.loader.exec_module(lp_utils_mod)
load_activations = lp_utils_mod.load_activations

# Nonlinear probe module (this folder)
nonlinear_spec = importlib.util.spec_from_file_location(
    "nonlinear_probe", str(_THIS_DIR / "nonlinear_probe.py")
)
nonlinear_mod = importlib.util.module_from_spec(nonlinear_spec)
nonlinear_spec.loader.exec_module(nonlinear_mod)


def main():
    parser = argparse.ArgumentParser(
        description="Run linear and nonlinear probes on same data, then plot comparison"
    )
    parser.add_argument("--model", "-m", type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument("--condition", "-c", type=str, required=True)
    parser.add_argument("--output-dir", "-o", type=str, default=None)
    parser.add_argument("--features", type=str, nargs="+", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else PROBE_RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    nonlinear_dir = output_dir / "nonlinear"
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger = probe_mod.setup_logging(output_dir, args.model, args.condition)
    activation_path = ACTIVATIONS_DIR / f"{args.model}_{args.condition}_activations.npz"

    if not activation_path.exists():
        logger.error(f"Activations not found at {activation_path}. Run extract_activations.py first.")
        return 1

    logger.info(f"Loading activations from {activation_path}")
    data = load_activations(activation_path)
    activations = data["activations"]
    feature_labels = data["feature_labels"]
    labels = data["labels"]

    # ---------- Linear probes: always ALL layers (unaffected by nonlinear layer config) ----------
    probe_mod.print_banner("Linear probes (all layers)")
    linear_experiment = probe_mod.run_probing_experiment(
        activations=activations,
        feature_labels=feature_labels,
        ground_truth_labels=labels,
        model_name=args.model,
        condition=args.condition,
        features_to_probe=args.features,
        output_dir=output_dir,
        logger=logger,
    )
    linear_pickle = output_dir / f"probe_{args.model}_{args.condition}.pkl"
    linear_experiment.save(linear_pickle)
    probe_mod.export_results_to_csv(linear_experiment, output_dir)
    probe_mod.export_config(linear_experiment, output_dir)

    # ---------- Nonlinear probes + control (shuffled labels): layers from nlp_config ----------
    nonlinear_mod.print_banner("Nonlinear probes (layers from nlp_config) + control tasks")
    nonlinear_experiment, control_experiment = nonlinear_mod.run_nonlinear_probing_experiment(
        activations=activations,
        feature_labels=feature_labels,
        ground_truth_labels=labels,
        model_name=args.model,
        condition=args.condition,
        features_to_probe=args.features,
        output_dir=nonlinear_dir,
        logger=logger,
    )
    nonlinear_pickle = nonlinear_dir / f"probe_nonlinear_{args.model}_{args.condition}.pkl"
    with open(nonlinear_pickle, "wb") as f:
        pickle.dump(nonlinear_experiment, f)
    control_pickle = nonlinear_dir / f"probe_nonlinear_control_{args.model}_{args.condition}.pkl"
    with open(control_pickle, "wb") as f:
        pickle.dump(control_experiment, f)
    nonlinear_mod.export_nonlinear_results(
        nonlinear_experiment, nonlinear_dir, control_experiment=control_experiment
    )

    # ---------- Comparison plots (linear, nonlinear, control) ----------
    try:
        plot_spec = importlib.util.spec_from_file_location(
            "plot_linear_vs_nonlinear",
            str(_THIS_DIR / "plot_linear_vs_nonlinear.py"),
        )
        plot_mod = importlib.util.module_from_spec(plot_spec)
        plot_spec.loader.exec_module(plot_mod)
        plot_mod.run_comparison_plots(
            linear_experiment=linear_experiment,
            nonlinear_experiment=nonlinear_experiment,
            output_dir=plots_dir,
            model_name=args.model,
            condition=args.condition,
            control_experiment=control_experiment,
        )
    except Exception as e:
        logger.warning(f"Comparison plotting failed: {e}")

    probe_mod.print_banner("Pipeline complete")
    logger.info(f"Linear results: {output_dir}")
    logger.info(f"Nonlinear results: {nonlinear_dir}")
    logger.info(f"Plots: {plots_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
