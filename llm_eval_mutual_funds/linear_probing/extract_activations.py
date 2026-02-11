# -*- coding: utf-8 -*-
"""
Activation Extraction using TransformerLens

This module extracts residual stream activations from transformer models
for use in linear probing experiments.
"""
import sys
import subprocess
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Callable, Tuple
from datetime import datetime
import argparse
import gc
import importlib.util

# Get paths
_THIS_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _THIS_DIR.parent.resolve()


def _load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


lp_config = _load_module_from_path("lp_config", _THIS_DIR / "lp_config.py")
lp_utils = _load_module_from_path("lp_utils", _THIS_DIR / "lp_utils.py")

# Also add project root for prompts
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# TransformerLens import (optional - only for supported models)
try:
    from transformer_lens import HookedTransformer

    TRANSFORMERLENS_AVAILABLE = True
except ImportError:
    TRANSFORMERLENS_AVAILABLE = False
    print("TransformerLens not installed - will use HuggingFace for all models")

# HuggingFace imports (for manual extraction)
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import from explicitly loaded modules
MODELS = lp_config.MODELS
DATA_PATH = lp_config.DATA_PATH
ACTIVATIONS_DIR = lp_config.ACTIVATIONS_DIR
CHECKPOINT_DIR = lp_config.CHECKPOINT_DIR
EXTRACTION_BATCH_SIZE = lp_config.EXTRACTION_BATCH_SIZE
EXTRACTION_CHECKPOINT_INTERVAL = lp_config.EXTRACTION_CHECKPOINT_INTERVAL
EXTRACTION_SAMPLE_SIZE = lp_config.EXTRACTION_SAMPLE_SIZE
EXTRACTION_RANDOM_STATE = lp_config.EXTRACTION_RANDOM_STATE
HOOK_PATTERN = lp_config.HOOK_PATTERN
TOKEN_POSITION = lp_config.TOKEN_POSITION
PROJECT_ROOT = lp_config.PROJECT_ROOT

create_feature_labels = lp_utils.create_feature_labels
save_activations = lp_utils.save_activations
save_checkpoint = lp_utils.save_checkpoint
load_latest_checkpoint = lp_utils.load_latest_checkpoint
get_activation_path = lp_utils.get_activation_path
print_banner = lp_utils.print_banner

# Import prompt builders from parent project
from prompts import build_prompt_baseline, build_prompt_zero_shot_cot, build_prompt_few_shot_cot
import config as parent_config

SYSTEM_MSG = parent_config.SYSTEM_MSG

# ============================================================================
# PROMPT BUILDERS REGISTRY
# ============================================================================

PROMPT_BUILDERS = {
    "build_prompt_baseline": build_prompt_baseline,
    "build_prompt_zero_shot_cot": build_prompt_zero_shot_cot,
    "build_prompt_few_shot_cot": build_prompt_few_shot_cot,
}


def get_prompt_builder(condition_name: str) -> Callable:
    """
    Get the prompt builder function for a given condition.
    """
    if "baseline" in condition_name:
        return build_prompt_baseline
    if "fewshot" in condition_name:
        return build_prompt_few_shot_cot
    if "zeroshot" in condition_name:
        return build_prompt_zero_shot_cot
    raise ValueError(f"Unknown condition: {condition_name}")


# ============================================================================
# MODEL DOWNLOAD CHECK
# ============================================================================


def is_model_downloaded(model_key: str) -> bool:
    """Return True if the model's local_path exists and contains config.json."""
    if model_key not in MODELS:
        return False
    local_path = MODELS[model_key]["local_path"]
    resolved = Path(local_path).resolve()
    return resolved.exists() and (resolved / "config.json").exists()


def ensure_models_downloaded(model_key: str) -> None:
    """
    If the requested model is not present locally, run download_model.py.
    The download script fetches all configured models; existing ones are skipped by the Hub.
    """
    if is_model_downloaded(model_key):
        return
    print(f"Model '{model_key}' not found at {MODELS[model_key]['local_path']}. Running download script...")
    script = PROJECT_ROOT / "llm_eval_mutual_funds/download_model.py"
    if not script.exists():
        raise FileNotFoundError(
            f"download_model.py not found at {script}. "
            "Please run it manually from the project root to download models."
        )
    subprocess.run(
        [sys.executable, str(script)],
        cwd=str(PROJECT_ROOT),
        check=True,
    )
    if not is_model_downloaded(model_key):
        raise RuntimeError(f"Download script completed but model '{model_key}' still not found at {MODELS[model_key]['local_path']}")


# ============================================================================
# MODEL LOADING
# ============================================================================


def load_model_transformerlens(model_key: str, device: str = "cuda") -> HookedTransformer:
    """
    Load a model with TransformerLens.
    """
    model_config = MODELS[model_key]
    hf_name = model_config["hf_name"]

    print("Loading via TransformerLens...")

    model = HookedTransformer.from_pretrained(
        hf_name,
        device=device,
        dtype=torch.float16,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )

    print(f"Model loaded: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")
    return model


def load_model_huggingface(model_key: str, device: str = "cuda"):
    """
    Load a model with HuggingFace transformers (for manual hook extraction).

    Returns:
        Tuple of (model, tokenizer)
    """
    model_config = MODELS[model_key]
    hf_name = model_config["hf_name"]
    local_path = model_config["local_path"]

    print("Loading via HuggingFace (manual extraction)...")

    # Check if local path exists and has model files
    local_path_resolved = local_path.resolve()
    config_file = local_path_resolved / "config.json"
    if local_path_resolved.exists() and config_file.exists():
        load_path = str(local_path_resolved)
        print(f"  Using local model: {load_path}")
    else:
        print(f"  WARNING: Local model not found at: {local_path_resolved}")
        if not local_path_resolved.exists():
            print(f"    Directory does not exist")
        elif not config_file.exists():
            print(f"    config.json not found (model may be incomplete)")
        print(f"  Attempting to download from HuggingFace Hub: {hf_name}")
        print(f"  Note: This may take a while and may timeout. To download models locally, run:")
        print(f"    cd {PROJECT_ROOT}")
        print(f"    python download_model.py")
        load_path = hf_name

    tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
    
    # Handle MPS device - use device_map="cpu" first, then move to mps
    # MPS doesn't support device_map="mps" directly
    try:
        if device == "mps":
            print("  Loading model to CPU first (will move to MPS after)...")
            model = AutoModelForCausalLM.from_pretrained(
                load_path,
                torch_dtype=torch.float16,
                device_map="cpu",  # Load to CPU first
                trust_remote_code=True,
                low_cpu_mem_usage=True,  # More memory efficient
            )
            print("  Moving model to MPS...")
            model = model.to("mps")  # Move to MPS after loading
        else:
            model = AutoModelForCausalLM.from_pretrained(
                load_path,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True,
            )
        model.eval()
    except Exception as e:
        print(f"  Error loading model: {e}")
        print(f"  If this is a timeout, the model may need to be downloaded locally.")
        print(f"  Run: python download_model.py")
        raise

    print(f"Model loaded: {model_config['n_layers']} layers, d_model={model_config['d_model']}")
    return model, tokenizer


def load_model(model_key: str, device: str = "cuda"):
    """
    Load a model using appropriate method (TransformerLens or HuggingFace).
    Automatically falls back to HuggingFace if MPS is detected (TransformerLens doesn't support MPS well).
    """
    model_config = MODELS[model_key]
    hf_name = model_config["hf_name"]
    use_tl = model_config.get("use_transformerlens", False)

    # Check if device is MPS - TransformerLens doesn't support MPS well on macOS
    if device == "mps" or (device == "cuda" and not torch.cuda.is_available() and torch.backends.mps.is_available()):
        print_banner(f"Loading {model_key}")
        print(f"Model name: {hf_name}")
        print(f"Device: MPS detected")
        print(f"Method: HuggingFace (manual hooks) - TransformerLens not compatible with MPS")
        return load_model_huggingface(model_key, device="mps")

    print_banner(f"Loading {model_key}")
    print(f"Model name: {hf_name}")
    print(f"Device: {device}")
    print(f"Method: {'TransformerLens' if use_tl else 'HuggingFace (manual hooks)'}")

    if use_tl:
        if not TRANSFORMERLENS_AVAILABLE:
            raise ImportError("TransformerLens required but not installed")
        return load_model_transformerlens(model_key, device)
    return load_model_huggingface(model_key, device)


# ============================================================================
# ACTIVATION EXTRACTION
# ============================================================================


class HuggingFaceActivationExtractor:
    """
    Activation extraction using HuggingFace's built-in output_hidden_states.
    Used for models not supported by TransformerLens.
    """

    def __init__(self, model, tokenizer, n_layers: int, d_model: int):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = n_layers
        self.d_model = d_model

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def extract(self, prompts: List[str], token_position: int = -1) -> np.ndarray:
        """
        Extract activations for a batch of prompts using output_hidden_states.
        """
        batch_size = len(prompts)

        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096
        )

        # Get device from model
        device = next(self.model.parameters()).device
        # Handle MPS device string conversion
        if str(device).startswith("mps"):
            device = "mps"
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)

        hidden_states = outputs.hidden_states

        attention_mask = inputs["attention_mask"]
        seq_lengths = attention_mask.sum(dim=1).tolist()

        activations = np.zeros((batch_size, self.n_layers, self.d_model), dtype=np.float32)

        for layer_idx in range(self.n_layers):
            layer_hidden = hidden_states[layer_idx + 1].cpu().numpy()

            for batch_idx in range(batch_size):
                seq_len = seq_lengths[batch_idx]
                pos = seq_len - 1 if token_position == -1 else min(token_position, seq_len - 1)
                activations[batch_idx, layer_idx, :] = layer_hidden[batch_idx, pos, :]

        del outputs
        del hidden_states
        del inputs

        return activations


def format_prompt_for_extraction(
    row: pd.Series, prompt_builder: Callable, tokenizer, system_msg: str = SYSTEM_MSG
) -> str:
    """
    Format a prompt for extraction, applying chat template.
    """
    user_content = prompt_builder(row)

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"<|system|>\n{system_msg}\n<|user|>\n{user_content}\n<|assistant|>\n"


def extract_batch_activations_transformerlens(
    model: HookedTransformer, prompts: List[str], token_position: int = -1
) -> np.ndarray:
    """
    Extract activations using TransformerLens run_with_cache.
    Note: This doesn't work well with MPS on macOS - use HuggingFace method instead.
    """
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    batch_size = len(prompts)

    # Only clear CUDA cache if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    names_filter = lambda name: name.endswith(HOOK_PATTERN)

    with torch.no_grad():
        _, cache = model.run_with_cache(prompts, names_filter=names_filter, stop_at_layer=None)

    activations = np.zeros((batch_size, n_layers, d_model), dtype=np.float32)

    for layer_idx in range(n_layers):
        hook_name = f"blocks.{layer_idx}.{HOOK_PATTERN}"
        if hook_name in cache:
            layer_acts = cache[hook_name].cpu().numpy()
            activations[:, layer_idx, :] = layer_acts[:, token_position, :]

    del cache
    del layer_acts
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return activations


def extract_batch_activations_huggingface(
    extractor: HuggingFaceActivationExtractor, prompts: List[str], token_position: int = -1
) -> np.ndarray:
    """
    Extract activations using HuggingFace forward hooks.
    Works with CUDA, MPS, and CPU.
    """
    # Only clear CUDA cache if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    activations = extractor.extract(prompts, token_position)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return activations


def extract_activations(
    model_key: str,
    condition_name: str,
    data: pd.DataFrame,
    batch_size: int = EXTRACTION_BATCH_SIZE,
    checkpoint_interval: int = EXTRACTION_CHECKPOINT_INTERVAL,
    resume: bool = True,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Extract activations for all samples in the dataset.

    Returns:
        Tuple of (activations, sample_indices, labels, feature_labels)
    """
    model_config = MODELS[model_key]
    n_layers = model_config["n_layers"]
    d_model = model_config["d_model"]
    use_transformerlens = model_config.get("use_transformerlens", False)
    n_samples = len(data)

    print_banner(f"Extracting Activations: {model_key} / {condition_name}")
    print(f"Samples: {n_samples}")
    print(f"Layers: {n_layers}, d_model: {d_model}")
    print(f"Batch size: {batch_size}")
    print(f"Method: {'TransformerLens' if use_transformerlens else 'HuggingFace (manual)'}")

    prompt_builder = get_prompt_builder(condition_name)
    print(f"Prompt builder: {prompt_builder.__name__}")

    # Check if MPS is being used - force HuggingFace method
    actual_device = device
    if device == "mps" or (device == "cuda" and not torch.cuda.is_available() and torch.backends.mps.is_available()):
        actual_device = "mps"
        use_transformerlens = False  # Force HuggingFace for MPS
        print("Note: MPS detected - using HuggingFace extraction (TransformerLens not compatible with MPS)")

    if actual_device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    ensure_models_downloaded(model_key)
    loaded = load_model(model_key, device=actual_device)

    if use_transformerlens:
        model = loaded
        tokenizer = model.tokenizer
        extractor = None
    else:
        model, tokenizer = loaded
        extractor = HuggingFaceActivationExtractor(model, tokenizer, n_layers, d_model)

    all_activations = np.zeros((n_samples, n_layers, d_model), dtype=np.float32)
    sample_indices = data.index.values.copy()

    # Create feature labels and target labels (Medalist)
    feature_labels = create_feature_labels(data)
    labels = feature_labels["medalist_f1_higher"].values.copy()

    # Check for existing checkpoint
    start_idx = 0
    checkpoint_prefix = f"extraction_{model_key}_{condition_name}"

    if resume:
        checkpoint = load_latest_checkpoint(CHECKPOINT_DIR, prefix=checkpoint_prefix)
        if checkpoint is not None:
            start_idx = checkpoint["processed_count"]
            all_activations[:start_idx] = checkpoint["activations"][:start_idx]
            print(f"Resuming from sample {start_idx}")

    # Process in batches
    pbar = tqdm(range(start_idx, n_samples, batch_size), desc="Extracting")

    for batch_idx, batch_start in enumerate(pbar):
        batch_end = min(batch_start + batch_size, n_samples)
        batch_data = data.iloc[batch_start:batch_end]

        prompts = []
        for _, row in batch_data.iterrows():
            prompt = format_prompt_for_extraction(row, prompt_builder, tokenizer)
            prompts.append(prompt)

        if use_transformerlens:
            batch_activations = extract_batch_activations_transformerlens(
                model, prompts, token_position=TOKEN_POSITION
            )
        else:
            batch_activations = extract_batch_activations_huggingface(
                extractor, prompts, token_position=TOKEN_POSITION
            )

        all_activations[batch_start:batch_end] = batch_activations

        del prompts
        del batch_activations

        if batch_idx % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        pbar.set_postfix(
            {
                "processed": batch_end,
                "memory_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            }
        )

        if (batch_end % checkpoint_interval == 0) or (batch_end == n_samples):
            save_checkpoint(
                {
                    "activations": all_activations,
                    "processed_count": batch_end,
                    "model_key": model_key,
                    "condition_name": condition_name,
                },
                CHECKPOINT_DIR,
                prefix=checkpoint_prefix,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\nCleaning up model and GPU memory...")
    del model
    del tokenizer
    if extractor is not None:
        del extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    return all_activations, sample_indices, labels, feature_labels


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("Extracting activations...")
    parser = argparse.ArgumentParser(description="Extract activations for linear probing")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        choices=list(MODELS.keys()),
        help="Model to extract from",
    )
    parser.add_argument(
        "--condition",
        "-c",
        type=str,
        required=True,
        help="Experimental condition (e.g., '0_baseline_temp0')",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=EXTRACTION_BATCH_SIZE,
        help="Batch size for extraction",
    )
    parser.add_argument(
        "--sample-size",
        "-n",
        type=int,
        default=EXTRACTION_SAMPLE_SIZE,
        help=f"Number of samples to extract (default: {EXTRACTION_SAMPLE_SIZE}, None for all)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--no-resume", action="store_true", help="Don't resume from checkpoint"
    )

    args = parser.parse_args()

    print(f"Loading data from {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(data)} samples from dataset")
    
    # Sample data if sample_size is specified
    sample_size = args.sample_size if args.sample_size > 0 else None
    if sample_size is not None and sample_size < len(data):
        print(f"Sampling {sample_size} samples (random_state={EXTRACTION_RANDOM_STATE})...")
        data = data.sample(n=sample_size, random_state=EXTRACTION_RANDOM_STATE).reset_index(drop=True)
        print(f"Using {len(data)} samples for extraction")
    elif sample_size is not None and sample_size >= len(data):
        print(f"Sample size ({sample_size}) >= dataset size ({len(data)}), using all samples")
    else:
        print(f"Using all {len(data)} samples")

    activations, sample_indices, labels, feature_labels = extract_activations(
        model_key=args.model,
        condition_name=args.condition,
        data=data,
        batch_size=args.batch_size,
        resume=not args.no_resume,
        device=args.device,
    )

    output_path = get_activation_path(ACTIVATIONS_DIR, args.model, args.condition)

    metadata = {
        "model": args.model,
        "condition": args.condition,
        "n_samples": len(data),
        "sample_size": args.sample_size if args.sample_size > 0 else None,
        "random_state": EXTRACTION_RANDOM_STATE,
        "extraction_date": datetime.now().isoformat(),
        "n_layers": MODELS[args.model]["n_layers"],
        "d_model": MODELS[args.model]["d_model"],
        "hook_pattern": HOOK_PATTERN,
        "token_position": TOKEN_POSITION,
        "batch_size": args.batch_size,
    }

    save_activations(
        activations=activations,
        sample_indices=sample_indices,
        labels=labels,
        feature_labels=feature_labels,
        path=output_path,
        metadata=metadata,
    )

    print_banner("Extraction Complete!")
    print(f"Activations saved to: {output_path}")
    print(f"Shape: {activations.shape}")


if __name__ == "__main__":
    main()
