# ============================================================================
# HUGGING FACE AUTHENTICATION
# ============================================================================
from huggingface_hub import login, snapshot_download
import os
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Login with your token (set HF_TOKEN environment variable)
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("Warning: HF_TOKEN not set. You may need to login manually with `huggingface-cli login`")

print("Logged in to Hugging Face!")
print("\n" + "="*80)
print("DOWNLOADING MODELS FOR MUTUAL FUND PROJECT")
print("="*80)

# ============================================================================
# DOWNLOAD LLAMA MODEL
# ============================================================================
print("\n[1/2] Downloading Llama-3.2-3B-Instruct model...")
print("Size: ~13 GB | This may take a few minutes depending on your connection.")
print("-" * 80)

llama_dir = MODELS_DIR / "Llama-3.2-3B-Instruct"
snapshot_download(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    local_dir=str(llama_dir),
    local_dir_use_symlinks=False
)

print("✓ Llama-3.2-3B-Instruct downloaded successfully!")
print(f"  Location: {llama_dir}")

# ============================================================================
# DOWNLOAD QWEN MODEL
# ============================================================================
print("\n[2/2] Downloading Qwen3-4B-Instruct-2507 model...")
print("Size: ~8 GB | This may take a few minutes depending on your connection.")
print("-" * 80)

qwen_dir = MODELS_DIR / "Qwen3-4B-Instruct-2507"
snapshot_download(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",
    local_dir=str(qwen_dir),
    local_dir_use_symlinks=False
)

print("✓ Qwen3-4B-Instruct-2507 downloaded successfully!")
print(f"  Location: {qwen_dir}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ALL MODELS DOWNLOADED SUCCESSFULLY!")
print("="*80)
print("\nDownloaded Models:")
print(f"  1. Llama-3.2-3B-Instruct      → {llama_dir}")
print(f"  2. Qwen3-4B-Instruct-2507     → {qwen_dir}")
print("\nTotal size: ~21 GB")
print("\nYou can now use these models in your Mutual Fund project!")
print("="*80)
