# Installation Guide

## Platform: Kaggle (Primary — Recommended)

All training was developed and validated on **Kaggle with NVIDIA Tesla T4 (15.6 GB VRAM)**.

### Step 1 — Create a Kaggle Notebook
1. Go to [kaggle.com](https://www.kaggle.com) → **Create** → **New Notebook**
2. Under **Settings** → **Accelerator** → Select **GPU T4 x2** (or GPU P100)
3. Enable **Internet** access

### Step 2 — Add HuggingFace Token
1. Go to **Add-ons** → **Secrets**
2. Add secret: **Label** = `HF_TOKEN`, **Value** = your HuggingFace token
3. Enable the secret in the notebook

### Step 3 — Install Dependencies (Cell 1 of any notebook)
```bash
# Pin exact versions — DO NOT upgrade individually
pip install "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes datasets
```

> ⚠️ **Critical:** `trl >= 0.25` + `transformers >= 5.6` causes a `PicklingError` with Unsloth.  
> The `--no-deps` flag on the second line prevents pip from pulling newer transformers.

### Step 4 — Clear Stale Cache + Disable Dynamo (Cell 2)
```python
import shutil, os
cache_path = "/kaggle/working/unsloth_compiled_cache"
if os.path.exists(cache_path):
    shutil.rmtree(cache_path)
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"]     = "1"
```

### Step 5 — Run the Scripts

upload the notebooks in the kaggle and execute in the following order

**Recommended order:**
1. `sft_math_metamathqa.ipynb` → trains SFT Math, pushes to `Suryansh7123/qwen2.5_lora_r16_finetune`
2. `sft_flan_cot.ipynb` → trains SFT QA, pushes to `kanishkav/qwen2.5-1.5b-SFT-FLANCOT`
3. `merge_models.ipynb` → merges both RL models via task vectors
4. `rl_grpo_math.ipynb` → loads SFT Math checkpoint, runs GRPO, pushes RL adapter
6. `evaluate.ipynb` → evaluates merged model on all three benchmarks

---

## Platform: Local GPU (Advanced)

Requires a GPU with **≥14 GB VRAM** and **CUDA CC ≥ 7.5** (RTX 2080 Ti or better).

### Prerequisites
- Python 3.10+
- CUDA 11.8 or 12.x
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/[YOUR_GITHUB_USERNAME]/[YOUR_REPO_NAME].git
cd [YOUR_REPO_NAME]

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies (same pins as Kaggle)
pip install "unsloth[cu118-ampere-torch230] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl==0.24.0 peft accelerate bitsandbytes datasets
pip install huggingface_hub transformers==5.5.0
```

### Set HuggingFace Token

```bash
export HF_TOKEN="hf_your_token_here"   # Linux/Mac
$env:HF_TOKEN = "hf_your_token_here"   # Windows PowerShell
```

### Run Training

```bash
# Phase 1: SFT
python src/sft_math.py
python src/sft_qa.py

# Phase 2: RL
python src/rl_math.py
python src/rl_qa_no_cot.py

# Phase 3: Merge
python src/merge_models.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --math_model Suryansh7123/qwen2.5_grpo_rl_r4_metamath \
  --qa_model   Suryansh7123/qwen2.5_grpo_rl_qa_no_cot \
  --output_dir ./merged_model

# Evaluate
python src/evaluate.py \
  --model_path ./merged_model \
  --benchmark all \
  --n_samples 500
```

---

## Reproducing Results from Notebooks

The easiest way to reproduce results is to run the Kaggle notebooks directly:

1. Go to the `notebooks/` folder in this repo
2. Download the notebook you want to reproduce
3. On Kaggle: **Create** → **New Notebook** → **File** → **Import Notebook**
4. Upload the downloaded `.ipynb` file
5. Set GPU accelerator and add `HF_TOKEN` secret
6. **Run All** — the notebook already contains all code cells in correct order

> All notebooks include **cell outputs with training logs** showing they ran successfully on T4 GPUs.

---

## Requirements

```
# Auto-generated — install via the pinned pip commands above, NOT this file directly
unsloth[kaggle-new]  @ git+https://github.com/unslothai/unsloth.git
trl==0.24.0          # pinned — newer versions cause PicklingError with Unsloth
transformers==5.5.0  # pinned — must match trl pin
peft                 # latest compatible with above
accelerate           # latest compatible
bitsandbytes         # for 8-bit AdamW optimizer
datasets             # HuggingFace datasets
huggingface_hub      # Hub push/pull
torch>=2.0.0         # with CUDA support
numpy
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `PicklingError` in SFTTrainer | `trl>=0.25` + `transformers>=5.6` | Pin versions as above |
| `clipped_ratio=0.9375` in GRPO | SFT/RL template mismatch | Ensure identical Alpaca template |
| `TypeError` on get_peft_model | Stacking LoRA on already-adapted model | Use two-stage LoRA (merge then re-inject) |
| CUDA OOM | VRAM insufficient | Reduce `per_device_batch`, enable 4-bit |
| `KeyError: 'prompt'` in GRPOTrainer | Missing `prompt` column | Add `prompt` column to dataset before training |
| Kaggle session timeout | Long training run | Keep-alive thread included in all scripts |
