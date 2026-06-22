# Installation Guide

## Platform: Kaggle (Primary — Recommended)

All training was developed and validated on **Kaggle with NVIDIA Tesla T4 (15.6 GB VRAM)**.

### Local GPU (Advanced)

Requires a GPU with **≥14 GB VRAM** and **CUDA CC ≥ 7.5** (RTX 2080 Ti or better).

### Prerequisites
- Python 3.10+
- CUDA 11.8 or 12.x
- Git

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
# 
!pip install "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate bitsandbytes datasets
!pip install huggingface_hub transformers==5.5.0
!pip install "lm_eval[hf]==0.4.4"
!pip install -q mergekit huggingface_hub
```

> ⚠️ **Critical:** `trl >= 0.25` + `transformers >= 5.6` causes a `PicklingError` with Unsloth.  
> The `--no-deps` flag on the second line prevents pip from pulling newer transformers.


### Step 4 — Notebooks to be uploaded

upload these notebooks in the kaggle 

1. `sft_math_metamathqa.ipynb` → trains SFT Math
2. `sft_flan_cot.ipynb` → trains SFT QA
3. `merge_models.ipynb` → merges both SFT models via Dare Ties method 
4. `rl_grpo_math.ipynb` → loads SFT Math checkpoint, runs GRPO, pushes RL adapter
6. `evaluate.ipynb` → evaluates merged model on all three benchmarks

---

### Run Training in the following order

```bash
# Phase 1: SFT
sft_math_metamathqa.ipynb 
sft_flan_cot.ipynb

# Phase 2: Merge
merge_models.ipynb

# Phase 3: RL
rl_grpo_math.ipynb

# Evaluate
evaluate.ipynb
```

---

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
