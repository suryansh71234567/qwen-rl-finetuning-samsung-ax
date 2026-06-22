# RL Fine-Tuning of Qwen2.5-1.5B for Multi-Benchmark QA

<!-- FILL IN BEFORE SUBMISSION -->
- **Problem Statement Number** - 6
- **Problem Statement Title** - Enhancing Reasoning in Small Language Models (SLMs) using Reinforcement Learning
- **Team name** - Reason_reinforced
- **Team members (Names)** - R Kanishka Varshini , Suryansh Singh
- **Institute/College Name** - Indian Institute Of Technology Roorkee , Roorkee ,  Roorkee - Haridwar Highway, Roorkee, Uttarakhand 247667
- **Final Presentation Google Drive Link** - https://drive.google.com/drive/folders/1SyJNupXwflVHkONSi2S88umqF5YNBTcI
- **Full Submission Demo Video Link** - 
- **Setup & Result Reproducibility Video Link** - 

---

## Project Overview

This project implements a **three-phase pipeline** for fine-tuning `Qwen/Qwen2.5-1.5B-Instruct` using Supervised Fine-Tuning (SFT) followed by Group Relative Policy Optimization (GRPO) Reinforcement Learning, evaluated across three benchmarks: **GSM8K**, **MMLU**, and **StrategyQA**.

> **Hardware:** Kaggle NVIDIA Tesla T4 (15.6 GB VRAM) · **Framework:** Unsloth + TRL

---

## Core Innovation: Domain-Specific Reasoning & Model Merging

Initially, it was hypothesized that Chain-of-Thought (CoT) would degrade performance on factual QA tasks. However, our experiments demonstrated that **CoT fine-tuning significantly improves QA performance**. 

When fine-tuned on the FLAN-CoT dataset, **StrategyQA accuracy improved from 57% to 61%**, demonstrating that reasoning traces benefit retrieval-heavy QA tasks. On the math side, our curriculum-based RL pipeline improved **GSM8K from ~56% to 63% (a 6-7% absolute improvement over the base model)**.

To combine these capabilities without sequential catastrophic forgetting, we utilized **Task Vector Model Merging**, combining the strengths of the math-reasoning and QA-reasoning specialists into a single model with minimal performance degradation.

---

## Experiments & Development Process

> **Note on GitHub commit history:** The vast majority of active development happened **directly on Kaggle** (iterative notebook runs, live GPU sessions, reward curve inspection). GitHub was used for final code organisation. The real "commit history" is the Kaggle notebook execution logs inside the [`src/`](./src/) folder — each notebook contains cell outputs, training loss curves, and reward breakdowns that prove successful execution on T4 GPUs.

We ran **10+ distinct experiments** across the three-phase pipeline. Many things worked; many did not — and both are documented:

| # | Experiment | Outcome |
|---|-----------|---------|
| 1 | SFT Math on MetaMathQA (150k, 1 epoch) | ✅ Strong GSM8K improvement |
| 2 | SFT QA with CommonsenseQA dataset | ❌ Breaks the multi-step logic needed for StrategyQA. |
| 3 | SFT QA with FLAN-CoT dataset | ✅ **Breakthrough:** StrategyQA improved from 57% to 61% |
| 4 | SFT QA direct-answer format | ❌ Underperformed CoT on complex reasoning |
| 5 | GRPO Math with `max_new_tokens=800` | ❌ Constant token truncation, clipped_ratio=93% |
| 6 | GRPO Math with mismatched prompt template | ❌ Model never generates EOS — `clipped_ratio=93.75%` |
| 7 | GRPO Math with fixed template + `max_new_tokens=500` | ✅ Reward signal established; GSM8K reached 63% |
| 8 | GRPO QA with CoT + `<think>` tags (Track B) | ❌ Reward sparsity without proper CoT bootstrapping |
| 9 | Task Vector Merge (λ=0.5/0.5) | ✅ Retained both domain gains with only minor dip |
| 10 | SFT stacking LoRA on quantised model | ❌ `TypeError` — fixed by two-stage LoRA strategy |

*[Add your own observations and numbers here — you know the experiments best.]*

---

## Results

### Benchmark Accuracy Across Training Phases

| Model Stage | GSM8K | StrategyQA | Notes |
|---|---|---|---|
| Base `Qwen2.5-1.5B-Instruct` | ~56.0% | 57.0% | Zero-shot baseline |
| After SFT Math (MetaMathQA 150k) | ~60.0% | - | checkpoint-1500 |
| After SFT QA (FLAN-CoT) | - | 61.0% | CoT reasoning |
| After RL Math (GRPO, MetaMathQA 2k) | **63.0%** | - | r=4 LoRA, 3-tier curriculum |
| **Merged Model (Task Vectors)** | **~62.0%** | **~60.5%** | λ_math=0.5, λ_qa=0.5 (Minor dip from merge) |


---

## Project Artefacts

### Technical Documentation
See the [`docs/`](./docs/) folder:
- [`docs/technical_architecture.md`](./docs/technical_architecture.md) — System architecture, training pipeline, design decisions
- [`docs/implementation_details.md`](./docs/implementation_details.md) — Code walkthrough, hyperparameters, debugging log
- [`docs/installation.md`](./docs/installation.md) — Setup instructions for Kaggle and local environments
- [`docs/results.md`](./docs/results.md) — Full benchmark results with training curves
- [`docs/ax.md`](./docs/ax.md) — **Agentic AI development methodology** (how Antigravity/Gemini was used)

### Kaggle Notebooks (With Execution Logs)
The [`src/`](./src/) folder contains the **original Kaggle notebooks with full training logs and outputs**, proving successful execution on Kaggle T4 GPUs:

| Notebook | Description |
|----------|-------------|
| [`src/sft_math_metamathqa.ipynb`](./src/sft_math_metamathqa.ipynb) | SFT Math — full training run with loss curves |
| [`src/sft_qa_strategy_mmlu.ipynb`](./src/sft_qa_strategy_mmlu.ipynb) | SFT QA — full training run |
| [`src/sft_qa_v2_strategy_mmlu.ipynb`](./src/sft_qa_v2_strategy_mmlu.ipynb) | SFT QA v2 (improved dataset loading) |
| [`src/sft_flan_cot.ipynb`](./src/sft_flan_cot.ipynb) | SFT with FLAN-COT (ablation study) |
| [`src/merge-models.ipynb`](./src/merge-models.ipynb) | Task vector merging execution logs |
| [`src/rl_grpo_math.ipynb`](./src/rl_grpo_math.ipynb) | GRPO RL Math — reward curves, curriculum promotions |
| [`src/evaluate.ipynb`](./src/evaluate.ipynb) | Evaluation pipeline execution logs |

### Models Used
- [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) — Base model (Apache 2.0)

### Models Published
- [kanishkav/qwen2.5-1.5b-merged-model](https://huggingface.co/kanishkav/qwen2.5-1.5b-merged-model) - Final Reasoning Enhanced SLM 
- [Suryansh7123/qwen2.5_lora_r16_finetune](https://huggingface.co/Suryansh7123/qwen2.5_lora_r16_finetune) — SFT Math LoRA checkpoint
- [Suryansh7123/qwen2.5_lora_r16_finetune_STRATEGY](https://huggingface.co/Suryansh7123/qwen2.5_lora_r16_finetune_STRATEGY) — SFT QA LoRA checkpoint
- [Suryansh7123/qwen2.5_grpo_rl_r4_metamath](https://huggingface.co/Suryansh7123/qwen2.5_grpo_rl_r4_metamath) — RL Math adapter (GRPO)
- [Suryansh7123/qwen2.5_grpo_rl_qa_no_cot](https://huggingface.co/Suryansh7123/qwen2.5_grpo_rl_qa_no_cot) — RL QA adapter (GRPO, no-CoT)
- [kanishkav/qwen2.5-1.5b-SFT-FLANCOT](https://huggingface.co/kanishkav/qwen2.5-1.5b-SFT-FLANCOT) -SFT on FLANCOT
- [kanishkav/qwen2.5-1.5B-SFT-CSQA](https://huggingface.co/kanishkav/qwen2.5-1.5B-SFT-CSQA) - SFT on CommonsenseQA

### Datasets Used
- [meta-math/MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) — Math SFT + RL (MIT License)
- [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) — MMLU multiple-choice QA (MIT License)
- [metaeval/strategy-qa](https://huggingface.co/datasets/metaeval/strategy-qa) — StrategyQA boolean QA (Apache 2.0)
- [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) — Grade School Math evaluation (MIT License)

### Datasets Published
- [kanishkav/OpenThoughts_processed_qwen_Dataset](https://huggingface.co/datasets/kanishkav/OpenThoughts_processed_qwen_Dataset) - dataset processed for qwen format

## Technical Pipeline

```
Qwen2.5-1.5B-Instruct (Base)
         │
    ┌────┴────┐
    │         │
 SFT Math   SFT QA
(MetaMathQA) (MMLU+StrategyQA)
    │         │
 RL Math   RL QA
 (GRPO+    (GRPO+
 Curriculum) Binary Reward)
    │         │
    └────┬────┘
         │
   Task Vector Merge
   (λ_math=0.5, λ_qa=0.5)
         │
   Merged Model ✅
```

---

## Key Engineering Contributions

1. **Two-stage LoRA Strategy** — Merge SFT LoRA into base before attaching RL LoRA (r=4). Fixes `TypeError` from stacking PEFT adapters; the r=4 RL adapter also acts as a regulariser against catastrophic forgetting.

2. **Template Consistency Enforcement** — Discovered that any mismatch between SFT and RL prompt templates causes `clipped_ratio=93.75%` (model never generates EOS). Fixed by enforcing identical Alpaca templates across all scripts.

3. **3-Tier Curriculum Learning** — GSM8K problems filtered by reasoning step count (easy ≤5, medium 9–15, hard >15 lines). Rolling accuracy tracker auto-promotes model to harder tier when pass@1 > 70%.

4. **Multi-Component Reward (Math RL)** — Correctness (0.80) + Format (0.10) + Length penalty (0.10) + Step structure bonus (0.05). Prevents degenerate policies that output bare numbers.

5. **Task Vector Merging** — Avoids sequential catastrophic forgetting by training domain specialists in parallel and merging via `θ_merged = θ_base + λ_A·τ_A + λ_B·τ_B`. Coefficient selection via harmonic mean grid search.

---

## Attribution

This project builds upon:
- **[Unsloth](https://github.com/unslothai/unsloth)** — Memory-efficient LLM fine-tuning framework (Apache 2.0)
- **[TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl)** — GRPO implementation (Apache 2.0)
- **[Qwen2.5](https://github.com/QwenLM/Qwen2.5)** — Base model (Apache 2.0)

**New contributions** (not present in base projects):
- 3-tier curriculum scheduler for GRPO
- Knowledge vs. reasoning split hypothesis and ablation study
- Two-stage LoRA merge strategy to prevent PEFT stacking errors
- Task vector merging for multi-benchmark LLM consolidation

---

## Setup

See [`docs/installation.md`](./docs/installation.md) for full setup instructions.

**Quick start (Kaggle):**
```bash
# Install pinned dependencies
pip install "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl==0.24.0 peft accelerate bitsandbytes datasets

# Run SFT (Math track)
python src/sft_math.py

# Run GRPO RL (Math track)  
python src/rl_math.py

# Evaluate merged model
python src/evaluate.py --model_path ./merged_model --benchmark all --n_samples 500
```
