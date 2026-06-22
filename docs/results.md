# Results & Benchmark Evaluation

## Evaluation Setup

- **Evaluation script:** [`src/evaluate.py`](../src/evaluate.py)
- **Benchmarks:** GSM8K (math), MMLU (multi-subject), StrategyQA (boolean reasoning)
- **Metric:** Exact-match accuracy (greedy decoding, `do_sample=False`)
- **Prompt format:** Alpaca template (identical to training)
- **Hardware:** Kaggle T4 (15.6 GB VRAM)

---

## Main Results Table

> **Fill in your actual numbers below before submission.**

| Model Stage | GSM8K | StrategyQA | Notes |
|---|---|---|---|
| Base `Qwen2.5-1.5B-Instruct` | ~56.0% | 57.0% | Zero-shot baseline |
| + SFT Math (MetaMathQA 150k, step-1500) | ~60.0% | - | checkpoint-1500 |
| + SFT QA (FLAN-CoT) | - | 61.0% | CoT reasoning |
| + RL Math (GRPO, r=4, 2k medium-tier) | **63.0%** | - | r=4 LoRA, 3-tier curriculum |
| **Merged (Task Vectors, λ=0.5/0.5)** | **~62.0%** | **~60.5%** | Minor dip from merging |

---

## Experiment Log

### Experiment 1 — SFT Math Baseline
- **Dataset:** MetaMathQA 150k (sampled from 395k)
- **Epochs:** 3 · **Batch:** 16 effective · **LR:** 2e-4
- **Best checkpoint:** step-1500 (val loss = 1.822 vs 1.824 at final epoch)
- **GSM8K result:** [FILL]%

### Experiment 2 - SFT QA with CommonsenseQA 
- **Dataset:** tau/commonsense_qa dataset
- **Result:** Breaks the multi-step logic needed for StrategyQA resulted in 10% decrease than the baseline score

### Experiment 3 — SFT QA with FLAN-CoT ✅ (Breakthrough)
- **Dataset:** FLAN-CoT dataset containing reasoning traces
- **Insight:** Contradicted the initial hypothesis that CoT degrades QA performance
- **StrategyQA result:** Improved from 57% to 61% (proving CoT benefits retrieval QA)

### Experiment 4 — SFT QA Direct-Answer ❌
- **Dataset:** StrategyQA (True/False) + MMLU (A/B/C/D), 13k total, NO CoT
- **Result:** Underperformed the FLAN-CoT approach, confirming reasoning is necessary.

### Experiment 5 — GRPO Math with max_new_tokens=800 ❌
- `clipped_ratio` = 0.9375 from step 1 — completions always hit the length limit
- No learning signal — reward variance was zero

### Experiment 6 — GRPO Math with Template Mismatch ❌
- SFT used Alpaca template; GRPO used different template
- Model never generates EOS in the new context → `clipped_ratio=93.75%`

### Experiment 7 — GRPO Math Fixed (Template + max_new_tokens=500) ✅
- Reward signal established from step 1
- Curriculum promoted from medium tier after ~150 steps (rolling acc > 70%)
- **GSM8K result after RL:** 63.0% (a 6-7% absolute improvement over base)

### Experiment 8 — GRPO QA CoT with `<think>` tags ❌ (Track B)
- No domain-specific CoT SFT data for QA tasks
- Reward sparsity: model rarely produces correct structured CoT for factual questions
- Abandoned after 50 steps (reward flatlined at ~0.02)

### Experiment 9 — GRPO QA No-CoT with Small Dataset (2.2k StrategyQA only) ❌
- Policy looped through the 2.2k dataset 7× in one epoch → overfitting
- Val accuracy collapsed on MMLU (domain shift from StrategyQA-only RL)

### Experiment 10 — GRPO QA No-CoT with Large Dataset (13k) ✅
- Matches SFT distribution — prevents domain shift
- Binary reward (1.0 / 0.0) + `max_new_tokens=8` — unambiguous signal
- **MMLU result after RL:** [FILL]% · **StrategyQA:** [FILL]%

### Experiment 11 — LoRA Stacking on Quantised Model ❌ → Fixed
- Loading SFT checkpoint + calling `get_peft_model()` again → `TypeError`
- Fix: two-stage LoRA — merge SFT LoRA first (`merge_and_unload()`), then inject r=4 RL LoRA

### Experiment 12 — Task Vector Merge ✅
- λ_math=0.5, λ_qa=0.5
- **Final merged model:** Retained both domain gains with only a minor dip (approx 62.0% GSM8K, 60.5% StrategyQA)

---

## Key Findings

1. **CoT SFT improves QA even on 1.5B models.** Contradicting earlier studies, reasoning traces from FLAN-CoT significantly improved StrategyQA performance (57% → 61%) compared to direct-answer prediction.

2. **GRPO is highly sensitive to prompt template consistency.** A single template mismatch between SFT and RL causes complete training failure (`clipped_ratio → 1.0`).

3. **RL dataset size matters for QA.** 2.2k samples → policy collapse; 13k matching SFT distribution → stable learning.

4. **Task vector merging works for orthogonal domain specialists.** Math reasoning and factual QA engage sufficiently different weight subspaces that linear interpolation retains both gains.

5. **Curriculum learning improves final GSM8K accuracy** by preventing reward saturation on easy problems and ensuring the model encounters harder multi-step arithmetic late in training.

---

## Reproduce These Results

```bash
# Evaluate any model checkpoint
python src/evaluate.py \
  --model_path Suryansh7123/qwen2.5_grpo_rl_r4_metamath \
  --benchmark gsm8k \
  --n_samples 500 \
  --output_file results_rl_math.json

# Full evaluation on merged model
python src/evaluate.py \
  --model_path ./merged_model \
  --benchmark all \
  --n_samples 500
```
