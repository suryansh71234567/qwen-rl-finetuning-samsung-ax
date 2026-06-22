# Technical Architecture

## System Overview

This project implements a three-phase LLM fine-tuning pipeline for `Qwen/Qwen2.5-1.5B-Instruct` targeting three benchmarks simultaneously: **GSM8K** (math reasoning), **MMLU** (multi-subject knowledge), and **StrategyQA** (multi-hop boolean reasoning).

```
┌─────────────────────────────────────────────────────────┐
│              Qwen/Qwen2.5-1.5B-Instruct                 │
│                    (Base Model)                         │
└───────────────────┬─────────────────────────────────────┘
                    │
          ┌─────────┴──────────┐
          │                    │
    ┌─────▼──────┐      ┌──────▼──────┐
    │  SFT Math  │      │   SFT QA    │
    │ MetaMathQA │      │ StrategyQA  │
    │  150k, r16 │      │  13k, r16   │
    └─────┬──────┘      └──────┬──────┘
          │                    │
          └─────────┬──────────┘
                    │
          ┌─────────▼──────────┐
          │  Dare Ties Merge    │
          └─────────────────────┘
                  │
            ┌─────▼──────┐      
            │  RL Math   │     
            │GRPO+Curric.│     
            │  2k, r4    │      
            └─────┬──────┘                   
                  │
            ┌─────▼──────┐      
            │  final     │     
            │  model     │           
            └─────┬──────┘        

```

---

## Phase 1: Supervised Fine-Tuning (SFT)

### SFT Math Track
- **Dataset:** `meta-math/MetaMathQA` — 395k augmented GSM8K + MATH problems. Subsampled to 150k.
- **Key design:** Full CoT responses (MetaMathQA includes step-by-step traces). This gives the model a cold-start for the GRPO phase.
- **LoRA config:** r=16, α=8, rsLoRA, 7 projection layers, fp16
- **Batch:** 2 per device × 8 accumulation = effective batch 16
- **Optimal checkpoint:** step-1500 (val loss minimum, confirmed empirically)

### SFT QA Track
- **Dataset:** `metaeval/strategy-qa` (2.29k) + `cais/mmlu` auxiliary_train (99.8k). Downsampled to 15k total (13k train / 2k val).
- **Key design:** Direct-answer format (NO CoT). For retrieval-heavy QA on a 1.5B model, CoT causes hallucination and lowers exact-match. Single-token answers (A/B/C/D, True/False) maximise P(correct token) under greedy decoding.
- **Same LoRA config** as Math SFT

---

## Phase 2: Reinforcement Learning (GRPO)

### Algorithm: Group Relative Policy Optimization (GRPO)
For each prompt, the policy generates G=6 completions. Relative rewards within the group are used to compute advantages — no separate value network needed.

```
For prompt x:
  Generate completions {y₁, y₂, ..., yG}   (G=6)
  Compute rewards {r₁, r₂, ..., rG}
  Advantage: Aᵢ = (rᵢ - mean(r)) / std(r)
  Loss: LGRPO = -E[Aᵢ · log π_θ(yᵢ|x)] + β·KL(π_θ || π_ref)
```

### RL Math Track
**Reward function (multi-component):**

| Component | Weight | Signal |
|-----------|--------|--------|
| Correctness | 0.80 | Exact numeric match after extraction |
| Format | 0.10 | "The answer is:" present |
| Length | 0.10 | Linear penalty above 300 words |
| Step bonus | +0.05 | ≥3 reasoning lines before answer |

**3-Tier Curriculum:**
```
Easy (≤5 GSM steps) → Medium (9-15 GSM steps) → Hard (MATH-derived, >15 steps)
                   ↑ promote when rolling pass@1 > 70% over 200-step window ↑
```

**Two-stage LoRA strategy (critical fix):**
```python
# 1. Load SFT checkpoint in fp16 (not 4-bit)
model, tokenizer = FastLanguageModel.from_pretrained(..., load_in_4bit=False)

# 2. If SFT LoRA still attached, merge into base
if isinstance(model, PeftModel):
    model = model.merge_and_unload()

# 3. Inject fresh RL LoRA (r=4)
model = FastLanguageModel.get_peft_model(model, r=4, ...)
```

### RL QA Track (No-CoT)
- **Reward:** Binary exact-match (1.0 / 0.0)
- `max_new_tokens=8` (QA outputs are 1-2 tokens; 800 caused constant truncation)
- `temperature=0.7`, `kl_coeff=0.05` (higher KL anchors near SFT to prevent collapse)
- Dataset: 13k (same distribution as SFT — smaller dataset caused 7× looping)

---

## Phase 3: Task Vector Model Merging

### Why Task Vectors?
Training Math-RL and QA-RL **sequentially** on the same model causes catastrophic forgetting. Training in parallel and merging avoids this.

### Merging Formula
```
τ_math = θ_math_RL - θ_base
τ_qa   = θ_qa_RL   - θ_base
θ_merged = θ_base + λ_math · τ_math + λ_qa · τ_qa
```

### Coefficient Selection
Grid search over λ ∈ {0.3, 0.4, 0.5, 0.6, 0.7} × λ ∈ {0.3, 0.4, 0.5, 0.6, 0.7}.  
Score: **Harmonic mean** of (GSM8K, MMLU, StrategyQA) — penalises extreme degradation on any single benchmark.

---

## Technology Stack

| Component | Library | Version |
|-----------|---------|---------|
| Base model | Qwen2.5-1.5B-Instruct | — |
| Efficient fine-tuning | Unsloth | 2026.6.x |
| SFT trainer | TRL SFTTrainer | 0.24.0 (pinned) |
| RL trainer | TRL GRPOTrainer | 0.24.0 (pinned) |
| Model patching | Transformers | 5.5.0 (pinned) |
| LoRA adapters | PEFT | latest compatible |
| Quantisation | bitsandbytes | latest compatible |
| Data | HuggingFace Datasets | latest |
| Compute | Kaggle T4 GPU | 15.6 GB VRAM |

### Version Pinning (Critical)
`trl >= 0.25` + `transformers >= 5.6` re-introduces a `PicklingError` with Unsloth. The `--no-deps` flag prevents pip from upgrading transformers when installing TRL:

```bash
pip install "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes datasets
```

---

## OSS Libraries Used

| Library | Purpose | Link |
|---------|---------|------|
| Unsloth | Memory-efficient LoRA fine-tuning on T4 | https://github.com/unslothai/unsloth |
| TRL | SFTTrainer + GRPOTrainer | https://github.com/huggingface/trl |
| Transformers | Model loading, tokenisation | https://github.com/huggingface/transformers |
| PEFT | LoRA adapter management | https://github.com/huggingface/peft |
| bitsandbytes | 8-bit AdamW optimizer | https://github.com/bitsandbytes-foundation/bitsandbytes |
| HuggingFace Datasets | Dataset loading + streaming | https://github.com/huggingface/datasets |
| HuggingFace Hub | Model/checkpoint push | https://github.com/huggingface/huggingface_hub |

---

## Salient Features

1. **T4-optimised throughout** — FP16 everywhere (T4 Turing has no BF16 registers), Unsloth gradient checkpointing, 8-bit AdamW, sequence packing
2. **Template consistency enforcement** — identical Alpaca template across SFT and RL scripts prevents `clipped_ratio=93.75%`
3. **Pre-flight smoke test** — 16-example mini-run before full training to catch VRAM, version, and checkpoint errors early
4. **Keep-alive thread** — prevents Kaggle session timeout during multi-hour training runs
5. **Hub push at every checkpoint** — durable checkpointing every N steps so partial training is never lost on Kaggle session expiry
