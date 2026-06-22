# Final Presentation Content Outline

*Use this content to build your final PDF presentation.*

## Slide 1: Title Slide
- **Project Title:** RL Fine-Tuning of Qwen2.5-1.5B for Multi-Benchmark QA
- **Problem Statement:** 7 - Fine-Tuning and Reinforcement Learning for Multi-Benchmark Language Model Optimization
- **Team:** [Your Team Name]
- **Member:** Suryansh Singh
- **Institute:** [Your Institute]

## Slide 2: Project Overview & Objectives
- **Goal:** Fine-tune a 1.5B parameter open-weight model (`Qwen2.5-1.5B-Instruct`) to excel at both mathematical reasoning (GSM8K) and factual QA (MMLU/StrategyQA).
- **Constraints Overcome:** Trained entirely on free Kaggle NVIDIA Tesla T4 hardware (15.6 GB VRAM).
- **Three-Phase Pipeline:**
  1. Supervised Fine-Tuning (SFT)
  2. Group Relative Policy Optimization (GRPO) Reinforcement Learning
  3. Task Vector Model Merging

## Slide 3: The Core Innovation
- **Initial Hypothesis vs. Reality:** We initially assumed Chain-of-Thought (CoT) would degrade factual QA by causing hallucinations. However, our experiments proved the opposite.
- **Breakthrough:** Fine-tuning on the FLAN-CoT dataset improved StrategyQA accuracy from 57% to 61%.
- **The Challenge:** Sequential training on math and QA caused catastrophic forgetting.
- **The Solution:** We trained domain specialists in parallel and merged them using Task Vector Arithmetic to retain both capabilities.

## Slide 4: Agentic AI Methodology (ax.md requirement)
- **Collaborator:** Built using Google Antigravity (Gemini-based agentic AI).
- **Agentic Workflows Used:**
  - Multi-step reasoning to diagnose deep technical bugs (e.g., `PicklingError` with Unsloth/TRL).
  - Parallel sub-agents orchestrated to extract and refactor Kaggle notebook code simultaneously.
  - Planning mode with explicit user-review checkpoints to prevent wasted implementation effort.

## Slide 5: Key Technical Fixes & Engineering
- **Fixing GRPO Clipping:** Prompt template mismatch between SFT and RL caused `clipped_ratio=93.75%`. Fixed by strictly enforcing identical Alpaca templates.
- **Solving PEFT Stacking Errors:** Stacking LoRA on an already quantised model raised `TypeError`. Solved by merging SFT LoRA into base weights in fp16 before injecting the new RL LoRA.
- **Curriculum Learning for RL:** Implemented a 3-tier curriculum (easy → medium → hard) based on reasoning step count to stabilize GRPO math training.

## Slide 6: Results & Impact
- **Mathematical Reasoning (GSM8K):** Improved from ~56% (base) to **63%** using our 3-tier curriculum GRPO pipeline (a massive 6-7% absolute improvement).
- **Boolean QA (StrategyQA):** Improved from 57% (base) to **61%** via CoT reasoning.
- **Merged Model Performance:** Task Vector Merge successfully combined both specialists with only a minor performance dip (~62% GSM8K, ~60% StrategyQA), achieving true multi-benchmark optimization on a 1.5B model.

## Slide 7: Conclusion & Reproducibility
- **10+ Verified Experiments:** Full iteration history documented in `docs/results.md`.
- **Proof of Work:** All original Kaggle notebooks containing execution logs and reward curves are published in the repository.
- **Fully Reproducible Codebase:** Scripts modularized in `src/` for SFT, RL, merging, and evaluation.
- **Thank You / Q&A**
