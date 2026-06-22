# docs/ax.md — Agentic AI Development Methodology

> **[Important]** This document explains how open-weight models and agentic development tools were used to implement this solution, as required by the Samsung EnnovateX AX Hackathon submission guidelines.

---

## Overview

This project was built using **Google Antigravity** (powered by Gemini), an agentic AI coding assistant, as the primary development partner. The assistant was used not just for code generation, but as a full agentic collaborator — planning, researching, debugging, and iterating across the entire three-phase pipeline.

---

## Agentic AI Setup

**Tool:** Google Antigravity (Gemini-based agentic assistant)  
**Mode:** Pair programming — the agent was given high-level goals and autonomously broke them down into research tasks, implementation plans, and code

**Open-weight model used in the project:**
- `Qwen/Qwen2.5-1.5B-Instruct` (Apache 2.0) — the model being fine-tuned

**Agentic tools used during development:**
- `read_url_content` — fetching Unsloth docs, TRL GRPOConfig API, HuggingFace dataset cards
- `grep_search` / `view_file` — navigating Kaggle notebook cells to extract and understand existing code
- `run_command` — executing Python scripts to dump notebook contents, verify JSON encoding
- `write_to_file` / `replace_file_content` / `multi_replace_file_content` — creating and iteratively refining all project scripts
- `invoke_subagent` — spawning parallel sub-agents to research notebook code while the main agent worked on implementation
- `search_web` — looking up specific error messages (e.g., `PicklingError`, `clipped_ratio=93.75%`, PEFT stacking `TypeError`)

---

## Agentic Workflows

### 1. Planning Phase
The agent was given two large project write-ups and 5 Kaggle notebooks. It autonomously:
- Read the write-ups to extract the architectural blueprint
- Spawned a **research subagent** to parse all notebook cells in parallel
- Produced a structured `implementation_plan.md` with file architecture, dependencies, and open questions
- Identified critical constraints (T4 GPU, `trl==0.24.0` pin, no BF16) by reading the notebooks

### 2. Debugging via Reasoning Pipelines
Several critical bugs were diagnosed by the agent through multi-step reasoning:

**Bug 1: PicklingError**
- Agent traced the error to `trl >= 0.25` + `transformers >= 5.6` breaking Unsloth's kernel patches
- Reasoning chain: `PicklingError` → process forking → Unsloth patches torch ops in-place → newer TRL tries to pickle patched objects → fix: pin `trl==0.24.0`

**Bug 2: clipped_ratio = 93.75%**  
- The GRPO trainer showed `clipped_ratio` near 1.0, meaning completions were all clipping the reward
- Agent diagnosed: SFT used Alpaca template, RL used a different template → model has no EOS association for the RL context → all completions run to `max_new_tokens` → clipping
- Fix: enforce identical templates across SFT and RL scripts

**Bug 3: TypeError on LoRA stacking**
- Loading an SFT LoRA checkpoint and then calling `FastLanguageModel.get_peft_model()` again raised `TypeError`
- Agent diagnosed: bitsandbytes refuses a second LoRA config on an already-adapted model
- Fix: the two-stage LoRA strategy — load in fp16 (not 4-bit), check `isinstance(model, PeftModel)`, merge via `merge_and_unload()`, then inject fresh RL LoRA

### 3. Tool Use / Tool Chaining
A notable tool chain used to extract notebook code:

```
search_web (PicklingError fix)
  → read_url_content (Unsloth GitHub issues)
  → view_file (read notebook as binary)
  → run_command (python dump script → txt file)
  → view_file (read extracted text in chunks)
  → write_to_file (generate clean Python scripts)
  → replace_file_content (iterative refinement)
```

### 4. Multi-Agent Orchestration
For the notebook extraction phase, the agent used **parallel subagent spawning**:

```
Main Agent (planning + writing)
    ├── Subagent 1: Extract code from rl-using-metamath-dataset.ipynb
    ├── Subagent 2: Extract code from notebook3c55043a44.ipynb
    └── Subagent 3: Extract code from stragety-mmlu-sft notebooks
```

The main agent continued writing implementation scripts while subagents worked in parallel on notebook parsing, reducing total wall-clock time significantly.

---

## Reasoning & Planning Pipelines

The agent operated in **Planning Mode** for all major decisions:

1. **Research** → Read all 5 notebooks + project write-ups
2. **Plan** → Write `implementation_plan.md` with proposed file structure
3. **Review** → Ask user for approval on 3 key decisions (code style, Hub config, benchmark numbers)
4. **Execute** → Write files in dependency order (SFT scripts → RL scripts → merge → eval)
5. **Verify** → Cross-check reward function logic against notebook outputs

Key planning artifacts produced by the agent:
- `implementation_plan.md` — complete technical blueprint
- `task.md` — TODO checklist updated as files were created

---

## What Worked

| Agentic Capability | Usefulness |
|--------------------|-----------|
| Multi-step reasoning for bug diagnosis | ⭐⭐⭐⭐⭐ Saved hours of manual debugging |
| Parallel subagent orchestration | ⭐⭐⭐⭐ Cut notebook extraction time ~50% |
| Automatic tool chaining | ⭐⭐⭐⭐ Binary notebook reading → clean Python |
| Web search for library-specific errors | ⭐⭐⭐⭐ Found Unsloth version fix quickly |
| Planning mode with user review checkpoints | ⭐⭐⭐⭐⭐ Prevented wasted implementation effort |

## What Did NOT Work

| Attempted Agentic Pattern | Problem |
|--------------------------|---------|
| Spawning 5 parallel file-writer subagents | Rate-limited (quota exhausted); sequential was needed |
| Agent auto-running training locally | T4 GPU only on Kaggle; agent cannot access Kaggle session |
| Agent auto-filling benchmark numbers | Numbers only available post-training; human must provide |
| CoT RL on QA via agent reasoning alone | Agent correctly identified the failure mode theoretically but could not verify without live training |

---

## Memory & Context Handling

The conversation accumulated significant context from 5 notebooks + 2 project write-ups. The agent used:
- **Transcript checkpoints** — when context was truncated, the system provided a summary and the agent continued from the checkpoint
- **Artifact files** — `implementation_plan.md` and `task.md` served as persistent memory across context windows
- **Scratch files** — `grpo_full.txt`, `rest_notebooks.txt` stored extracted notebook content that was too large for direct context

---

## Coding Assistants & Skills Used

The agent operated with access to the **Antigravity skill system** including science plugins, but the primary skills activated for this project were:
- Built-in `write_file` / `read_file` tools
- `run_command` for local script execution (Windows PowerShell)
- `search_web` for library documentation and error resolution
- `invoke_subagent` for parallel research delegation

---

## Summary

The agentic workflow transformed what would have been a fragmented collection of Kaggle notebooks into a structured, documented, reproducible codebase in a single session. The most powerful agentic contribution was **multi-step causal reasoning for bug diagnosis** — identifying root causes of `clipped_ratio`, `PicklingError`, and PEFT stacking errors that would have taken hours of manual trial-and-error to isolate.
