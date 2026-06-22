"""
sft_math.py
===========
Phase 1 — Supervised Fine-Tuning for the GSM8K / Math Track.

Dataset   : meta-math/MetaMathQA (~395k samples, sampled to 150k)
Template  : Alpaca (identical to the downstream GRPO notebook)
LoRA      : r=16, alpha=8, rsLoRA, all 7 projection layers
Precision : FP16 (T4 Turing does NOT support BF16 registers)
Hardware  : Kaggle NVIDIA Tesla T4 (15.6 GB VRAM)

Key design decisions:
- Alpaca template MUST match the RL phase exactly (mismatch causes clipped_ratio > 90%)
- Best checkpoint is step-1500 (validation loss minimum); load this for GRPO phase
- MetaMathQA responses include full CoT traces — ideal cold-start for RL

Usage (Kaggle):
    1. Add HF_TOKEN to Kaggle Secrets
    2. Run top-to-bottom
    3. Checkpoint is auto-pushed to HF_REPO every save_steps
"""

# ---------------------------------------------------------------------------
# 0. Environment setup (MUST run before any imports)
# ---------------------------------------------------------------------------
import shutil, os

# Clear stale Unsloth compiled kernels from prior Kaggle sessions.
# Stale kernels cause silent dtype/unimplemented errors on T4.
cache_path = "/kaggle/working/unsloth_compiled_cache"
if os.path.exists(cache_path):
    shutil.rmtree(cache_path)
    print("Stale Unsloth cache cleared.")
else:
    print("No stale cache — clean start.")

# Disable TorchDynamo: conflicts with Unsloth's Triton kernels on T4 (CC 7.5)
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"]     = "1"
print("Env vars set.")

# ---------------------------------------------------------------------------
# 1. Imports (Unsloth MUST come before transformers — it patches torch ops)
# ---------------------------------------------------------------------------
import json, math, random, re, time
from typing import List, Optional

import numpy as np
import torch

# Unsloth FIRST — kernel patching must happen before any HF model loads
from unsloth import FastLanguageModel

import transformers
from transformers import TrainerCallback
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

print("Torch        :", torch.__version__)
print("Transformers :", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    dev = torch.cuda.get_device_properties(0)
    cc  = f"{dev.major}.{dev.minor}"
    print(f"GPU          : {dev.name}")
    print(f"VRAM         : {dev.total_memory / 1e9:.1f} GB")
    print(f"Compute Cap  : {cc}")
    if float(cc) < 7.5:
        raise RuntimeError("This script targets CC >= 7.5 (Turing / T4).")

from huggingface_hub import login

# ---------------------------------------------------------------------------
# 2. Auth
# ---------------------------------------------------------------------------
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    login(token=hf_token)
    print("Logged in to Hugging Face Hub")
except Exception:
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        login(token=hf_token)
        print("Logged in via HF_TOKEN env var")
    else:
        print("WARNING: No HF_TOKEN — Hub push will be disabled")
        hf_token = None

HF_REPO = "Suryansh7123/qwen2.5_lora_r16_finetune"
print(f"Checkpoints will push to: {HF_REPO}")

# ---------------------------------------------------------------------------
# 3. Config
# ---------------------------------------------------------------------------
CFG = dict(
    model_name        = "Qwen/Qwen2.5-1.5B-Instruct",
    max_seq_len       = 2048,

    # LoRA r=16 for SFT — sufficient capacity for math reasoning patterns
    lora_r            = 16,
    lora_alpha        = 8,
    lora_dropout      = 0.05,
    lora_target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias              = "none",
    use_rslora        = True,      # rank-stabilised LoRA, prevents gradient instability

    dataset_name      = "meta-math/MetaMathQA",
    train_size        = 150_000,
    val_size          = 2_000,
    seed              = 42,

    output_dir        = "/kaggle/working/qwen1b5_metamath_sft",
    num_train_epochs  = 3,
    per_device_batch  = 2,
    grad_accum        = 8,         # effective batch = 16
    learning_rate     = 2e-4,
    warmup_ratio      = 0.03,
    lr_scheduler_type = "cosine",
    weight_decay      = 0.01,
    max_grad_norm     = 1.0,

    fp16              = True,      # T4 Turing has FP16 Tensor Cores, NO BF16
    bf16              = False,

    use_gradient_checkpointing = "unsloth",
    packing           = True,

    logging_steps     = 10,
    save_steps        = 500,
    save_total_limit  = 4,
    eval_steps        = 500,
)

os.makedirs(CFG["output_dir"], exist_ok=True)
print(json.dumps({k: str(v) for k, v in CFG.items()}, indent=2))

# ---------------------------------------------------------------------------
# 4. Load model + LoRA
# ---------------------------------------------------------------------------
print("Loading model + tokeniser...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name    = CFG["model_name"],
    max_seq_length= CFG["max_seq_len"],
    dtype         = None,
    load_in_4bit  = False,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("Injecting LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r                          = CFG["lora_r"],
    target_modules             = CFG["lora_target_modules"],
    lora_alpha                 = CFG["lora_alpha"],
    lora_dropout               = CFG["lora_dropout"],
    bias                       = CFG["bias"],
    use_rslora                 = CFG["use_rslora"],
    use_gradient_checkpointing = CFG["use_gradient_checkpointing"],
    random_state               = CFG["seed"],
)

total_p     = sum(p.numel() for p in model.parameters())
trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params     : {total_p/1e6:.1f} M")
print(f"Trainable params : {trainable_p/1e6:.1f} M  ({100*trainable_p/total_p:.2f}%)")

# ---------------------------------------------------------------------------
# 5. Prompt template
# ---------------------------------------------------------------------------
# CRITICAL: This template must be IDENTICAL to the GRPO phase.
# Mismatch between SFT and RL prompt causes clipped_ratio > 90% — the model
# never generates EOS because it was not trained with that context format.
ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

def formatting_func(example) -> List[str]:
    """
    Format MetaMathQA examples as Alpaca-style prompt+response strings.
    MetaMathQA has 'query' and 'response' columns. Responses include full CoT.
    """
    outputs   = []
    queries   = example["query"]    if isinstance(example["query"],    list) else [example["query"]]
    responses = example["response"] if isinstance(example["response"], list) else [example["response"]]
    for q, r in zip(queries, responses):
        prompt = ALPACA_TEMPLATE.format(instruction=q.strip())
        full   = prompt + r.strip() + tokenizer.eos_token
        outputs.append(full)
    return outputs

# Sanity check
sample = formatting_func({"query": "What is 2+2?", "response": "The answer is 4."})
print("\nPrompt template sanity check:")
print(sample[0])

# ---------------------------------------------------------------------------
# 6. Dataset
# ---------------------------------------------------------------------------
print("\nLoading MetaMathQA...")
raw_ds = load_dataset(CFG["dataset_name"], split="train")
print(f"Total examples: {len(raw_ds):,}")

val_ratio = CFG["val_size"] / len(raw_ds)
split     = raw_ds.train_test_split(test_size=val_ratio, seed=CFG["seed"], shuffle=True)
train_raw = split["train"].select(range(CFG["train_size"]))
val_raw   = split["test"]

print(f"Train : {len(train_raw):,}  |  Val : {len(val_raw):,}")

overlap = set(train_raw["query"]) & set(val_raw["query"])
print(f"Overlap between splits: {len(overlap)}")

# ---------------------------------------------------------------------------
# 7. Keep-alive thread
# ---------------------------------------------------------------------------
import threading

def keep_alive():
    while True:
        time.sleep(30)
        with open("/kaggle/working/.keepalive", "w") as f:
            f.write(str(time.time()))

t = threading.Thread(target=keep_alive, daemon=True)
t.start()
print("Keep-alive thread running.")

# ---------------------------------------------------------------------------
# 8. Trainer
# ---------------------------------------------------------------------------
import gc
gc.collect()
torch.cuda.empty_cache()

trainer = SFTTrainer(
    model           = model,
    tokenizer       = tokenizer,
    train_dataset   = train_raw,
    eval_dataset    = val_raw,
    formatting_func = formatting_func,

    args = SFTConfig(
        output_dir                    = CFG["output_dir"],
        max_seq_length                = CFG["max_seq_len"],
        packing                       = CFG["packing"],
        dataset_num_proc              = 1,

        num_train_epochs              = CFG["num_train_epochs"],
        per_device_train_batch_size   = CFG["per_device_batch"],
        gradient_accumulation_steps   = CFG["grad_accum"],
        learning_rate                 = CFG["learning_rate"],
        weight_decay                  = CFG["weight_decay"],
        warmup_ratio                  = CFG["warmup_ratio"],
        lr_scheduler_type             = CFG["lr_scheduler_type"],
        max_grad_norm                 = CFG["max_grad_norm"],

        fp16                          = CFG["fp16"],
        bf16                          = CFG["bf16"],

        optim                         = "adamw_8bit",
        gradient_checkpointing        = True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},

        logging_steps                 = CFG["logging_steps"],
        logging_first_step            = True,

        save_strategy                 = "steps",
        save_steps                    = CFG["save_steps"],
        save_total_limit              = CFG["save_total_limit"],

        push_to_hub                   = bool(hf_token),
        hub_model_id                  = HF_REPO,
        hub_strategy                  = "checkpoint",

        eval_strategy                 = "steps",
        eval_steps                    = CFG["eval_steps"],
        per_device_eval_batch_size    = CFG["per_device_batch"] * 2,
        # NOTE: load best model manually — checkpoint-1500 is the empirically optimal
        load_best_model_at_end        = False,

        seed                          = CFG["seed"],
        remove_unused_columns         = False,
        report_to                     = "none",
    ),
)

print(f"train_dataset size : {len(trainer.train_dataset)}")
print(f"eval_dataset  size : {len(trainer.eval_dataset)}")

# ---------------------------------------------------------------------------
# 9. Train
# ---------------------------------------------------------------------------
print("\nStarting training...")
trainer_stats = trainer.train()
print("Training complete.")
print(f"Peak VRAM : {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
print(trainer_stats)

# ---------------------------------------------------------------------------
# 10. Save LoRA adapter
# ---------------------------------------------------------------------------
# IMPORTANT: The GRPO phase loads checkpoint-1500 explicitly, NOT this final adapter.
# Val loss at step 1500 = 1.822; final epoch = 1.824 (slight overfitting started).
adapter_dir = os.path.join(CFG["output_dir"], "lora_adapter")
os.makedirs(adapter_dir, exist_ok=True)
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
print(f"LoRA adapter saved to: {adapter_dir}")
print("NOTE: For GRPO, load checkpoint-1500 from the Hub, not this final adapter.")

# ---------------------------------------------------------------------------
# 11. Inference test
# ---------------------------------------------------------------------------
FastLanguageModel.for_inference(model)

test_questions = [
    "What is the sum of the first 10 natural numbers?",
    "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and uses 4 to bake muffins. She sells the rest at the market for $2 each. How much does she make per day?",
    "Solve for x: 3x + 7 = 22",
]

print("\n" + "=" * 70)
print("  POST-SFT INFERENCE TEST")
print("=" * 70)

for q in test_questions:
    prompt = ALPACA_TEMPLATE.format(instruction=q)
    enc    = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=CFG["max_seq_len"] // 2).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens = 300,
            do_sample      = False,
            temperature    = 1.0,
            pad_token_id   = tokenizer.pad_token_id,
            eos_token_id   = tokenizer.eos_token_id,
            use_cache      = True,
        )
    response = tokenizer.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\nQ: {q}")
    print(f"A: {response}")
    print("-" * 70)

print("\nDone.")
