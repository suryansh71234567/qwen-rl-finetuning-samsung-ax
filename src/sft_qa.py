"""
sft_qa.py — Phase 1 SFT for QA Track (MMLU + StrategyQA).
Direct-answer format, NO chain-of-thought.
"""
import shutil, os, json, re, time, random, threading
from typing import List, Optional
import numpy as np, torch
from unsloth import FastLanguageModel
import transformers
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login

# ── env ──────────────────────────────────────────────────────────────────────
cache_path = "/kaggle/working/unsloth_compiled_cache"
if os.path.exists(cache_path): shutil.rmtree(cache_path)
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"]     = "1"

random.seed(42); np.random.seed(42); torch.manual_seed(42)

# ── auth ─────────────────────────────────────────────────────────────────────
try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("HF_TOKEN"); login(token=hf_token)
except Exception:
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token: login(token=hf_token)
    else: hf_token = None

HF_REPO = "Suryansh7123/qwen2.5_lora_r16_finetune_STRATEGY"

# ── config ────────────────────────────────────────────────────────────────────
CFG = dict(
    model_name="Qwen/Qwen2.5-1.5B-Instruct", max_seq_len=2048,
    lora_r=16, lora_alpha=8, lora_dropout=0.05,
    lora_target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none", use_rslora=True,
    total_samples=15_000, val_size=2_000, seed=42,
    output_dir="/kaggle/working/qwen_strategy_mmlu_sft",
    num_train_epochs=3, per_device_batch=4, grad_accum=4,
    learning_rate=1e-4, warmup_ratio=0.1, lr_scheduler_type="cosine",
    weight_decay=0.05, max_grad_norm=1.0,
    fp16=True, bf16=False, use_gradient_checkpointing="unsloth", packing=True,
    logging_steps=10, save_steps=200, save_total_limit=3, eval_steps=100,
)
os.makedirs(CFG["output_dir"], exist_ok=True)

# ── model ─────────────────────────────────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    CFG["model_name"], max_seq_length=CFG["max_seq_len"], dtype=None, load_in_4bit=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
model = FastLanguageModel.get_peft_model(
    model, r=CFG["lora_r"], target_modules=CFG["lora_target_modules"],
    lora_alpha=CFG["lora_alpha"], lora_dropout=CFG["lora_dropout"],
    bias=CFG["bias"], use_rslora=CFG["use_rslora"],
    use_gradient_checkpointing=CFG["use_gradient_checkpointing"], random_state=CFG["seed"])

# ── template ─────────────────────────────────────────────────────────────────
# CRITICAL: Identical to rl_qa_no_cot.py — mismatch → clipped_ratio=93.75%
ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

def formatting_func(example) -> List[str]:
    outputs = []
    queries   = example["query"]    if isinstance(example["query"],    list) else [example["query"]]
    responses = example["response"] if isinstance(example["response"], list) else [example["response"]]
    for q, r in zip(queries, responses):
        outputs.append(ALPACA_TEMPLATE.format(instruction=q.strip()) + r.strip() + tokenizer.eos_token)
    return outputs

# ── dataset ───────────────────────────────────────────────────────────────────
strategy_ds = load_dataset("metaeval/strategy-qa", split="train")
strategy_ds = strategy_ds.map(
    lambda ex: {"query": ex["question"], "response": str(ex["answer"])},
    remove_columns=strategy_ds.column_names)

try:   mmlu_ds = load_dataset("cais/mmlu", "all", split="auxiliary_train")
except: mmlu_ds = load_dataset("cais/mmlu", "all", split="test")
def mmlu_to_qa(ex):
    opts = "\n".join([f"{chr(65+i)}. {o}" for i,o in enumerate(ex["choices"])])
    return {"query": f"{ex['question']}\n\nOptions:\n{opts}\n\nAnswer with the letter only.",
            "response": chr(65+ex["answer"])}
mmlu_ds = mmlu_ds.map(mmlu_to_qa, remove_columns=mmlu_ds.column_names)

combined = concatenate_datasets([strategy_ds, mmlu_ds]).shuffle(seed=CFG["seed"])
combined = combined.select(range(min(CFG["total_samples"], len(combined))))
val_size  = CFG["val_size"]
train_raw = combined.select(range(len(combined) - val_size))
val_raw   = combined.select(range(len(combined) - val_size, len(combined)))
print(f"Train: {len(train_raw):,}  Val: {len(val_raw):,}")

# keep-alive
threading.Thread(target=lambda: [time.sleep(30) or open("/kaggle/working/.keepalive","w").write(str(time.time())) for _ in iter(int,1)], daemon=True).start()

# ── trainer ───────────────────────────────────────────────────────────────────
import gc; gc.collect(); torch.cuda.empty_cache()

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=train_raw, eval_dataset=val_raw, formatting_func=formatting_func,
    args=SFTConfig(
        output_dir=CFG["output_dir"], max_seq_length=CFG["max_seq_len"],
        packing=CFG["packing"], dataset_num_proc=1,
        num_train_epochs=CFG["num_train_epochs"],
        per_device_train_batch_size=CFG["per_device_batch"],
        gradient_accumulation_steps=CFG["grad_accum"],
        learning_rate=CFG["learning_rate"], weight_decay=CFG["weight_decay"],
        warmup_ratio=CFG["warmup_ratio"], lr_scheduler_type=CFG["lr_scheduler_type"],
        max_grad_norm=CFG["max_grad_norm"], fp16=CFG["fp16"], bf16=CFG["bf16"],
        optim="adamw_8bit", gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=CFG["logging_steps"], logging_first_step=True,
        save_strategy="steps", save_steps=CFG["save_steps"],
        save_total_limit=CFG["save_total_limit"],
        push_to_hub=bool(hf_token), hub_model_id=HF_REPO, hub_strategy="checkpoint",
        eval_strategy="steps", eval_steps=CFG["eval_steps"],
        per_device_eval_batch_size=CFG["per_device_batch"]*2,
        load_best_model_at_end=False,
        seed=CFG["seed"], remove_unused_columns=False, report_to="none",
    ))

print("Starting SFT QA training...")
stats = trainer.train()
print(f"Done. Peak VRAM: {torch.cuda.max_memory_reserved()/1e9:.2f} GB")

adapter_dir = os.path.join(CFG["output_dir"], "lora_adapter")
os.makedirs(adapter_dir, exist_ok=True)
model.save_pretrained(adapter_dir); tokenizer.save_pretrained(adapter_dir)
print(f"Adapter saved: {adapter_dir}")
