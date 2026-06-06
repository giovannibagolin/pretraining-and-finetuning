import json
from pathlib import Path

cells = []

def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)})

def code(s):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(True)})

md('''# Fine-tune an existing ~100M model from Hugging Face on Google Colab

This notebook fine-tunes an existing small causal language model from Hugging Face.

Default model:

```text
HuggingFaceTB/SmolLM-135M
```

That model has roughly **135M parameters**, close to the requested 100M scale.

Default dataset:

```text
roneneldan/TinyStories
```

The notebook includes all code directly and supports two modes:

1. **LoRA fine-tuning** — default, cheaper and safer on Colab.
2. **Full fine-tuning** — set `USE_LORA = False`, requires more GPU memory.

Recommended Colab setting:

```text
Runtime -> Change runtime type -> GPU
```

Use T4 or better.
''')

md('''## 1. Check GPU''')
code('''!nvidia-smi
''')

md('''## 2. Install dependencies''')
code('''!pip install -q transformers datasets accelerate peft evaluate tqdm
''')

md('''## 3. Imports''')
code('''import math
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

SEED = 3407
set_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)

if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
''')

md('''## 4. Configuration

For a quick Colab smoke test, keep `MAX_TRAIN_SAMPLES` small.

For a better run, increase it or set it to `None`.
''')
code('''MODEL_ID = "HuggingFaceTB/SmolLM-135M"
DATASET_ID = "roneneldan/TinyStories"
OUTPUT_DIR = "smollm_135m_tinystories_finetuned"

# Safer on Colab. Set False for full fine-tuning.
USE_LORA = True

# Dataset/tokenization settings
BLOCK_SIZE = 512
MAX_TRAIN_SAMPLES = 20_000      # Use 2_000 for a very quick test, None for much more data
MAX_VALIDATION_SAMPLES = 1_000

# Training settings
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
EPOCHS = 1
LEARNING_RATE = 2e-4 if USE_LORA else 5e-5
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 20
EVAL_STEPS = 200
SAVE_STEPS = 200

print("MODEL_ID:", MODEL_ID)
print("USE_LORA:", USE_LORA)
print("OUTPUT_DIR:", OUTPUT_DIR)
''')

md('''## 5. Load tokenizer and base model''')
code('''tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=dtype,
    device_map="auto" if torch.cuda.is_available() else None,
)

model.config.pad_token_id = tokenizer.pad_token_id

print("Loaded model")
print("Vocab size:", len(tokenizer))
print("Pad token:", tokenizer.pad_token, tokenizer.pad_token_id)
''')

md('''## 6. Test the base model before fine-tuning''')
code('''@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=120, temperature=0.8, top_p=0.95):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

prompt = "Once upon a time, there was a little robot who wanted to"
print(generate_text(model, tokenizer, prompt))
''')

md('''## 7. Add LoRA adapters, optional

LoRA trains a small number of adapter parameters instead of updating the whole model.

For SmolLM/Llama-like models, the common target modules are:

```python
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```
''')
code('''if USE_LORA:
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
else:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Full fine-tuning: trainable {trainable / 1e6:.2f}M / {total / 1e6:.2f}M parameters")
''')

md('''## 8. Load TinyStories dataset''')
code('''raw_train = load_dataset(DATASET_ID, split="train")
raw_val = load_dataset(DATASET_ID, split="validation")

if MAX_TRAIN_SAMPLES is not None:
    raw_train = raw_train.shuffle(seed=SEED).select(range(min(MAX_TRAIN_SAMPLES, len(raw_train))))

if MAX_VALIDATION_SAMPLES is not None:
    raw_val = raw_val.shuffle(seed=SEED).select(range(min(MAX_VALIDATION_SAMPLES, len(raw_val))))

print(raw_train)
print(raw_val)
print(raw_train[0]["text"][:500])
''')

md('''## 9. Tokenize and pack text into fixed-length blocks

For causal language modeling, we concatenate many texts and split them into fixed-length token blocks.
''')
code('''def tokenize_function(examples):
    return tokenizer(examples["text"], add_special_tokens=False)

remove_columns = raw_train.column_names

tokenized_train = raw_train.map(
    tokenize_function,
    batched=True,
    remove_columns=remove_columns,
    desc="Tokenizing train",
)

tokenized_val = raw_val.map(
    tokenize_function,
    batched=True,
    remove_columns=remove_columns,
    desc="Tokenizing validation",
)


def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])

    # Drop remainder so all chunks have exactly BLOCK_SIZE tokens.
    total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE

    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_train = tokenized_train.map(group_texts, batched=True, desc="Packing train")
lm_val = tokenized_val.map(group_texts, batched=True, desc="Packing validation")

print(lm_train)
print(lm_val)
''')

md('''## 10. Data collator''')
code('''data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)
''')

md('''## 11. Training arguments''')
code('''training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    logging_steps=LOGGING_STEPS,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    bf16=False,
    report_to="none",
    seed=SEED,
    dataloader_num_workers=2,
)

training_args
''')

md('''## 12. Create Trainer''')
code('''trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_train,
    eval_dataset=lm_val,
    data_collator=data_collator,
    processing_class=tokenizer,
)
''')

md('''## 13. Evaluate before fine-tuning''')
code('''eval_before = trainer.evaluate()
print(eval_before)

if "eval_loss" in eval_before:
    print("perplexity before:", math.exp(eval_before["eval_loss"]))
''')

md('''## 14. Fine-tune''')
code('''train_result = trainer.train()
print(train_result)
''')

md('''## 15. Evaluate after fine-tuning''')
code('''eval_after = trainer.evaluate()
print(eval_after)

if "eval_loss" in eval_after:
    print("perplexity after:", math.exp(eval_after["eval_loss"]))
''')

md('''## 16. Save the fine-tuned model

If `USE_LORA=True`, this saves the LoRA adapter.  
If `USE_LORA=False`, this saves the full fine-tuned model.
''')
code('''final_dir = Path(OUTPUT_DIR) / "final"
final_dir.mkdir(parents=True, exist_ok=True)

trainer.save_model(str(final_dir))
tokenizer.save_pretrained(str(final_dir))

print("Saved to", final_dir)
!find {OUTPUT_DIR} -maxdepth 3 -type f | head -50
''')

md('''## 17. Generate with the fine-tuned model''')
code('''prompts = [
    "Once upon a time, there was a little robot who wanted to",
    "Lily and Ben went to the park and found",
    "The small dragon was scared because",
]

for p in prompts:
    print("=" * 100)
    print("PROMPT:", p)
    print(generate_text(model, tokenizer, p, max_new_tokens=150))
''')

md('''## 18. Optional: merge LoRA adapter into the base model

If you trained with LoRA, this creates a normal full Hugging Face model directory.

This can use more RAM/GPU memory. If it fails, just keep the adapter from `OUTPUT_DIR/final`.
''')
code('''if USE_LORA:
    merged_dir = Path(OUTPUT_DIR) / "merged"
    print("Merging LoRA adapter into base model...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    print("Saved merged model to", merged_dir)
else:
    print("USE_LORA=False, so there is nothing to merge.")
''')

md('''## 19. Optional: load saved adapter later

Use this if you want to reload a LoRA adapter from disk.
''')
code('''# adapter_dir = Path(OUTPUT_DIR) / "final"
# base = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID,
#     dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#     device_map="auto" if torch.cuda.is_available() else None,
# )
# loaded_model = PeftModel.from_pretrained(base, adapter_dir)
# loaded_tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
# loaded_model.eval()
# print(generate_text(loaded_model, loaded_tokenizer, "Once upon a time", max_new_tokens=100))
''')

md('''## 20. Optional: save to Google Drive

Colab runtimes are temporary. Save outputs to Drive if you want to keep them.
''')
code('''# from google.colab import drive
# drive.mount('/content/drive')
# !mkdir -p /content/drive/MyDrive/hf_100m_finetune_outputs
# !cp -r {OUTPUT_DIR} /content/drive/MyDrive/hf_100m_finetune_outputs/
''')

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

Path('notebooks').mkdir(exist_ok=True)
out = Path('notebooks/finetune_existing_100m_hf_colab.ipynb')
out.write_text(json.dumps(nb, indent=2), encoding='utf-8')
print(out)
