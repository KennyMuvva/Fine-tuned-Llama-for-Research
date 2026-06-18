import os, sys, glob, json, subprocess
# OFFLINE MODE + MEMORY ALLOCATOR FIX
os.environ["HF_HUB_OFFLINE"]            = "1"
os.environ["HF_DATASETS_OFFLINE"]       = "1"
os.environ["TOKENIZERS_PARALLELISM"]    = "false"
# Reduces memory fragmentation  directly addresses the OOM error message
os.environ["PYTORCH_CUDA_ALLOC_CONF"]   = "expandable_segments:True"
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
# GPU DETECTION  pick the best single GPU for training
def detect_gpus():
    gpus = []
    if not torch.cuda.is_available():
        print("[ERROR] No CUDA GPUs detected.")
        sys.exit(1)
    for i in range(torch.cuda.device_count()):
        props      = torch.cuda.get_device_properties(i)
        total_gb   = props.total_memory / 1024**3
        # free memory requires a dummy allocation probe
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        free_bytes = torch.cuda.mem_get_info(i)[0]
        free_gb    = free_bytes / 1024**3
        gpus.append((i, total_gb, free_gb))
        print(f"    GPU {i}: {props.name}  |  Total: {total_gb:.1f} GB  |  Free: {free_gb:.1f} GB")
    return gpus
print("\n--> Detecting GPUs...")
gpu_list = detect_gpus()
# Pick the GPU with the most FREE memory for training
best_gpu = max(gpu_list, key=lambda x: x[2])
TRAIN_GPU_INDEX = best_gpu[0]
TRAIN_GPU_FREE_GB = best_gpu[2]
TRAIN_GPU_TOTAL_GB = best_gpu[1]
print(f"\n--> Selected GPU {TRAIN_GPU_INDEX} for training "
      f"({TRAIN_GPU_TOTAL_GB:.1f} GB total, {TRAIN_GPU_FREE_GB:.1f} GB free)\n")
os.environ["CUDA_VISIBLE_DEVICES"] = str(TRAIN_GPU_INDEX)
DEVICE = "cuda:0"
# ADAPTIVE HYPERPARAMETERS based on available VRAM
if TRAIN_GPU_FREE_GB >= 38:
    BATCH_SIZE     = 4
    MAX_SEQ_LENGTH = 2048
    USE_PACKING    = True
    print(f"--> VRAM tier: 40GB+  ?  batch={BATCH_SIZE}, seq={MAX_SEQ_LENGTH}, packing={USE_PACKING}")
elif TRAIN_GPU_FREE_GB >= 18:
    BATCH_SIZE     = 2
    MAX_SEQ_LENGTH = 1024
    USE_PACKING    = True
    print(f"--> VRAM tier: 20GB+  ?  batch={BATCH_SIZE}, seq={MAX_SEQ_LENGTH}, packing={USE_PACKING}")
elif TRAIN_GPU_FREE_GB >= 9:
    BATCH_SIZE     = 1
    MAX_SEQ_LENGTH = 512
    USE_PACKING    = False
    print(f"--> VRAM tier: 10GB   ?  batch={BATCH_SIZE}, seq={MAX_SEQ_LENGTH}, packing={USE_PACKING}")
else:
    print(f"[ERROR] GPU {TRAIN_GPU_INDEX} only has {TRAIN_GPU_FREE_GB:.1f} GB free.")
    print("        LLaMA 3.1 8B in bfloat16 requires ~16 GB minimum.")
    print("        Options: use 4-bit quantization (QLoRA) or a larger GPU.")
    sys.exit(1)
LOCAL_MODEL_PATH  = "/home/asds/317/FullBaseModel/"
DATASETS_ROOT_DIR = "/home/asds/317/Dataset/Laser, Plasma & Advanced Physics/"
OUTPUT_DIR        = "/home/asds/317/fine_tuned_output_FM/fine_tuned_outputs_LPAP"
FINAL_ADAPTER_DIR = "/home/asds/317/FM_my_llama3_1_adapters/my_llama3_1_adapter_LPAP/"
PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task targeting the domain of [{}], "
    "paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{}\n\n"
    "### Input:\n{}\n\n"
    "### Response:\n{}"
)
# DATASET LOADING
def load_and_flatten_all_json(root_dir: str) -> Dataset:
    file_paths = glob.glob(os.path.join(root_dir, "**/*.json"), recursive=True)
    if not file_paths:
        print(f"[ERROR] No JSON files found under: {root_dir}")
        sys.exit(1)
    print(f"--> Found {len(file_paths)} JSON file(s).")
    flat_data = []
    for path in file_paths:
        domain_tag = os.path.basename(os.path.dirname(path))
        try:
            with open(path, "r", encoding="utf-8") as f:
                file_content = json.load(f)
        except Exception as exc:
            print(f"[WARN] Could not parse {path}: {exc}")
            continue
        entries = file_content if isinstance(file_content, list) else [file_content]
        for entry in entries:
            chapters = entry.get("chapter_summaries_and_generalized_qa", [])
            full_doc_ctx = "\n\n".join(
                f"[{ch.get('chapter_title', 'Section')}]: {ch.get('summary', '')}"
                for ch in chapters
            )
            for ch in chapters:
                specific_ctx = (
                    f"Section: {ch.get('chapter_title', 'Section')}\n"
                    f"Context: {ch.get('summary', '')}"
                )
                for qa in ch.get("generalized_qa", []):
                    q, a = qa.get("question", "").strip(), qa.get("answer", "").strip()
                    if q and a:
                        flat_data.append({"domain": domain_tag, "instruction": q,
                                          "input": specific_ctx, "output": a})
            for qa in entry.get("fifty_granular_qa", []):
                q, a = qa.get("question", "").strip(), qa.get("answer", "").strip()
                if q and a:
                    flat_data.append({"domain": domain_tag, "instruction": q,
                                      "input": full_doc_ctx.strip(), "output": a})
    if not flat_data:
        print("[ERROR] No valid QA samples found.")
        sys.exit(1)
    print(f"--> Flattened dataset: {len(flat_data):,} QA rows.")
    return Dataset.from_list(flat_data)
# TOKENIZER
print("--> Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
EOS_TOKEN = tokenizer.eos_token
def format_prompts(examples: dict) -> dict:
    texts = []
    for domain, instruction, inp, output in zip(
        examples["domain"], examples["instruction"],
        examples["input"],  examples["output"],
    ):
        text = PROMPT_TEMPLATE.format(domain, instruction, inp, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}
print("--> Loading and formatting dataset...")
raw_dataset = load_and_flatten_all_json(DATASETS_ROOT_DIR)
dataset = raw_dataset.map(format_prompts, batched=True,
                          remove_columns=raw_dataset.column_names)
# 8. MODEL LOADING  single GPU, no device_map splitting
print(f"--> Loading base model onto {DEVICE} only (no multi-GPU split)...")
# Try Flash-Attention-2 first; fall back to SDPA (Scaled Dot Product Attention)
attn_impl = "sdpa"
try:
    import importlib
    if importlib.util.find_spec("flash_attn") is not None:
        attn_impl = "flash_attention_2"
        print("    [OK] flash_attn package found  using Flash-Attention-2.")
    else:
        print("    [INFO] flash_attn not installed  using SDPA.")
        print("           (run: pip install flash-attn --no-build-isolation)")
except Exception:
    pass
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map={"": DEVICE},
    local_files_only=True,
    attn_implementation=attn_impl,
)
model.config.use_cache = False
# LoRA ADAPTER
print("--> Injecting LoRA adapters...")
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# TRAINING CONFIG  conservative and safe
training_args = SFTConfig(
    # Batch (scaled to VRAM tier detected above)
    per_device_train_batch_size   = BATCH_SIZE,
    gradient_accumulation_steps   = 4,   # compensate for small batch;
                                          # effective batch = BATCH_SIZE * 4
    # Schedule
    num_train_epochs              = 3,
    warmup_steps                  = 10,
    learning_rate                 = 2e-4,
    lr_scheduler_type             = "cosine",
    weight_decay                  = 0.01,
    # Precision
    bf16                          = True,
    fp16                          = False,
    # Optimizer  8-bit saves ~1.5 GB vs adamw_torch on 8B model
    optim                         = "adamw_8bit",
    # Sequence / packing (conservative per VRAM tier)
    dataset_text_field            = "text",
    max_length                = MAX_SEQ_LENGTH,
    packing                       = USE_PACKING,
    # Memory safety  CRITICAL on smaller GPUs
    gradient_checkpointing        = True,
    gradient_checkpointing_kwargs = {"use_reentrant": False},
    # I/O
    output_dir                    = OUTPUT_DIR,
    logging_steps                 = 5,
    save_strategy                 = "epoch",
    report_to                     = "none",
    # DataLoader  keep workers low to not exhaust CPU RAM
    dataloader_num_workers        = 2,
    dataloader_pin_memory         = True,
    seed                          = 3407,
)
# TRAIN
print("--> Initializing SFTTrainer...")
trainer = SFTTrainer(
    model            = model,
    train_dataset    = dataset,
    processing_class = tokenizer,
    args             = training_args,
)
eff_batch = BATCH_SIZE * training_args.gradient_accumulation_steps
print("\n" + "="*65)
print("  Fine-Tuning  OOM-Safe Single-GPU Mode")
print(f"  Training GPU     : GPU {TRAIN_GPU_INDEX} ({TRAIN_GPU_TOTAL_GB:.0f} GB)")
print(f"  Effective batch  : {eff_batch}  ({BATCH_SIZE} * 4 grad accum)")
print(f"  Sequence length  : {MAX_SEQ_LENGTH} tokens  (packing={USE_PACKING})")
print(f"  Attention        : {attn_impl}")
print(f"  Optimizer        : adamw_8bit")
print(f"  Alloc config     : expandable_segments=True")
print("="*65 + "\n")
trainer.train()
# SAVE
print(f"\n--> Training complete! Saving adapters ? {FINAL_ADAPTER_DIR}")
os.makedirs(FINAL_ADAPTER_DIR, exist_ok=True)
model.save_pretrained(FINAL_ADAPTER_DIR)
tokenizer.save_pretrained(FINAL_ADAPTER_DIR)
print("--> Done.")