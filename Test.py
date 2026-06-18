import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
BASE_MODEL = "/home/asds/317/FullBaseModel/"
ADAPTER    = "/home/asds/317/FM_my_llama3_1_adapters/my_llama3_1_adapter_EIC/"
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map={"": "cuda:0"},
    attn_implementation="eager",
)
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    model,
    ADAPTER,
)
model.eval()
print("\nModel loaded successfully.")
print("GPU:", torch.cuda.get_device_name(0))
while True:
    question = input("\nQuestion (or 'quit'): ").strip()
    if question.lower() in ["quit", "exit"]:
        break
    prompt = f"""### Instruction:
{question}
### Input:
### Response:
"""
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            # max_new_tokens=512,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    print("\nAnswer:\n")
    print(response)