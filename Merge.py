import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
# Configuration
BASE_MODEL = "/home/asds/317/FullBaseModel/"
OUTPUT_PATH = "/home/asds/317/local_merged_11_output"
# Mapping names to paths (Fixed to be a dictionary)
ADAPTER_MAP = {
    "BASE": "/home/asds/317/my_llama3_1_adapters/",
    "ESHP": "/home/asds/317/my_llama3_1_adapters_ESHP/",
    "GIN": "/home/asds/317/my_llama3_1_adapters_GIN/",
    "IAT": "/home/asds/317/my_llama3_1_adapters_IAT/",
    "MSM": "/home/asds/317/my_llama3_1_adapters_MSM/",
    "NRE": "/home/asds/317/my_llama3_1_adapters_NRE/",
    "PAP": "/home/asds/317/my_llama3_1_adapters_PAP/",
    "RARH": "/home/asds/317/my_llama3_1_adapters_RARH/",
    "RCS": "/home/asds/317/my_llama3_1_adapters_RCS/",
    "RMB": "/home/asds/317/my_llama3_1_adapters_RMB/",
    "WMD": "/home/asds/317/my_llama3_1_adapters_WMD/"
}
def main():
    #  Isolate the 48GB GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print("--- Starting 11-Adapter Fusion on 48GB GPU ---")
     Load Base Model onto GPU
    # Using float16 is standard, but you have 48GB, so you could even use bfloat16 if your model supports it.
    print("Loading base model to GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    # Load the first adapter
    adapter_names = list(ADAPTER_MAP.keys())
    first_name = adapter_names[0]
    print(f"Initializing PeftModel with: {first_name}")
    peft_model = PeftModel.from_pretrained(model, ADAPTER_MAP[first_name], adapter_name=first_name)
    # Load the remaining adapters
    for name in adapter_names[1:]:
        print(f"Loading adapter: {name}")
        peft_model.load_adapter(ADAPTER_MAP[name], adapter_name=name)
    # Perform the Merge
    # Equal weights for all 11
    weights = [1.0 / len(adapter_names)] * len(adapter_names
    print("Fusing adapters via add_weighted_adapter...")
    peft_model.add_weighted_adapter(
        adapters=adapter_names,
        weights=weights,
        adapter_name="combined_model",
        combination_type="linear"
    )
    # Save the merged model
    print("Finalizing merge and saving...")
    # merge_and_unload() creates a new model object with weights baked in
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_PATH)
    print(f"\nSuccess! Merged model saved to: {OUTPUT_PATH}")
if __name__ == "__main__":
    main()