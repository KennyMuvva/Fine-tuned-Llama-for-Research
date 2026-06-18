### 1. Automated PDF Collection & Extraction
* Scraping: A custom Python script designed to systematically download documentation, technical manuals, and PDFs from targeted website.

### 2. Multi Topic Document Classification
* Extracted document datasets are run through an automated thematic classifier(llama3.1) that aggregates and segments the data into **11 unique domain/subject matter categories**.
* This structural segmentation builds the isolated foundation required to train crisp, domain-specific localized expert networks.

### 3. Synthetic Data Engineering (Via Gemma)
* To map raw text into rich instruction tuning structures, a linguistic pipeline leverages **Gemma** to analyze the thematic document payloads.
* Gemma automatically handles context chunking and features synthesis to construct pristine, complex Question and Answer training pairs, which are exported directly into standardized, clean JSON instruction schemas.

### 4. Specialized Multi Adapter LoRA Finetuning (Llama 3.1)
* **Base Architecture:** Llama 3.1 (8B Parameter Architecture).
* **Training Strategy:** Rather than tuning a single model on all datasets which degrades edge reasoning and causes thematic cross contamination **11 specialized LoRA adapters** are trained independently using the corresponding subject matter JSON datasets. 
* Base weights are locked in a low precision matrix to optimize training VRAM footprint, while updating adapter matrices in native high precision format.

### 5. High Precision Local Weight Fusion (48GB RAM Workstation)
* To protect the fine tuning intelligence against severe matrix truncation errors, the 11 specialized adapter weights are mathematically fused back into the **pristine 16GB unquantized FP16 Base Model**.
* Because loading and blending an 8B parameters model causes a temporary peak memory spike of **32GB to 34GB RAM**, a dedicated 48GB RAM workstation runs the script natively on system memory to prevent instant Windows Out of Memory (OOM) terminal closures.

### 6. Cloud GGUF Packaging & Quantizatized
* The combined, full precision Safetensors shards are processed in 48GB workstation.
* Utilizing `llama.cpp`, the model configurations are consolidated into a single `.gguf` file container (`--outtype f16`).
* The unified file is compressed down using a 4-bit quantization layout (`Q4_K_M`) to balance ultra-low compute overhead with complete retention of the model's new fine tuned knowledge.
