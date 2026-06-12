import os
import re
import json
import requests
import fitz  # PyMuPDF

# --- CONFIGURATION ---
API_URL = "http://10.35.98.7:8430/v1/chat/completions"
MODEL_NAME = "/gemma4-26B-A4b"  # Exact ID from your server configuration


INPUT_ROOT = "F:\Scrape\BARC_Classified_Library"
OUTPUT_ROOT = "F:\Scrape\Jsons"

TARGET_GRANULAR_QA = 50
CHUNK_SIZE = 2000  # Size of text blocks for precise QA extraction
MACRO_SECTION_SIZE = 15000  # Approx size for fallback chapters if no regex match found


def extract_text_from_pdf(pdf_path):
    """Extracts raw text string from target PDF path."""
    try:
        doc = fitz.open(pdf_path)
        full_text = []
        for page in doc:
            text = page.get_text()
            if text:
                full_text.append(text)
        return "\n".join(full_text)
    except Exception as e:
        print(f"  [Error] Failed to read PDF {os.path.basename(pdf_path)}: {e}")
        return ""


def identify_chapters(text):
    """Splits text by matching chapter or section headers.

    Falls back to block chunking if no explicit markers exist.
    """
    # Regex matching lines starting with Chapter/Section/Module/Part followed by numbers/numerals
    chapter_regex = r'(?m)^(?:\s*)(CHAPTER|SECTION|MODULE|PART)\s+(\d+|[IVXLCDM]+)\b'

    matches = list(re.finditer(chapter_regex, text, re.IGNORECASE))

    chapters = []
    if len(matches) > 1:
        for i in range(len(matches)):
            start = matches[i].start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            # Extract first line as tentative title
            heading_line = text[start:end].split('\n')[0].strip()
            chapters.append({
                "title": heading_line if heading_line else f"Section_{i + 1}",
                "content": text[start:end].strip()
            })
    else:
        # Fallback layout split if text contains no clean regex milestones
        print("  [Info] No explicit chapter patterns identified. Generating structural macro-sections...")
        words = text.split()
        segment_count = 1
        for i in range(0, len(words), MACRO_SECTION_SIZE // 5):
            chunk_words = words[i:i + (MACRO_SECTION_SIZE // 5)]
            if chunk_words:
                chapters.append({
                    "title": f"Document Section {segment_count}",
                    "content": " ".join(chunk_words)
                })
                segment_count += 1

    return chapters


def generate_chapter_summary_and_qa(chapter_title, chapter_content):
    """Asks LLM to synthesize a thorough text summary and generalized conceptual Q&As."""
    prompt = f"""
Analyze the technical document segment provided below titled "{chapter_title}".
Perform two sequential tasks, outputting the result strictly as a valid JSON object. Do not deviate from the text facts.

Expected JSON output format:
{{
  "summary": "A deep, precise multi-paragraph technical summary covering the core concepts, designs, architectures, or datasets mentioned.",
  "generalized_qa": [
    {{"question": "A high-level conceptual question about this section's core topic?", "answer": "A long-form, thoroughly detailed answer synthesizing the textual facts."}},
    {{"question": "Another broad structural question?", "answer": "Another detailed structured response based on the document facts."}}
  ]
}}

Context Text:
{chapter_content[:12000]}  # Safety context cap to protect memory window boundaries
"""

    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 1500
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        raw_content = response.json()['choices'][0]['message']['content'].strip()

        if raw_content.startswith("```"):
            raw_content = re.sub(r'^```json\s*|```$', '', raw_content, flags=re.MULTILINE)

        return json.loads(raw_content)
    except Exception as e:
        print(f"  [Error] Failed parsing summary for chapter {chapter_title}: {e}")
        return {"summary": "Failed to generate summary.", "generalized_qa": []}


def generate_granular_qa_from_chunk(text_chunk, count):
    """Generates highly specific, strictly bounded, long-form QA pairs from small data text blocks."""
    prompt = f"""
You are an expert data tuning engine. Extract exactly {count} highly detailed and long-form question-and-answer pairs based directly on the text block below.

RULES:
1. LONG & PRECISE: Make the answers comprehensive and deeply technical.
2. ZERO DEVIATION: Do not extrapolate or add outside knowledge. If the exact answer is not explicitly written in the context, do not invent it.
3. OUTPUT FORMAT: Respond ONLY with a raw JSON array. No markdown code blocks.

[
  {{
    "question": "Highly specific technical question extracted from text?",
    "answer": "Long-form, detailed, precise answer mirroring text documentation metrics or guidelines exactly."
  }}
]

Context Block:
{text_chunk}
"""

    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,  # Low variance to guarantee zero deviation rules
        "max_tokens": 1500
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        raw_content = response.json()['choices'][0]['message']['content'].strip()

        if raw_content.startswith("```"):
            raw_content = re.sub(r'^```json\s*|```$', '', raw_content, flags=re.MULTILINE)

        return json.loads(raw_content)
    except Exception as e:
        return []


def chunk_text(text, chunk_size=2000):
    """Splits full text uniformly for the 50 granular Q&As generation stage."""
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    chunks = []
    overlap = 200
    start = 0
    while start < len(cleaned_text):
        end = start + chunk_size
        chunks.append(cleaned_text[start:end])
        start += (chunk_size - overlap)
    return chunks


def process_pdf(pdf_path, output_dir):
    pdf_name = os.path.basename(pdf_path)
    base_name = os.path.splitext(pdf_name)[0]
    output_file_path = os.path.join(output_dir, f"{base_name}_comprehensive_dataset.json")

    if os.path.exists(output_file_path):
        print(f"Skipping {pdf_name} (Dataset target already constructed).")
        return

    print(f"\n=========================================\nStarting Extraction Pipeline: {pdf_name}")
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text.strip():
        return

    # --- STAGE 1: CHAPTER SUMMARIES & GENERALIZED QA ---
    print("Executing Stage 1: Identifying Chapters & Constructing Thematic Summaries...")
    chapters = identify_chapters(raw_text)
    processed_chapters_summary = []

    for ch in chapters:
        print(f"  -> Synthesizing summary & generalized topics for: {ch['title']}")
        analysis = generate_chapter_summary_and_qa(ch['title'], ch['content'])
        processed_chapters_summary.append({
            "chapter_title": ch['title'],
            "summary": analysis.get("summary", ""),
            "generalized_qa": analysis.get("generalized_qa", [])
        })

    # --- STAGE 2: 50 GRANULAR EXPLICIT QA PAIRS ---
    print("Executing Stage 2: Compiling 50 Long & Precise Granular QA Pairs...")
    chunks = chunk_text(raw_text, chunk_size=CHUNK_SIZE)
    granular_qa_list = []

    for idx, chunk in enumerate(chunks):
        needed = TARGET_GRANULAR_QA - len(granular_qa_list)
        if needed <= 0:
            break

        request_batch_size = min(5, needed)
        print(f"  -> Mining chunk {idx + 1}/{len(chunks)} (Progress: {len(granular_qa_list)}/{TARGET_GRANULAR_QA})")

        extracted_pairs = generate_granular_qa_from_chunk(chunk, request_batch_size)
        for qa in extracted_pairs:
            if "question" in qa and "answer" in qa:
                granular_qa_list.append({
                    "question": qa["question"].strip(),
                    "answer": qa["answer"].strip()
                })
                if len(granular_qa_list) >= TARGET_GRANULAR_QA:
                    break

    # --- STAGE 3: MERGE AND SAVE ACCORDING TO SCHEMA ---
    final_output_structure = {
        "source_file": pdf_name,
        "chapter_summaries_and_generalized_qa": processed_chapters_summary,
        "fifty_granular_qa": granular_qa_list
    }

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(final_output_structure, f, indent=2, ensure_ascii=False)
    print(f"[Success] Saved finalized comprehensive output data structure to:\n  -> {output_file_path}")


def main():
    if not os.path.exists(INPUT_ROOT):
        print(f"Error: Specified root input directory missing: {INPUT_ROOT}")
        return

    # 1. Iterate through everything inside the BARC_Classified_Library folder
    for folder_name in os.listdir(INPUT_ROOT):
        input_folder_path = os.path.join(INPUT_ROOT, folder_name)

        # 2. Make sure it's actually a subfolder (and not a loose file)
        if os.path.isdir(input_folder_path):
            print(f"\n=======================================================")
            print(f" NOW PROCESSING TOPIC: {folder_name.upper()} ")
            print(f"=======================================================")

            # 3. Create a matching destination subfolder in your output root
            output_folder_path = os.path.join(OUTPUT_ROOT, folder_name)
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
                print(f"Created dynamic save directory: {output_folder_path}")

            # 4. Find all PDFs specifically inside this topic subfolder
            pdf_files = [
                os.path.join(input_folder_path, f)
                for f in os.listdir(input_folder_path)
                if f.lower().endswith('.pdf')
            ]

            print(f"Discovered {len(pdf_files)} candidate PDFs inside '{folder_name}'.")

            # 5. Run your extraction pipeline for each PDF found in this folder
            for path in pdf_files:
                process_pdf(path, output_folder_path)

    print("\n[Finished] All folders and batch datasets have been processed successfully!")


if __name__ == "__main__":
    main()