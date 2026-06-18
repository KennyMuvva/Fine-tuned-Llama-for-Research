import os
import shutil
import requests
import json
# Define the exact 11 allowed categories
CATEGORIES = [
    "Nuclear Reactors & Engineering",
    "Radiochemistry & Chemical Sciences",
    "Materials Science & Metallurgy",
    "Radiation Medicine & Biosciences",
    "Electronics, Instrumentation & Computing",
    "Environmental Sciences & Health Physics",
    "Laser, Plasma & Advanced Physics",
    "Isotope Applications & Technology",
    "Robotics, Automation & Remote Handling",
    "Waste Management & Decommissioning",
    "General & Institutional News"
]
def query_local_llama(title_text):
    """Sends the title to a local Ollama instance and returns a verified category."""
    url = "http://localhost:11434/api/generate"
    # Construct a highly rigid prompt to prevent conversational fluff from the LLM
    categories_formatted = "\n".join([f"- {cat}" for cat in CATEGORIES])
    prompt = f"""You are a strict data classification assistant at a nuclear research archive.
Your task is to classify a scientific newsletter article based ONLY on its title and header information.
CRITICAL INSTRUCTION: You must reply with EXACTLY ONE category string from the approved list below. Do NOT include markdown tags, introduction, punctuation, reasoning, or explanations. Your entire response must match one of these lines word-for-word.
[APPROVED CATEGORY LIST]
{categories_formatted}
[ARTICLE TO CLASSIFY]
{title_text}
Your exact single-line category match:"""
    payload = {
        "model": "my-custom-model",  # Switch to your preferred local model string if different
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json().get('response', '').strip()
            # Remove potential edge-case surrounding quotes the LLM might include
            result = result.replace('"', '').replace("'", "").strip()
            # Verify the LLM response perfectly matches one of our targets
            for category in CATEGORIES:
                if result.lower() == category.lower() or category.lower() in result.lower():
                    return category
            print(f"    [Warning] LLM output '{result}' did not exactly match category list. Defaulting to General.")
            return "General & Institutional News"
    except Exception as e:
        print(f"    [Error] Failed to connect to local Ollama API: {e}")
        return "General & Institutional News"
def organize_archive_by_ai():
    # Paths based on our previous download setup
    source_library = "F:\Scrape\Library"
    destination_library = "F:\Scrape\Classified_Library"
    if not os.path.exists(source_library):
        print(f"Source folder '{source_library}' not found. Please run the downloader first.")
        return
    # Walk recursively through all files in the downloaded library
    for root, dirs, files in os.walk(source_library):
        for file in files:
            # We use the text files to extract titles and drive the logic
            if file.endswith("_text.txt"):
                txt_path = os.path.join(root, file)
                # Deduce the matching PDF filename
                pdf_file_name = file.replace("_text.txt", ".pdf")
                pdf_path = os.path.join(root, pdf_file_name)
                print(f"\nProcessing file: {file}")
                # 1. Read the first few lines of the text file to grab title/header data
                header_snippet = ""
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        lines = [f.readline() for _ in range(8)]
                        header_snippet = "".join(lines).strip()
                except Exception as e:
                    print(f"   Failed to read text snippet: {e}")
                    continue
                if not header_snippet:
                    print("   File empty. Skipping.")
                    continue
                # 2. Let LLaMA classify the text content snippet
                print("   Analyzing content with local LLaMA...")
                assigned_category = query_local_llama(header_snippet)
                print(f"   AI Classification Resolution: ---> [{assigned_category}]")
                # 3. Create the physical folder destination path
                target_folder = os.path.join(destination_library, assigned_category)
                os.makedirs(target_folder, exist_ok=True)
                # 4. Safely move the files into their newly classified home
                try:
                    # Move Text Index
                    shutil.move(txt_path, os.path.join(target_folder, file))
                    # Move companion PDF if it exists
                    if os.path.exists(pdf_path):
                        shutil.move(pdf_path, os.path.join(target_folder, pdf_file_name))
                    print(f"   [SUCCESS] Re-routed items to: {target_folder}")
                except Exception as file_err:
                    print(f"   [Error] Failed to move files: {file_err}")
if __name__ == "__main__":
    organize_archive_by_ai()