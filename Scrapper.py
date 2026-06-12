import os
import time
import requests
from pypdf import PdfReader


def harvest_chapterwise_newsletters():
    base_dir = "Library"
    bi_months = ['0102', '0304', '0506', '0708', '0910', '1112']

    # Sweep systematically from 2021 through 2026
    for year in range(2021, 2027):
        print(f"\n=========================================")
        print(f" PROCESSING YEAR: {year}")
        print(f"=========================================")

        for issue in bi_months:
            # Create a clean, categorized sub-folder for this specific issue
            issue_dir = os.path.join(base_dir, str(year), f"Issue_{issue}")
            os.makedirs(issue_dir, exist_ok=True)

            print(f"\n--> Checking chapters for Issue {issue}...")

            # Probe sequentially for chapters up to a reasonable max limit per issue
            for ch_num in range(1, 25):
                padded_ch = str(ch_num).zfill(2)
                file_name = f"{year}{issue}{padded_ch}.pdf"

                # Internal Campus Network URL
                url = f"http://url/{year}/{file_name}"

                try:
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    response = requests.get(url, headers=headers, timeout=10)

                    # DYNAMIC BREAK: If 404, we've hit the end of chapters for this issue
                    if response.status_code == 404:
                        if ch_num == 1:
                            print(f"    No chapters found for Issue {issue} yet (Skipping entire issue).")
                        else:
                            print(f"    Finished. Total chapters found for Issue {issue}: {ch_num - 1}")
                        break

                        # Catch other unexpected network blocks safely
                    if response.status_code != 200:
                        print(f"    [Alert] Unexpected status {response.status_code} on Ch {padded_ch}. Skipping file.")
                        continue

                    # Pathing targets
                    pdf_path = os.path.join(issue_dir, file_name)
                    txt_path = os.path.join(issue_dir, f"{year}{issue}{padded_ch}_text.txt")

                    # 1. Commit the individual Chapter PDF to the issue folder
                    with open(pdf_path, 'wb') as pdf_file:
                        pdf_file.write(response.content)

                    # 2. Extract and index text data from the chapter PDF layers
                    try:
                        reader = PdfReader(pdf_path)
                        extracted_text = []
                        for idx, page in enumerate(reader.pages):
                            text = page.extract_text()
                            if text:
                                extracted_text.append(f"--- Page {idx + 1} ---\n{text}")

                        with open(txt_path, 'w', encoding='utf-8') as txt_file:
                            txt_file.write(f"Newsletter | Year: {year} | Issue: {issue} | Chapter: {padded_ch}\n")
                            txt_file.write(f"Source Link: {url}\n")
                            txt_file.write("=" * 60 + "\n\n")
                            txt_file.write("\n\n".join(extracted_text))

                        print(f"    [Saved] Chapter {padded_ch} -> PDF & Text index generated.")
                    except Exception as parse_err:
                        print(f"    [Saved] Chapter {padded_ch} -> PDF saved, but text extraction failed: {parse_err}")

                    # Standard courtesy delay to keep requests stable
                    time.sleep(0.5)

                except requests.exceptions.RequestException as e:
                    print(f"    [Network Error] Link failed for Chapter {padded_ch}: {e}")
                    # If the server is down or dropping connections completely, skip to next issue
                    break


if __name__ == "__main__":
    harvest_chapterwise_newsletters()