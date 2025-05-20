import json
import re
from transformers import AutoTokenizer
import argparse
import concurrent.futures
import threading


tokenizer = None
tokenizer_lock = threading.Lock()

# Function to parse XML file and extract text segments
def extract_text_from_xml(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # Regular expression to extract book and chapter text
        books = re.findall(r'<BOOK id="(\d+)">.*?<CHAPTER id="(\d+)">(.*?)</CHAPTER>.*?</BOOK>', content, re.DOTALL)

        extracted_text = []
        for book_id, chapter_id, text in books:
            cleaned_text = text.strip().replace('\n', ' ').replace('\t', ' ').strip()
            extracted_text.append((book_id, chapter_id, cleaned_text))

        return extracted_text
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []


def process_files(file_x, file_zh):
    try:
        print(
            f"Thread {threading.current_thread().name} is processing {file_x} and {file_zh}"
        )
        global tokenizer
        with tokenizer_lock:
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-235B-A22B")

        x = file_x.split('/')[-1].split('.')[-1]
        output_file = f"../rm4mt_dataset/WMT-Literary/zh-{x}.jsonl"

        with open(output_file, "w", encoding="utf-8") as jsonl_file:
            x_texts = extract_text_from_xml(file_x)
            zh_texts = extract_text_from_xml(file_zh)

            # Ensure both files have the same number of entries
            assert len(x_texts) == len(zh_texts), f"Mismatch in book/chapter pairs between {file_x} and {file_zh}"

            # Combine the texts into JSONL format
            for (x_book_id, x_chapter_id, x_text), (y_book_id, y_chapter_id, y_text) in zip(x_texts, zh_texts):
                assert x_book_id == y_book_id and x_chapter_id == y_chapter_id, "Mismatched book/chapter IDs"

                x_tokens = tokenizer(x_text, return_length=True)["length"][0] if x_text else 0
                y_tokens = tokenizer(y_text, return_length=True)["length"][0] if y_text else 0

                json_line = {
                    # "book_id": x_book_id,
                    # "chapter_id": x_chapter_id,
                    f"{x}_tokens": x_tokens,
                    "zh_tokens": y_tokens,                
                    f"{x}_text": x_text,
                    "zh_text": y_text,
                }
                jsonl_file.write(json.dumps(json_line, ensure_ascii=False) + "\n")

            print(f"Conversion completed. JSONL file saved as {output_file}.")
    except Exception as e:
        print(f"Error processing {file_x} and {file_zh}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=1, help="Number of threads to use for processing")
    args = parser.parse_args()

    # Initialize Qwen3-235B-A22B tokenizer
    global tokenizer
    with tokenizer_lock:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-235B-A22B")

    # Extract text from multiple XML file pairs
    file_pairs = [
        ("../rm4mt_dataset/WMT-Literary/V1/VALID/valid.en", "../rm4mt_dataset/WMT-Literary/V1/VALID/valid.zh"),
        ("../rm4mt_dataset/WMT-Literary/V2/VALID_1_Chinese-German/valid.de", "../rm4mt_dataset/WMT-Literary/V2/VALID_1_Chinese-German/valid.zh"),
        ("../rm4mt_dataset/WMT-Literary/V2/VALID_1_Chinese-Russian/valid.ru", "../rm4mt_dataset/WMT-Literary/V2/VALID_1_Chinese-Russian/valid.zh")
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        for file_x, file_zh in file_pairs:
            executor.submit(process_files, file_x, file_zh)


if __name__ == "__main__":
    main()
