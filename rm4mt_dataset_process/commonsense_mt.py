import csv
import json
import argparse
import os
from transformers import AutoTokenizer
from tqdm import tqdm


def process_csv(input_csv, output_dir, tokenizer):
    os.makedirs(output_dir, exist_ok=True)
    with open(input_csv, "r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        for row in tqdm(reader, desc="Processing CSV"):
            src = row.get("chinese_source", "").strip()
            tgt = row.get("english_target_correct", "").strip()
            src_lang, tgt_lang = "zh", "en"
            src_tokens = tokenizer(src, return_length=True)["length"][0] if src else 0
            tgt_tokens = tokenizer(tgt, return_length=True)["length"][0] if tgt else 0
            data = {
                f"{src_lang}_tokens": src_tokens,
                f"{tgt_lang}_tokens": tgt_tokens,
                f"{src_lang}_text": src,
                f"{tgt_lang}_text": tgt,
            }
            output_file = os.path.join(output_dir, f"{src_lang}-{tgt_lang}.jsonl")
            with open(output_file, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv", type=str, required=True, help="Path to input CSV file"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to output directory"
    )
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-235B-A22B")
    process_csv(args.input_csv, args.output_dir, tokenizer)


if __name__ == "__main__":
    main()
