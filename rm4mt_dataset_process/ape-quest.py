import os
import json
from transformers import AutoTokenizer
import argparse

def process_ape_pair(src_file, ref_file, output_file, tokenizer, src_lang="en", tgt_lang="xx"):
    with open(src_file, "r", encoding="utf-8") as f_src, open(ref_file, "r", encoding="utf-8") as f_ref:
        src_lines = [line.strip() for line in f_src]
        ref_lines = [line.strip() for line in f_ref]

    assert len(src_lines) == len(ref_lines), f"Line count mismatch: {src_file} and {ref_file}"

    with open(output_file, "w", encoding="utf-8") as fout:
        for src_text, tgt_text in zip(src_lines, ref_lines):
            src_tokens = tokenizer(src_text, return_length=True)["length"][0] if src_text else 0
            tgt_tokens = tokenizer(tgt_text, return_length=True)["length"][0] if tgt_text else 0

            fout.write(json.dumps({
                f"{src_lang}_tokens": src_tokens,
                f"{tgt_lang}_tokens": tgt_tokens,
                f"{src_lang}_text": src_text,
                f"{tgt_lang}_text": tgt_text
            }, ensure_ascii=False) + "\n")

    print(f"[✓] {output_file} written.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-235B-A22B")

    for fname in os.listdir(input_dir):
        if fname.endswith(".src.txt"):
            prefix = fname.replace(".src.txt", "")
            src_path = os.path.join(input_dir, f"{prefix}.src.txt")
            ref_path = os.path.join(input_dir, f"{prefix}.ref.txt")

            src_lang = "en"
            tgt_lang = prefix.split("-")[1]

            output_file = os.path.join(output_dir, f"{src_lang}-{tgt_lang}.jsonl")

            if os.path.exists(ref_path):
                process_ape_pair(src_path, ref_path, output_file, tokenizer, src_lang, tgt_lang)

if __name__ == "__main__":
    main()
