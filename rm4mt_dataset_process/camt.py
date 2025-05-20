import os
import json
from transformers import AutoTokenizer
import argparse


def process_camt_file(input_path, output_path, src_lang, tgt_lang, tokenizer):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(output_path, "w", encoding="utf-8") as fout:
        for entry in data:
            src_text = entry.get("source", "")
            tgt_text = entry.get("target", "")

            src_tokens = (
                tokenizer(src_text, return_length=True)["length"][0] if src_text else 0
            )
            tgt_tokens = (
                tokenizer(tgt_text, return_length=True)["length"][0] if tgt_text else 0
            )

            fout.write(
                json.dumps(
                    {
                        f"{src_lang}_tokens": src_tokens,
                        f"{tgt_lang}_tokens": tgt_tokens,
                        f"{src_lang}_text": src_text,
                        f"{tgt_lang}_text": tgt_text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"[✓] {os.path.basename(output_path)} written.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-235B-A22B")

    for fname in os.listdir(input_dir):
        if fname.endswith(".json") and fname.startswith("en-"):
            tgt_lang = fname.split("-")[1].replace(".json", "")
            src_lang = "en"

            input_path = os.path.join(input_dir, fname)
            output_path = os.path.join(output_dir, f"{src_lang}-{tgt_lang}.jsonl")

            process_camt_file(input_path, output_path, src_lang, tgt_lang, tokenizer)


if __name__ == "__main__":
    main()
