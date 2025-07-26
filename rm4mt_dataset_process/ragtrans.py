import json
from transformers import AutoTokenizer
import argparse
from tqdm import tqdm


def process_jsonl(input_file, output_file, tokenizer):
    with open(input_file, "r", encoding="utf-8") as fin, open(
        output_file, "w", encoding="utf-8"
    ) as fout:
        for line in tqdm(fin, desc="Processing"):
            data = json.loads(line)
            en_text = data.get("src", "")
            zh_text = data.get("tgt", "")
            doc = data.get("en_doc", "")

            en_tokens = (
                tokenizer(en_text, return_length=True)["length"][0] if en_text else 0
            )
            zh_tokens = (
                tokenizer(zh_text, return_length=True)["length"][0] if zh_text else 0
            )

            new_line = {
                "en_tokens": en_tokens,
                "zh_tokens": zh_tokens,
                "en_text": en_text,
                "zh_text": zh_text,
                "doc": doc,
            }
            fout.write(json.dumps(new_line, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-235B-A22B")
    process_jsonl(args.input, args.output, tokenizer)


if __name__ == "__main__":
    main()
