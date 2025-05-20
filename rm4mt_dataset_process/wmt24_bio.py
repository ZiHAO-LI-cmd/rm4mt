import os
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse


def process_pair(src_lang, tgt_lang, test_path, gold_path, output_path, tokenizer):
    with open(test_path, "r", encoding="utf-8") as f_src, open(
        gold_path, "r", encoding="utf-8"
    ) as f_tgt, open(output_path, "w", encoding="utf-8") as fout:

        for src_line, tgt_line in tqdm(
            zip(f_src, f_tgt), desc=f"Processing {os.path.basename(output_path)}"
        ):
            src = src_line.strip()
            tgt = tgt_line.strip()

            src_tokens = tokenizer(src, return_length=True)["length"][0] if src else 0
            tgt_tokens = tokenizer(tgt, return_length=True)["length"][0] if tgt else 0

            fout.write(
                json.dumps(
                    {
                        f"{src_lang}_tokens": src_tokens,
                        f"{tgt_lang}_tokens": tgt_tokens,
                        f"{src_lang}_text": src,
                        f"{tgt_lang}_text": tgt,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_dir", type=str, required=True, help="Path to the test directory"
    )
    parser.add_argument(
        "--gold_dir", type=str, required=True, help="Path to the gold directory"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to the output directory"
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-235B-A22B")

    test_dir = args.test_dir
    gold_dir = args.gold_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    test_files = sorted(os.listdir(test_dir))

    for test_file in test_files:
        lang_pair = test_file.split("_")[0]  # e.g. en2de
        src_lang = test_file.split("_")[1].split(".")[0]  # e.g. en

        # test: src side, gold: tgt side
        test_path = os.path.join(test_dir, test_file)
        # gold file: swap src/tgt
        tgt_lang = lang_pair.split("2")[1]  # e.g. de
        gold_file = f"{lang_pair}_{tgt_lang}.txt"
        gold_path = os.path.join(gold_dir, gold_file)

        # output
        output_file = f"{src_lang}-{tgt_lang}.jsonl"
        output_path = os.path.join(output_dir, output_file)

        if os.path.exists(test_path) and os.path.exists(gold_path):
            process_pair(
                src_lang, tgt_lang, test_path, gold_path, output_path, tokenizer
            )
        else:
            print(f"Missing pair: {test_path} / {gold_path}")


if __name__ == "__main__":
    main()
