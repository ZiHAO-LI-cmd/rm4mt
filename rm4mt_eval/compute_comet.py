import os
import json
from glob import glob
from tqdm import tqdm
from comet import download_model, load_from_checkpoint
import argparse


def load_jsonl(filepath):
    with open(filepath, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def compute_comet_scores(model, items, use_ref=True):
    data = []
    for item in items:
        entry = {"src": item["src_text"], "mt": item["hyp_text"]}
        if use_ref and "tgt_text" in item:
            entry["ref"] = item["tgt_text"]
        data.append(entry)
    output = model.predict(data, batch_size=8, gpus=1)
    return output.scores


def main(input_root, output_root):
    comet_model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_model_path)

    cometkiwi_model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    cometkiwi_model = load_from_checkpoint(cometkiwi_model_path)

    jsonl_files = glob(os.path.join(input_root, "*", "budget_*", "*.jsonl"))

    for input_path in tqdm(jsonl_files, desc="Processing files"):
        data = load_jsonl(input_path)

        comet_scores = compute_comet_scores(comet_model, data, use_ref=True)
        cometkiwi_scores = compute_comet_scores(cometkiwi_model, data, use_ref=False)

        for i in range(len(data)):
            data[i]["comet_score"] = comet_scores[i]
            data[i]["comet_kiwi_score"] = cometkiwi_scores[i]

        relative_path = os.path.relpath(input_path, input_root)
        output_path = os.path.join(output_root, relative_path)

        save_jsonl(data, output_path)
        print(f"✔ {input_path} → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_root", required=True, help="Path to original data root"
    )
    parser.add_argument(
        "--output_root", required=True, help="Path to write scored data"
    )
    args = parser.parse_args()

    main(args.input_root, args.output_root)
