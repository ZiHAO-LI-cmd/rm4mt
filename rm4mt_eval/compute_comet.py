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


def compute_comet_scores(model, items, gpu_num, use_ref=True):
    print("gpu_num:", gpu_num)
    data = []
    for item in items:
        entry = {"src": item["src_text"], "mt": item["hyp_text"]}
        if use_ref and "tgt_text" in item:
            entry["ref"] = item["tgt_text"]
        data.append(entry)
    output = model.predict(data, batch_size=8, gpus=gpu_num)
    return output.scores


def main(input_root, output_root, gpu_num, overwrite=False):
    comet_model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_model_path)

    cometkiwi_model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    cometkiwi_model = load_from_checkpoint(cometkiwi_model_path)

    # Find JSONL files in budget_* and reasoning_effort_* directories
    patterns = [
        os.path.join(input_root, "*", "budget_*", "*.jsonl"),
        os.path.join(input_root, "*", "reasoning_effort_*", "*.jsonl")
    ]
    
    jsonl_files = []
    for pattern in patterns:
        jsonl_files.extend(glob(pattern))

    print(f"Found {len(jsonl_files)} files to process")

    for input_path in tqdm(jsonl_files, desc="Processing files"):
        relative_path = os.path.relpath(input_path, input_root)
        output_path = os.path.join(output_root, relative_path)

        # Skip if output file exists and overwrite is False
        if os.path.exists(output_path) and not overwrite:
            print(f"⏭️ Skipping {input_path} (already processed)")
            continue

        data = load_jsonl(input_path)

        comet_scores = compute_comet_scores(comet_model, data, gpu_num, use_ref=True)
        cometkiwi_scores = compute_comet_scores(
            cometkiwi_model, data, gpu_num, use_ref=False
        )

        for i in range(len(data)):
            data[i]["comet_score"] = comet_scores[i]
            data[i]["comet_kiwi_score"] = cometkiwi_scores[i]

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
    parser.add_argument(
        "--gpu_num", type=int, default=1, help="GPU number to use"
    )
    parser.add_argument(
        "--overwrite", action="store_true", 
        help="Overwrite existing output files (default: skip already processed files)"
    )
    args = parser.parse_args()

    main(args.input_root, args.output_root, args.gpu_num, args.overwrite)
