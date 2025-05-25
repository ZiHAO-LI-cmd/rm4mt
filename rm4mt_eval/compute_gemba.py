import os
import json
from glob import glob
from tqdm import tqdm
from google import genai
from dotenv import load_dotenv
import re
import argparse
import concurrent.futures
from typing import List, Dict, Any, Optional
from functools import partial

load_dotenv()

LANG_CODE_TO_NAME = {
    "en": "English",
    "fr": "French",
    "nl": "Dutch",
    "pt": "Portuguese",
    "es": "Spanish",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "zh": "Chinese",
    "de": "German",
    "ru": "Russian",
    "it": "Italian",
}


def load_jsonl(filepath):
    with open(filepath, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_prompt(item, use_ref):
    src_lang_name = LANG_CODE_TO_NAME[item["src_lang"]]
    tgt_lang_name = LANG_CODE_TO_NAME[item["tgt_lang"]]
    src_text = item["src_text"]
    hyp_text = item["hyp_text"]
    tgt_text = item["tgt_text"]
    if use_ref == True:
        prompt = prompt = (
            f'Score the following translation from {src_lang_name} to {tgt_lang_name} with respect to the human reference on a continuous scale from 0 to 100, where a score of zero means "no meaning preserved" and score of one hundred means "perfect preservation of meaning, with faithfulness, expressiveness, and elegance".\nOnly output the score number.\n{src_lang_name} source: "{src_text}"\n{tgt_lang_name} human reference: "{tgt_text}"\n{tgt_lang_name} translation: "{hyp_text}"\nScore:'
        )
    else:
        prompt = prompt = (
            f'Score the following translation from {src_lang_name} to {tgt_lang_name} on a continuous scale from 0 to 100, where a score of zero means "no meaning preserved" and score of one hundred means "perfect preservation of meaning, with faithfulness, expressiveness, and elegance".\nOnly output the score number.\n{src_lang_name} source: "{src_text}"\n{tgt_lang_name} translation: "{hyp_text}"\nScore:'
        )
    return prompt


def compute_gemba_score(client, item: Dict[str, Any], use_ref: bool) -> Any:
    """Compute GEMBA score for a single item."""
    prompt = build_prompt(item, use_ref)
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-05-20",
        contents=prompt,
        config={"seed": 42, "stopSequences": ["\n"]},
    )
    match = re.search(r"\b(\d{1,3})\b", response.text)
    if match:
        return int(match.group(1))
    else:
        return response.text


def compute_gemba_scores(
    client, items: List[Dict[str, Any]], use_ref: bool, max_workers: int = 10
) -> List[Any]:
    """Compute GEMBA scores for multiple items in parallel."""
    scores = []
    process_func = partial(compute_gemba_score, client, use_ref=use_ref)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_func, item) for item in items]

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"Computing GEMBA scores ({'with' if use_ref else 'without'} reference)",
            unit="item",
        ):
            scores.append(future.result())

    return scores


def main(
    input_root: str,
    output_root: str,
    overwrite: bool = False,
    batch_size: int = 100,
    max_workers: int = 10,
):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    jsonl_files = glob(os.path.join(input_root, "*", "budget_*", "*.jsonl"))

    for input_path in tqdm(jsonl_files, desc="Processing files"):
        relative_path = os.path.relpath(input_path, input_root)
        output_path = os.path.join(output_root, relative_path)

        # Skip if output file exists and overwrite is False
        if os.path.exists(output_path) and not overwrite:
            print(f"⏭️ Skipping {input_path} (already processed)")
            continue

        data = load_jsonl(input_path)

        # Process data in batches to avoid overwhelming the API
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]

            gemba_scores = compute_gemba_scores(
                client, batch, use_ref=True, max_workers=max_workers
            )
            gemba_noref_scores = compute_gemba_scores(
                client, batch, use_ref=False, max_workers=max_workers
            )

            for j in range(len(batch)):
                batch[j]["gemba_score"] = gemba_scores[j]
                batch[j]["gemba_noref_score"] = gemba_noref_scores[j]

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
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files (default: skip already processed files)",
    )
    parser.add_argument(
    "--batch_size", 
    type=int, 
    default=100, 
    help="Number of items to process in a batch"
    )
    parser.add_argument(
        "--max_workers", 
        type=int, 
        default=10, 
        help="Maximum number of parallel workers"
    )
    args = parser.parse_args()

    main(args.input_root, args.output_root, args.overwrite)
