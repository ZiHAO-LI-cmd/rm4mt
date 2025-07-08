import os
import json
from glob import glob
from tqdm import tqdm
from google import genai
from dotenv import load_dotenv
import re
import argparse
import concurrent.futures
from typing import List, Dict, Any
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


def build_prompt(item, scale):
    src_lang_name = LANG_CODE_TO_NAME[item["src_lang"]]
    tgt_lang_name = LANG_CODE_TO_NAME[item["tgt_lang"]]
    src_text = item["src_text"]
    hyp_text = item["hyp_text"]

    if scale == 100:
        sys_prompt = (
            f"Please evaluate the following {tgt_lang_name} translation of an {src_lang_name} text. Rate the translation on a scale of 0 to 100, where:\n"
            f"- 10 points: Poor translation; the text is somewhat understandable but contains significant errors and awkward phrasing that greatly hinder comprehension for a {tgt_lang_name} reader.\n"
            f"- 30 points: Fair translation; the text conveys the basic meaning but lacks fluency and contains several awkward phrases or inaccuracies, making it challenging for a {tgt_lang_name} reader to fully grasp the intended message.\n"
            f"- 50 points: Good translation; the text is mostly fluent and conveys the original meaning well, but may have minor awkwardness or slight inaccuracies that could confuse a {tgt_lang_name} reader.\n"
            f"- 70 points: Very good translation; the text is smooth and natural, effectively conveying the intended meaning, but may still have minor issues that could slightly affect understanding for a {tgt_lang_name} reader.\n"
            f"- 90 points: Excellent translation; the text is fluent and natural, conveying the original meaning clearly and effectively, with no significant issues that would hinder understanding for a {tgt_lang_name} reader.\n"
            f"Please only output the score number."
        )
    elif scale == 5:
        sys_prompt = (
            f"Please evaluate the following {tgt_lang_name} translation of an {src_lang_name} text. Rate the translation on a scale of 0 to 5, where:\n"
            f"- 1 point: Poor translation; the text is somewhat understandable but contains significant errors and awkward phrasing that greatly hinder comprehension for a {tgt_lang_name} reader.\n"
            f"- 2 points: Fair translation; the text conveys the basic meaning but lacks fluency and contains several awkward phrases or inaccuracies, making it challenging for a {tgt_lang_name} reader to fully grasp the intended message.\n"
            f"- 3 points: Good translation; the text is mostly fluent and conveys the original meaning well, but may have minor awkwardness or slight inaccuracies that could confuse a {tgt_lang_name} reader.\n"
            f"- 4 points: Very good translation; the text is smooth and natural, effectively conveying the intended meaning, but may still have minor issues that could slightly affect understanding for a {tgt_lang_name} reader.\n"
            f"- 5 points: Excellent translation; the text is fluent and natural, conveying the original meaning clearly and effectively, with no significant issues that would hinder understanding for a {tgt_lang_name} reader.\n"
            f"Please only output the score number."
        )
    else:
        return        

    prompt = (
        f'<text>\n{src_text}\n</text>\n<translation>\n{hyp_text}\n</translation>'
    )
    return sys_prompt, prompt


def compute_score(client, item: Dict[str, Any], scale: int) -> Any:
    """Compute GEA score for a single item."""
    sys_prompt, prompt = build_prompt(item, scale)
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={"system_instruction": sys_prompt, "temperature": 0.1, "seed": 42, "stopSequences": ["\n"]},
        )

        # Check if response.text exists and is not None
        if hasattr(response, "text") and response.text is not None:
            match = re.search(r"\b(\d{1,3})\b", response.text)
            if match:
                return int(match.group(1))
            else:
                return "No score found in response: " + response.text
        else:
            return "Empty response"
    except Exception as e:
        error_message = str(e)
        # Check if this is a quota/rate limit error
        if "429" in error_message and "RESOURCE_EXHAUSTED" in error_message:
            print("\n\nERROR: Hit Gemini API quota limit. Terminating program.")
            print(f"Error details: {error_message}")
            print("\nPlease try again later when your quota resets.")
            # Exit the program with an error code
            import sys

            sys.exit(1)
        # Return error message for other types of errors
        return f"Error: {str(e)}"


def compute_scores(
    client, items: List[Dict[str, Any]], scale: int = 5, max_workers: int = 10
) -> List[Any]:
    """Compute GEA scores for multiple items in parallel."""
    scores = []
    process_func = partial(compute_score, client, scale=scale)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_func, item) for item in items]

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"Computing GEA scores",
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

    print(f"Found {len(jsonl_files)} JSONL file")

    for input_path in tqdm(jsonl_files, desc="Processing files"):
        relative_path = os.path.relpath(input_path, input_root)
        output_path = os.path.join(output_root, relative_path)
        temp_output_path = output_path + ".temp"

        # Skip if output file exists and overwrite is False
        if os.path.exists(output_path) and not overwrite:
            print(f"Skipping {input_path} (already processed)")
            continue

        # Check if temp file exists, load it if it does
        if os.path.exists(temp_output_path):
            data = load_jsonl(temp_output_path)
            print(f"Resuming from temporary file: {temp_output_path}")
        else:
            data = load_jsonl(input_path)

        # Track which items have been processed
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]

            # Check which items in the batch need processing
            items_needing_gea_100 = [item for item in batch if "gea_100" not in item]
            items_needing_gea_5 = [item for item in batch if "gea_5" not in item]

            # Process items that need GEA100 scores
            if items_needing_gea_100:
                gea100_scores = compute_scores(
                    client, items_needing_gea_100, scale=100, max_workers=max_workers
                )
                for item, score in zip(items_needing_gea_100, gea100_scores):
                    item["gea_100"] = score

            # Process items that need GEA5 scores
            if items_needing_gea_5:
                gea5_scores = compute_scores(
                    client, items_needing_gea_5, scale=5, max_workers=max_workers
                )
                for item, score in zip(items_needing_gea_5, gea5_scores):
                    item["gea_5"] = score                    

            # Save intermediate results after each batch
            save_jsonl(data, temp_output_path)
            print(
                f"Saved progress to {temp_output_path} (batch {i//batch_size + 1}/{(len(data) + batch_size - 1)//batch_size})"
            )

        # Once all batches are processed, rename temp file to final output
        os.rename(temp_output_path, output_path)
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

    main(
        args.input_root,
        args.output_root,
        args.overwrite,
        args.batch_size,
        args.max_workers,
    )
