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
    try:
        response = client.models.generate_content(
            # model="gemini-2.5-flash-preview-05-20",
            model="gemini-2.0-flash",
            contents=prompt,
            config={"seed": 42, "stopSequences": ["\n"]},
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
            items_needing_gemba = [item for item in batch if "gemba_score" not in item]
            items_needing_gemba_noref = [
                item for item in batch if "gemba_noref_score" not in item
            ]

            # Process items that need gemba scores
            if items_needing_gemba:
                gemba_scores = compute_gemba_scores(
                    client, items_needing_gemba, use_ref=True, max_workers=max_workers
                )
                for item, score in zip(items_needing_gemba, gemba_scores):
                    item["gemba_score"] = score

            # Process items that need gemba_noref scores
            if items_needing_gemba_noref:
                gemba_noref_scores = compute_gemba_scores(
                    client,
                    items_needing_gemba_noref,
                    use_ref=False,
                    max_workers=max_workers,
                )
                for item, score in zip(items_needing_gemba_noref, gemba_noref_scores):
                    item["gemba_noref_score"] = score

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
