import os
import json
import argparse
from google import genai
from google.genai import types
from tqdm import tqdm
from dotenv import load_dotenv

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


def translate_dataset(
    input_dir, output_dir, model_name, api_key, thinking_budget, seed
):
    client = genai.Client(api_key=api_key)

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".jsonl"):
            continue

        src_lang, tgt_lang = filename.replace(".jsonl", "").split("-")
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename)

        already_translated = set()
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as fdone:
                for line in fdone:
                    try:
                        obj = json.loads(line)
                        already_translated.add(obj.get("src_text", "").strip())
                    except Exception:
                        continue

        with open(input_file, "r", encoding="utf-8") as fin:
            total_lines = sum(1 for _ in fin)
        with open(input_file, "r", encoding="utf-8") as fin, open(
            output_file, "a", encoding="utf-8"
        ) as fout:
            progress_bar = tqdm(
                fin, total=total_lines, desc=f"Translating {filename}", unit="examples"
            )

            for line in progress_bar:
                item = json.loads(line)
                src_text = item.get(f"{src_lang}_text", "").strip()
                if not src_text or src_text in already_translated:
                    continue

                src_lang_name = LANG_CODE_TO_NAME[src_lang]
                tgt_lang_name = LANG_CODE_TO_NAME[tgt_lang]
                prompt = f"Translate the following text from {src_lang_name} to {tgt_lang_name}\n{src_lang_name}: {src_text}\n{tgt_lang_name}: "

                # messages = [{"role": "user", "content": prompt}]

                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(
                                thinking_budget=thinking_budget
                            ),
                            seed=seed,
                        ),
                    )

                    output = {
                        "model": model_name,
                        "thinking_budget": thinking_budget,
                        "src_lang": src_lang,
                        "tgt_lang": tgt_lang,
                        "src_text": src_text,
                        "tgt_text": item.get(f"{tgt_lang}_text", "").strip(),
                        "hyp_text": response.text.strip(),
                    }
                    fout.write(json.dumps(output, ensure_ascii=False) + "\n")

                except Exception as e:
                    print("=" * 30)
                    print(f"Skipping due to error: {e}")
                    print(f"src_text: {src_text}")
                    print("=" * 30)
                    continue

        print(f"✔ Translated {filename} → saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Path to input .jsonl files")
    parser.add_argument("--output_dir", required=True, help="Directory to save results")
    parser.add_argument(
        "--model", default="gemini-2.5-flash-preview-05-20", help="Model name"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thinking_budget", type=int, default=0)
    args = parser.parse_args()

    translate_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        api_key=os.getenv("GEMINI_API_KEY"),
        thinking_budget=args.thinking_budget,
        seed=args.seed,
    )
