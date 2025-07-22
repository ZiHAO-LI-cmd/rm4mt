import os
import json
import argparse
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import httpx

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


def translate_dataset(input_dir, output_dir, model_name, base_url, api_key, extra_body):
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=httpx.Timeout(3600.0))

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
                prompt = f"Translate the following text from {src_lang_name} to {tgt_lang_name}\nPlease only provide me with the translated content, without any additional explanations\n{src_lang_name}: {src_text}\n{tgt_lang_name}: "

                messages = [{"role": "user", "content": prompt}]

                try:
                    completion = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        extra_body=extra_body,
                    )

                    reasoning_content = completion.choices[0].message.reasoning_content
                    answer_content = completion.choices[0].message.content

                    output = {
                        "model": model_name,
                        "reasoning_effort": extra_body.get("reasoning_effort", "low"),
                        "src_lang": src_lang,
                        "tgt_lang": tgt_lang,
                        "src_text": src_text,
                        "tgt_text": item.get(f"{tgt_lang}_text", ""),
                        "hyp_text": answer_content,
                        "reasoning": reasoning_content,
                    }
                    fout.write(json.dumps(output, ensure_ascii=False) + "\n")
                    fout.flush()  # Ensure data is written immediately

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
    parser.add_argument("--model", default="grok-3-mini", help="Model name")
    parser.add_argument(
        "--base_url", default="https://api.x.ai/v1"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reasoning_effort", type=str, default="low")
    args = parser.parse_args()

    extra_body = {
        "seed": args.seed,
        "reasoning_effort": args.reasoning_effort,
    }

    translate_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        base_url=args.base_url,
        api_key=os.getenv("xAI_API_KEY"),
        extra_body=extra_body,
    )
