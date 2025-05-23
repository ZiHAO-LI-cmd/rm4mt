import os
import json
import argparse
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

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
    client = OpenAI(api_key=api_key, base_url=base_url)

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".jsonl"):
            continue

        src_lang, tgt_lang = filename.replace(".jsonl", "").split("-")
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename)

        with open(input_file, "r", encoding="utf-8") as fin:
            total_lines = sum(1 for _ in fin)
        with open(input_file, "r", encoding="utf-8") as fin, open(
            output_file, "w", encoding="utf-8"
        ) as fout:
            progress_bar = tqdm(
                fin, total=total_lines, desc=f"Translating {filename}", unit="examples"
            )

            for line in progress_bar:
                item = json.loads(line)
                src_text = item.get(f"{src_lang}_text", "").strip()
                if not src_text:
                    continue

                src_lang_name = LANG_CODE_TO_NAME[src_lang]
                tgt_lang_name = LANG_CODE_TO_NAME[tgt_lang]
                prompt = f"Translate the following text from {src_lang_name} to {tgt_lang_name}\n{src_lang_name}: {src_text}\n{tgt_lang_name}: "

                messages = [{"role": "user", "content": prompt}]

                try:
                    completion = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        extra_body=extra_body,
                        stream=True,
                    )

                    reasoning_content = ""
                    answer_content = ""
                    for chunk in completion:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        if (
                            hasattr(delta, "reasoning_content")
                            and delta.reasoning_content
                        ):
                            reasoning_content += delta.reasoning_content
                        if hasattr(delta, "content") and delta.content:
                            answer_content += delta.content

                    output = {
                        "model": model_name,
                        "thinking_budget": extra_body.get("thinking_budget", 0),
                        "src_lang": src_lang,
                        "tgt_lang": tgt_lang,
                        "src_text": src_text,
                        "tgt_text": item.get(f"{tgt_lang}_text", "").strip(),
                        "hyp_text": answer_content.strip(),
                        "reasoning": reasoning_content.strip(),
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
    parser.add_argument("--model", default="qwen3-1.7b", help="Model name")
    parser.add_argument(
        "--base_url", default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thinking_budget", type=int, default=250)
    parser.add_argument(
        "--enable_thinking", action="store_true", help="Enable thinking mode"
    )
    args = parser.parse_args()

    extra_body = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
    }

    if args.enable_thinking:
        extra_body["enable_thinking"] = True
        extra_body["thinking_budget"] = args.thinking_budget
    else:
        extra_body["enable_thinking"] = False

    print(f"extra_body: {extra_body}")

    translate_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        base_url=args.base_url,
        api_key=os.getenv("ALICLOUD_API_KEY"),
        extra_body=extra_body,
    )
