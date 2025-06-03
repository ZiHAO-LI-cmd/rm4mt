import os
import json
import argparse
import torch
from transformers import LogitsProcessor, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

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


class ThinkingTokenBudgetProcessor(LogitsProcessor):
    """
    A processor where after a maximum number of tokens are generated,
    a </think> token is added at the end to stop the thinking generation,
    and then it will continue to generate the response.
    """

    def __init__(self, tokenizer, max_thinking_tokens=None):
        self.tokenizer = tokenizer
        self.max_thinking_tokens = max_thinking_tokens

        # 获取关键token的ID
        self.think_start_token = self.tokenizer.encode(
            "<think>", add_special_tokens=False
        )[0]
        self.think_end_token = self.tokenizer.encode(
            "</think>", add_special_tokens=False
        )[0]
        self.nl_token = self.tokenizer.encode("\n", add_special_tokens=False)[0]

        # 状态跟踪
        self.tokens_generated = 0
        self.thinking_tokens_count = 0
        self.in_thinking = False
        self.stopped_thinking = False
        self.neg_inf = float("-inf")

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # 检查当前是否在思考状态中
        if input_ids.shape[1] > 0:
            last_token = input_ids[0, -1].item()

            # 检测思考开始
            if last_token == self.think_start_token:
                self.in_thinking = True
                self.thinking_tokens_count = 0

            # 检测思考结束
            elif last_token == self.think_end_token:
                self.in_thinking = False
                self.stopped_thinking = True

        # 如果在思考状态中，增加思考token计数
        if self.in_thinking:
            self.thinking_tokens_count += 1

        self.tokens_generated += 1

        # 处理 max_thinking_tokens=0 的情况：禁止思考
        if self.max_thinking_tokens == 0 and not self.stopped_thinking:
            # 如果在思考状态中，立即结束（不允许任何思考内容）
            if self.in_thinking:
                # 强制立即结束思考：换行 + </think>
                if self.thinking_tokens_count == 0:
                    # 第一个思考token：强制换行
                    scores[:] = self.neg_inf
                    scores[0][self.nl_token] = 0
                else:
                    # 第二个思考token：强制结束
                    scores[:] = self.neg_inf
                    scores[0][self.think_end_token] = 0
                    self.stopped_thinking = True
                return scores

        # 处理有限制的思考token数量
        if (
            self.max_thinking_tokens is not None
            and self.max_thinking_tokens > 0
            and self.in_thinking
            and not self.stopped_thinking
        ):
            # 当接近token限制时，增加结束思考的概率
            if self.thinking_tokens_count >= self.max_thinking_tokens * 0.95:
                # 增强换行和结束思考token的概率
                boost_factor = 1 + (
                    self.thinking_tokens_count / self.max_thinking_tokens
                )
                scores[0][self.nl_token] = scores[0][self.nl_token] * boost_factor
                scores[0][self.think_end_token] = (
                    scores[0][self.think_end_token] * boost_factor
                )

            # 强制结束思考
            if self.thinking_tokens_count >= self.max_thinking_tokens - 1:
                if self.thinking_tokens_count == self.max_thinking_tokens - 1:
                    # 倒数第二个token，优先选择换行
                    scores[:] = self.neg_inf
                    scores[0][self.nl_token] = 0
                else:
                    # 最后一个token，强制选择结束思考
                    scores[:] = self.neg_inf
                    scores[0][self.think_end_token] = 0
                    self.stopped_thinking = True

        return scores


def extract_thinking_and_response(text):
    """
    Extract thinking content and response from generated text
    """
    thinking_content = ""
    response_content = ""

    # Find thinking tags
    think_start = text.find("<think>")
    think_end = text.find("</think>")

    if think_start != -1 and think_end != -1:
        thinking_content = text[think_start + 7 : think_end].strip()
        response_content = text[think_end + 8 :].strip()
    else:
        response_content = text.strip()

    return thinking_content, response_content


def translate_dataset(
    input_dir,
    output_dir,
    model_name,
    thinking_budget,
    temperature,
    top_p,
    max_new_tokens,
    seed,
    device_map,
):
    """
    Translate dataset using local model inference
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Load tokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Determine device configuration
    if device_map:
        print(f"Using device_map: {device_map}")
        device = None  # device should be None when using device_map
    else:
        device = 0 if torch.cuda.is_available() else -1
        print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )

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
                messages = [{"role": "user", "content": prompt}]

                try:
                    if thinking_budget == 0:
                        text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                        model_inputs = tokenizer([text], return_tensors="pt").to(
                            model.device
                        )
                        generated_ids = model.generate(
                            **model_inputs,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                        )
                    else:
                        text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=True,
                        )
                        model_inputs = tokenizer([text], return_tensors="pt").to(
                            model.device
                        )
                        processor = ThinkingTokenBudgetProcessor(
                            tokenizer, max_thinking_tokens=thinking_budget
                        )
                        generated_ids = model.generate(
                            **model_inputs,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            logits_processor=[processor],
                        )

                    generated_text = tokenizer.decode(
                        generated_ids[0], skip_special_tokens=True
                    )

                    # Extract thinking and response
                    reasoning_content, answer_content = extract_thinking_and_response(
                        generated_text
                    )

                    output = {
                        "model": model_name,
                        "thinking_budget": thinking_budget,
                        "src_lang": src_lang,
                        "tgt_lang": tgt_lang,
                        "src_text": src_text,
                        "tgt_text": item.get(f"{tgt_lang}_text", ""),
                        "hyp_text": answer_content,
                        "reasoning": reasoning_content,
                        "all_generated_text": generated_text,
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
    parser = argparse.ArgumentParser(
        description="Local model inference for translation evaluation"
    )
    parser.add_argument("--input_dir", required=True, help="Path to input .jsonl files")
    parser.add_argument("--output_dir", required=True, help="Directory to save results")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B", help="Model name or path")
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Temperature for generation"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95, help="Top-p for generation"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1500,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--thinking_budget",
        type=int,
        default=0,
        help="Maximum thinking tokens (0 to disable)",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        help="Device map for multi-GPU (e.g., 'auto', 'balanced', or custom mapping)",
    )

    args = parser.parse_args()

    # Parse device_map if it's a string representation of a dict
    if args.device_map and args.device_map not in [
        "auto",
        "balanced",
        "balanced_low_0",
        "sequential",
    ]:
        try:
            # Try to parse as JSON for custom device mapping
            import ast

            args.device_map = ast.literal_eval(args.device_map)
        except:
            # If parsing fails, use as string
            pass

    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Input directory: {args.input_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-p: {args.top_p}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Thinking budget: {args.thinking_budget}")
    print(f"  Device map: {args.device_map}")
    print(f"  Seed: {args.seed}")
    print()

    translate_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        thinking_budget=args.thinking_budget,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        device_map=args.device_map,
    )
