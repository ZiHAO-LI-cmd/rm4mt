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

    def __init__(self, tokenizer, max_thinking_tokens=None, enable_wait_insertion=True):
        self.tokenizer = tokenizer
        self.max_thinking_tokens = max_thinking_tokens
        self.enable_wait_insertion = enable_wait_insertion  # 是否开启强制插入wait

        # 获取关键token的ID
        self.think_start_token = self.tokenizer.encode(
            "<think>", add_special_tokens=False
        )[0]
        self.think_end_token = self.tokenizer.encode(
            "</think>", add_special_tokens=False
        )[0]
        self.nl_token = self.tokenizer.encode("\n", add_special_tokens=False)[0]
        self.wait_token = self.tokenizer.encode("wait", add_special_tokens=False)[
            0
        ]  # 延长思考的token

        # 状态跟踪
        self.tokens_generated = 0
        self.thinking_tokens_count = 0
        self.in_thinking = False
        self.stopped_thinking = False
        self.wait_inserted = False  # 跟踪是否已插入wait
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
                self.wait_inserted = False  # 重置wait状态

            # 检测思考结束
            elif last_token == self.think_end_token:
                self.in_thinking = False
                self.stopped_thinking = True

        # 如果在思考状态中，增加思考token计数
        if self.in_thinking:
            self.thinking_tokens_count += 1

        self.tokens_generated += 1

        # 处理有限制的思考token数量
        if (
            self.max_thinking_tokens is not None
            and self.max_thinking_tokens > 0
            and self.in_thinking
            and not self.stopped_thinking
        ):
            # 如果开启了wait插入功能，且模型想要结束思考但还没达到最大tokens，且还没插入过wait，就插入wait
            if (
                self.enable_wait_insertion
                and self.thinking_tokens_count < self.max_thinking_tokens
                and not self.wait_inserted
                and self.thinking_tokens_count > 5  # 至少思考5个token后才考虑插入wait
                and scores[0][self.think_end_token] == scores[0].max()
            ):  # </think>必须是概率最高的token

                # 强制选择wait token
                scores[:] = self.neg_inf
                scores[0][self.wait_token] = 0
                self.wait_inserted = True
                return scores

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


def extract_thinking_and_response(thinking_budget, text):
    """
    Extract thinking content and response from generated text
    """
    thinking_content = ""
    response_content = ""
    
    if thinking_budget == 0:
        # For budget=0, extract content between "assistant\n\n" and the next "\n\n"
        start_marker = "assistant\n\n"
        start_pos = text.find(start_marker)
        
        if start_pos != -1:
            content_start = start_pos + len(start_marker)
            # # Find the next "\n\n" after the start marker
            # end_pos = text.find("\n\n", content_start)
            # if end_pos != -1:
            #     response_content = text[content_start:end_pos].strip()
            # else:
            #     # If no ending "\n\n" found, take everything after the start marker
            #     response_content = text[content_start:].strip()
            response_content = text[content_start:].strip()
        else:
            # Fallback: if no "assistant\n\n" found, return the whole text
            response_content = text.strip()
    else:
        # For budget>0, use thinking tags
        think_start = text.find("<think>")
        think_end = text.find("</think>")

        if think_start != -1 and think_end != -1:
            thinking_content = text[think_start + 7 : think_end].strip()
            
            # Find content between "\n\n" after </think> and the next "\n\n"
            after_think_end = text[think_end + 8:]  # Skip "</think>"
            # Look for the first "\n\n" after </think>
            start_marker = "\n\n"
            start_pos = after_think_end.find(start_marker)
            
            if start_pos != -1:
                content_start = start_pos + len(start_marker)
                # Find the next "\n\n(Note" after the start marker
                end_pos = after_think_end.find("\n\n(Note", content_start)
                if end_pos != -1:
                    response_content = after_think_end[content_start:end_pos].strip()
                else:
                    # If no ending "\n\n(Note" found, take everything after the start marker
                    response_content = after_think_end[content_start:].strip()
            else:
                # Fallback: if no "\n\n" found after </think>, take everything after </think>
                response_content = after_think_end.strip()
        else:
            response_content = text.strip()

    return thinking_content, response_content


def translate_dataset(
    input_dir,
    output_dir,
    model_name,
    thinking_budget,
    max_new_tokens,
    seed,
    device_map,
    enable_wait_insertion,
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

    task = input_dir.split("/")[-1]
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
                if task == "RAGtrans":
                    sys_prompt = f"You are a professional translator, and your task is to translate an given input sentence from {src_lang_name} to {tgt_lang_name}. In addition to the input sentence, you will be provided with a document that may contain relevant information to aid in the translation. However, be aware that some documents may contain irrelevant or noisy information."
                    doc = item.get("doc", "")
                    prompt = f"<document>\n{doc}\n<document>\nTranslate the following text from {src_lang_name} to {tgt_lang_name}\n{src_lang_name}: {src_text}\n{tgt_lang_name}: "
                    messages = [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt},
                    ]
                else:
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
                            tokenizer,
                            max_thinking_tokens=thinking_budget,
                            enable_wait_insertion=enable_wait_insertion,
                        )
                        generated_ids = model.generate(
                            **model_inputs,
                            max_new_tokens=max_new_tokens,
                            logits_processor=[processor],
                        )

                    generated_text = tokenizer.decode(
                        generated_ids[0], skip_special_tokens=True
                    )

                    # Extract thinking and response
                    reasoning_content, answer_content = extract_thinking_and_response(
                        thinking_budget, generated_text
                    )

                    # Calculate thinking length
                    thinking_length = tokenizer(reasoning_content, return_length=True)[
                        "length"
                    ][0]

                    output = {
                        "model": model_name,
                        "thinking_budget": thinking_budget,
                        "thinking_length": thinking_length,
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
    parser.add_argument(
        "--model",
        default="deepcogito/cogito-v1-preview-llama-3B",
        help="Model name or path",
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
        "--enable_wait_insertion",
        action="store_true",
        help="Enable wait insertion for longer thinking",
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
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Thinking budget: {args.thinking_budget}")
    print(f"  enable_wait_insertion: {args.enable_wait_insertion}")
    print(f"  Device map: {args.device_map}")
    print(f"  Seed: {args.seed}")
    print()

    translate_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        thinking_budget=args.thinking_budget,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        device_map=args.device_map,
        enable_wait_insertion=args.enable_wait_insertion,
    )
