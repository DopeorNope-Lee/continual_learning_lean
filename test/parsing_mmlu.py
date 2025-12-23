#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Optional: post-parse MMLU answers using a 2nd LLM + extraction template.

This mirrors the style of data_parsing_math.py / data_parsing_fin.py.
- Input: CSV from eval_mmlu.py or eval_gsm8k-like scripts (must have columns: answer, GT)
- Output: adds gen_answer (A/B/C/D/INVALID) and prints accuracy.

Usage:
  python data_parsing_mmlu.py --data_file ./mmlu_results.csv --tensor_parallel_size 4

Tip: If your generation already outputs a single letter reliably, you can skip this and just regex-parse.
"""

import argparse
import re
import pandas as pd
from typing import List

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

CHOICE_LETTERS = ("A", "B", "C", "D")


def extract_letter(text: str) -> str:
    if text is None:
        return "INVALID"
    t = text.strip()
    if "</think>" in t:
        t = t.split("</think>", 1)[1].strip()

    # Answer: B / The answer is (C) / starts with D.
    m = re.search(r"(?i)(?:the\s+answer\s+is|answer)\s*[:\-]?\s*\(?([A-D])\)?", t)
    if m:
        return m.group(1).upper()
    m = re.match(r"^\s*\(?([A-D])\)?(?:\.|\)|:|\b)", t)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-D])\b", t)
    if m:
        return m.group(1).upper()
    return "INVALID"


def generate_extract_prompt(extract_template: str, answer_text: str, tokenizer) -> str:
    filled = extract_template.format(answer_text=answer_text.strip())
    messages = [{"role": "user", "content": filled}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt += "\n- **Model's Final Answer is:** "
    return prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_file", required=True, help="CSV with columns: answer, GT")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--tensor_parallel_size", type=int, default=4)
    ap.add_argument("--template", default="./test/mmlu_parsing.txt")
    ap.add_argument("--use_second_model", action="store_true", help="if not set, do regex-only")
    args = ap.parse_args()

    df = pd.read_csv(args.data_file)

    if not args.use_second_model:
        df["gen_answer"] = [extract_letter(x) for x in df["answer"].astype(str).tolist()]
    else:
        with open(args.template, "r", encoding="utf-8") as f:
            extract_template = f.read()

        llm = LLM(
            args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=8192,
            trust_remote_code=True,
            gpu_memory_utilization=0.95,
            dtype="auto",
            enforce_eager=True,
        )
        tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")

        prompts = [generate_extract_prompt(extract_template, ans, tok) for ans in df["answer"].astype(str).tolist()]
        params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=8)
        outs = llm.generate(prompts, params)
        df["gen_answer"] = [o.outputs[0].text.strip() for o in outs]
        df["gen_answer"] = [extract_letter(x) for x in df["gen_answer"].astype(str).tolist()]

    # normalize GT
    def norm_gt(x):
        s = str(x).strip().upper()
        if s in CHOICE_LETTERS:
            return s
        # allow '#### B'
        if "####" in s:
            s2 = s.split("####", 1)[1].strip()
            if s2 and s2[0] in CHOICE_LETTERS:
                return s2[0]
        return s

    gt = [norm_gt(x) for x in df["GT"].tolist()]
    pred = df["gen_answer"].astype(str).str.strip().str.upper().tolist()

    correct = [int(p == g) for p, g in zip(pred, gt)]
    df["correct_parsed"] = correct

    acc = sum(correct) / max(1, len(correct))
    print("=" * 80)
    print(f"ACC(parsed): {acc:.4%}  (N={len(df)})")
    print("=" * 80)

    df.to_csv(args.data_file, index=False)
    print(f"Updated CSV saved: {args.data_file}")


if __name__ == "__main__":
    main()