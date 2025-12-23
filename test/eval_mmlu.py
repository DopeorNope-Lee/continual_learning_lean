#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MMLU evaluation (vLLM inference + exact-match scoring).

- JSONL 테스트 파일을 읽고
- chat_template 있으면 그걸로 prompt 구성 (없으면 instruction wrapper)
- vLLM batched inference
- 출력에서 A/B/C/D를 robust하게 추출
- GT와 비교해서 accuracy 계산
- 기존 파이프라인처럼 CSV 저장 (query, answer, GT 포함)

지원 JSONL 포맷:

Schema A ("bench style"):
  {"query": "...question+choices...", "response": "#### B. ...", "subject": "high_school_physics"}

Schema B ("structured"):
  {"subject": "high_school_physics", "question": "...", "choices": ["...","...","...","..."], "answer": 1}
    - answer: 0-3 또는 "A"-"D"

Few-shot(MMLU 표준)도 지원:
  --dev_file + --nshot > 0

실행 예시:
  python eval_mmlu.py \
    --model /path/to/model \
    --data_file ./data/test/mmlu_test.jsonl \
    --dev_file  ./data/dev/mmlu_dev.jsonl \
    --nshot 5 \
    --tensor_parallel_size 4 \
    --batch_size 256 \
    --seq_len 8 \
    --outdir mmlu_5shot
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


MAX_INT = sys.maxsize
CHOICE_LETTERS = ("A", "B", "C", "D")


def seed_everything(seed: int = 1111) -> None:
    """Best-effort reproducibility for prompt sampling."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    try:
        from transformers import set_seed
        set_seed(seed)
    except Exception:
        pass


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """Read a JSONL file without requiring the `jsonlines` package."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def batch_data(items: List[str], batch_size: int) -> List[List[str]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)] if batch_size > 0 else [items]


def _strip_think(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1]
    return text


def extract_choice(text: str) -> Optional[str]:
    """Robustly extract A/B/C/D from a model completion."""
    if text is None:
        return None
    text = _strip_think(text).strip()

    # 1) "The answer is: B" / "Answer: B" 류
    m = re.search(r"(?i)(?:the\s+answer\s+is\s*[:\-]\s*)\(?([A-D])\)?", text)
    if m:
        return m.group(1).upper()
    m = re.search(r"(?i)answer\s*[:\-]\s*\(?([A-D])\)?", text)
    if m:
        return m.group(1).upper()

    # 2) 시작 토큰이 "B" / "B." / "(B)" 류
    m = re.match(r"^\s*\(?([A-D])\)?(?:\.|\)|:|\b)", text)
    if m:
        return m.group(1).upper()

    # 3) 본문 어딘가의 첫 standalone A-D
    m = re.search(r"\b([A-D])\b", text)
    if m:
        return m.group(1).upper()

    return None


def extract_gt_from_item(item: Dict[str, Any]) -> Optional[str]:
    """Extract ground-truth as A/B/C/D from different possible schemas."""
    if "answer" in item and item["answer"] is not None:
        ans = item["answer"]
        if isinstance(ans, int) and 0 <= ans < 4:
            return CHOICE_LETTERS[ans]
        if isinstance(ans, str):
            ans = ans.strip().upper()
            if ans in CHOICE_LETTERS:
                return ans

    # bench 스타일: response에 "#### B. ..."
    if "response" in item and item["response"] is not None:
        resp = str(item["response"])
        if "####" in resp:
            resp = resp.split("####", 1)[1]
        return extract_choice(resp)

    # 다른 키 이름들
    for k in ("label", "gold", "gt"):
        if k in item and isinstance(item[k], str) and item[k].strip().upper() in CHOICE_LETTERS:
            return item[k].strip().upper()
    return None


def build_query_text(item: Dict[str, Any], *, add_letter_only_instruction: bool = True) -> str:
    """Build the user-visible query text (without chat_template wrapping)."""
    if item.get("query"):
        q = str(item["query"]).rstrip()
        # prompt 끝을 Answer slot으로 맞춰주기
        if not re.search(r"(?i)\banswer\b\s*[:\-]?\s*$", q):
            q += "\n\nAnswer:"
        if add_letter_only_instruction:
            q += " (choose one of A, B, C, D)"
        return q

    # structured 형태
    subject = item.get("subject") or item.get("category") or ""
    question = str(item.get("question", "")).strip()
    choices = item.get("choices") or item.get("options")

    lines: List[str] = []
    if subject:
        lines.append(f"Subject: {subject}")
    lines.append(f"Question: {question}")

    if isinstance(choices, dict):
        for L in CHOICE_LETTERS:
            if L in choices:
                lines.append(f"{L}. {str(choices[L]).strip()}")
    else:
        if choices is None:
            choices = []
        for i, opt in enumerate(list(choices)[:4]):
            lines.append(f"{CHOICE_LETTERS[i]}. {str(opt).strip()}")

    if add_letter_only_instruction:
        lines.append("Answer: (choose one of A, B, C, D)")
    else:
        lines.append("Answer:")
    return "\n".join(lines)


def wrap_with_model_template(user_text: str, tokenizer) -> str:
    """Apply chat template if present; otherwise fall back to instruction wrapper."""
    has_chat = bool(getattr(tokenizer, "chat_template", None))
    if has_chat:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{user_text}\n\n### Response:\n"
    )


def format_qa_block(question_text: str, answer_letter: Optional[str]) -> str:
    """Few-shot block in classic MMLU format."""
    answer_letter = answer_letter or ""
    return f"{question_text}\n{answer_letter}\n\n"


def build_fewshot_prefix(
    subject: str,
    dev_examples: List[Tuple[str, str]],
    nshot: int,
    rng: random.Random,
) -> str:
    if nshot <= 0 or not dev_examples:
        return ""
    if len(dev_examples) <= nshot:
        picked = dev_examples
    else:
        picked = rng.sample(dev_examples, k=nshot)

    header = f"The following are multiple choice questions (with answers) about {subject}.\n\n" if subject else ""
    blocks = []
    for q_text, gt in picked:
        # q_text는 Answer slot으로 끝나있고, 다음 줄에 정답 letter를 붙여 few-shot 구성
        blocks.append(format_qa_block(q_text, gt))
    return header + "".join(blocks)


@dataclass
class Example:
    subject: str
    query_text: str        # 사람이 읽기 쉬운(=모델 템플릿 적용 전) query
    prompt_text: str       # 모델 템플릿까지 적용된 최종 prompt
    gt: Optional[str]


def load_examples(
    data_file: str,
    tokenizer,
    *,
    start: int = 0,
    end: int = MAX_INT,
    nshot: int = 0,
    dev_file: Optional[str] = None,
    seed: int = 1111,
) -> List[Example]:

    rng = random.Random(seed)

    # dev pool per subject
    dev_pool: Dict[str, List[Tuple[str, str]]] = {}
    if dev_file and nshot > 0:
        for item in iter_jsonl(dev_file):
            subj = str(item.get("subject") or item.get("category") or "")
            q_text = build_query_text(item, add_letter_only_instruction=False).rstrip()
            if not re.search(r"(?i)\banswer\b\s*[:\-]?\s*$", q_text):
                q_text += "\nAnswer:"
            gt = extract_gt_from_item(item)
            if gt is None:
                continue
            dev_pool.setdefault(subj, []).append((q_text, gt))

    examples: List[Example] = []
    for i, item in enumerate(iter_jsonl(data_file)):
        if i < start:
            continue
        if i >= end:
            break
        subj = str(item.get("subject") or item.get("category") or "")
        gt = extract_gt_from_item(item)

        test_q = build_query_text(item, add_letter_only_instruction=True).rstrip()

        prefix = build_fewshot_prefix(subj, dev_pool.get(subj, []), nshot, rng)
        full_user_text = prefix + test_q

        prompt = wrap_with_model_template(full_user_text, tokenizer)
        examples.append(Example(subject=subj, query_text=full_user_text, prompt_text=prompt, gt=gt))

    return examples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model name or local path")
    ap.add_argument("--data_file", required=True, help="JSONL test file")
    ap.add_argument("--dev_file", default=None, help="JSONL dev file (for few-shot); optional")
    ap.add_argument("--nshot", type=int, default=0, help="Number of few-shot examples per subject")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=MAX_INT)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--tensor_parallel_size", type=int, default=4)
    ap.add_argument("--max_model_len", type=int, default=4096)

    # 기존 eval 스크립트랑 맞추려고 `seq_len` 이름을 사용합니다.
    # (MMLU는 보통 A/B/C/D만 뽑으면 되므로 8~16 정도면 충분)
    ap.add_argument("--seq_len", type=int, default=16)
    # 호환용 alias
    ap.add_argument("--max_tokens", type=int, default=None)

    ap.add_argument("--seed", type=int, default=1111)
    ap.add_argument("--outdir", type=str, default="mmlu_results", help="output prefix; saves <outdir>.csv")
    ap.add_argument("--out_csv", type=str, default=None, help="(optional) explicit output csv path")
    args = ap.parse_args()

    # max_tokens 우선순위: --max_tokens가 주어지면 그걸 사용, 아니면 --seq_len 사용
    max_tokens = args.max_tokens if args.max_tokens is not None else args.seq_len

    seed_everything(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    examples = load_examples(
        args.data_file,
        tokenizer,
        start=args.start,
        end=args.end,
        nshot=args.nshot,
        dev_file=args.dev_file,
        seed=args.seed,
    )

    prompts = [ex.prompt_text for ex in examples]

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )

    stop_tokens = ["\n", "</s>", "<|endoftext|>", "<|end_of_text|>"]
    if getattr(tokenizer, "eos_token", None):
        stop_tokens.append(tokenizer.eos_token)

    sampling_params = SamplingParams(
        temperature=0,
        top_p=1.0,
        max_tokens=max_tokens,
        stop=stop_tokens,
        seed=args.seed,
    )

    completions: List[str] = []
    for batch in batch_data(prompts, args.batch_size):
        for out in llm.generate(batch, sampling_params):
            completions.append(out.outputs[0].text)

    # Length correction (일관성 유지용)
    if len(completions) != len(prompts):
        diff = len(prompts) - len(completions)
        print(f"[warn] completions {len(completions)} vs prompts {len(prompts)}", file=sys.stderr)
        if diff > 0:
            completions.extend(["[no_output]"] * diff)
        completions = completions[: len(prompts)]

    preds: List[str] = []
    correct: List[int] = []
    for ex, comp in zip(examples, completions):
        pred = extract_choice(comp)
        preds.append(pred if pred is not None else "[invalid]")
        correct.append(int(pred is not None and ex.gt is not None and pred == ex.gt))

    df = pd.DataFrame(
        {
            "subject": [ex.subject for ex in examples],
            "query": [ex.query_text for ex in examples],
            # 기존 파이프라인과 동일한 컬럼명
            "answer": completions,              # raw model output
            "GT": [ex.gt for ex in examples],   # ground truth letter
            # 추가로 유용한 컬럼
            "pred": preds,
            "correct": correct,
        }
    )

    out_path = args.out_csv or f"{args.outdir}.csv"
    df.to_csv(out_path, index=False)

    # Console summary
    overall = float(df["correct"].mean()) if len(df) else 0.0
    print("=" * 60)
    print(f"samples: {len(df)}")
    print(f"model: {args.model}")
    print(f"nshot: {args.nshot}")
    print(f"overall accuracy: {overall:.4%}")

    if (df["subject"].fillna("") != "").any():
        by_subj = df.groupby("subject")["correct"].mean().sort_values(ascending=False)
        print("\nper-subject accuracy (top 15):")
        print(by_subj.head(15).to_string())
        print("\nper-subject accuracy (bottom 15):")
        print(by_subj.tail(15).to_string())

    print(f"CSV saved -> {os.path.abspath(out_path)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
