#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Prepare MMLU JSONL for this repo's pipeline.

Downloads from Hugging Face: `cais/mmlu`.

Outputs JSONL in the same schema used by eval_gsm8k.py etc:
  {"subject": "...", "query": "...", "response": "#### B"}

Also optionally emits a dev JSONL (5 examples/subject) for few-shot prompting.

Example:
  python prepare_mmlu_hf.py --out_test ./data/mmlu_test.jsonl --out_dev ./data/mmlu_dev.jsonl

Notes:
- `cais/mmlu` is multi-config: one config per subject.
- Splits include: auxiliary_train, dev, validation, test.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

from datasets import get_dataset_config_names, load_dataset

CHOICE_LETTERS = ("A", "B", "C", "D")


def build_query(question: str, choices: List[str], subject: str | None = None) -> str:
    lines: List[str] = []
    if subject:
        lines.append(f"Subject: {subject}")
    lines.append(f"Question: {question}")
    for i, opt in enumerate(choices[:4]):
        lines.append(f"{CHOICE_LETTERS[i]}. {opt}")
    lines.append("Answer: (choose one of A, B, C, D)")
    return "\n".join(lines)


def norm_answer(ans) -> str:
    # In HF `cais/mmlu`, answer is a ClassLabel; it often materializes as 'A'/'B'/'C'/'D'.
    if isinstance(ans, str) and ans.strip().upper() in CHOICE_LETTERS:
        return ans.strip().upper()
    # Sometimes it could be int index
    if isinstance(ans, int) and 0 <= ans < 4:
        return CHOICE_LETTERS[ans]
    return str(ans)


def write_jsonl(rows: List[Dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cais/mmlu")
    ap.add_argument("--out_test", required=True)
    ap.add_argument("--out_dev", default=None)
    ap.add_argument("--include_subject_in_query", action="store_true")
    args = ap.parse_args()

    subjects = get_dataset_config_names(args.dataset)

    test_rows: List[Dict] = []
    dev_rows: List[Dict] = []

    for subj in subjects:
        ds = load_dataset(args.dataset, subj)

        if "test" in ds:
            for ex in ds["test"]:
                q = build_query(ex["question"], ex["choices"], subj if args.include_subject_in_query else None)
                a = norm_answer(ex["answer"])
                test_rows.append({"subject": subj, "query": q, "response": f"#### {a}"})

        if args.out_dev and "dev" in ds:
            for ex in ds["dev"]:
                q = build_query(ex["question"], ex["choices"], subj if args.include_subject_in_query else None)
                a = norm_answer(ex["answer"])
                dev_rows.append({"subject": subj, "query": q, "response": f"#### {a}"})

    write_jsonl(test_rows, args.out_test)
    print(f"Wrote test JSONL: {args.out_test} ({len(test_rows)} rows)")

    if args.out_dev:
        write_jsonl(dev_rows, args.out_dev)
        print(f"Wrote dev JSONL:  {args.out_dev} ({len(dev_rows)} rows)")


if __name__ == "__main__":
    main()
