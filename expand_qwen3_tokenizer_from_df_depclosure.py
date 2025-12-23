#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tokenizer 확장 (DF 코퍼스 기반) — 안전/효율 버전
=================================================
요구사항 반영:
- 너가 만든 df['text'] 코퍼스를 그대로 사용 (Goedel-Pset-v1 + Lean-workbook-proofs)
- train_new_from_iterator 로 "후보 토크나이저" 학습 (완전 재학습) -> candidate
- candidate로 코퍼스를 토크나이즈해서 token frequency를 구함
- 새 토큰은 "freq × savings"로 top-K만 선별 (효율)
- merges는 기존(base) merges를 절대 교체하지 않고 그대로 두고,
  선별된 새 토큰을 만들기 위해 필요한 merges만 "의존성 클로저"로 뽑아 뒤에 append (안정)
- base merges로 이미 생성 가능한(base_producible) 토큰을 재생성하는 merges는 자동으로 배제 (중복 제거)
  단, base vocab에 있지만 base merges로는 생성이 안 되는 "죽은 토큰"이 중간 단계로 필요하면 예외적으로 허용
- Lean 핵심 토큰(키워드/연산자)은 AddedToken으로 보장

실행:
  python expand_qwen3_tokenizer_from_df_depclosure.py

주의:
- 실행 후 모델을 쓰려면 반드시:
    model.resize_token_embeddings(len(tokenizer))
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
import unicodedata as ud
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from tqdm import tqdm

import datasets
import pandas as pd
from transformers import AutoTokenizer

try:
    from tokenizers import AddedToken  # tokenizers
except Exception:
    from transformers import AddedToken  # fallback (일부 환경)


# =========================
# Config
# =========================
BASE_MODEL = "Qwen/Qwen3-8B"

BASE_DIR = "./pangea_qwen3_tokenizer"   # old_tokenizer.save_pretrained 결과
NEW_DIR  = "./new_tokenizer_qwen3_new"  # candidate tokenizer save dir
OUT_DIR  = "./pangea_qwen3_tokenizer_safe"  # 최종 확장 토크나이저 출력 dir

# Candidate tokenizer training
CANDIDATE_VOCAB_SIZE = 30_000
BATCH_SIZE = 100_000  # train_new_from_iterator에서 yield batch 크기

# Efficient selection
MAX_TARGET_NEW_TOKENS = 5_000
MIN_NEW_TOKEN_FREQ = 5
MIN_SAVINGS = 1  # base tokenize 대비 최소 절감량

# Frequency computation sampling (너무 크면 조절)
FREQ_TEXT_LIMIT: Optional[int] = 200_000  # None이면 df 전체 사용

# Lean important tokens (guarantee)
FORCE_KEYWORDS  = ["theorem", "lemma", "simp", "by", "have", "intro", "rw", "cases", "fun", "match"]
FORCE_OPERATORS = ["∀", "∃", "↦", "⊢", ":=", "->", "=>"]
FORCE_TOKENS    = FORCE_KEYWORDS + FORCE_OPERATORS

# Garbage filter allowlist
OPERATOR_ALLOWLIST = set(FORCE_OPERATORS) | {
    "::", "≠", "≤", "≥", "↔", "∧", "∨", "¬", "⟨", "⟩", "⟹", "⊆", "⊂",
    "∘", "⋆", "×", "⋅", "·", "=", "<", ">", "+", "-", "*", "/", "→"
}

# Force tokens are better as AddedToken only (안정/명확)
FORCE_TOKENS_AS_ADDED_ONLY = True


# =========================
# Corpus build (your code)
# =========================
def build_df_corpus() -> pd.DataFrame:
    """
    너가 보여준 방식 그대로:
      data1= datasets.load_dataset('Goedel-LM/Goedel-Pset-v1')
      data2= datasets.load_dataset('Goedel-LM/Lean-workbook-proofs')
      formal_statement, informal_statement, full_proof 모아서 df['text'] 생성
    """
    data1 = datasets.load_dataset("Goedel-LM/Goedel-Pset-v1")
    data2 = datasets.load_dataset("Goedel-LM/Lean-workbook-proofs")

    text_lst: List[str] = []

    for ex in data1["train"]:
        fs = ex.get("formal_statement", None)
        if isinstance(fs, str) and fs.strip():
            text_lst.append(fs)
        ins = ex.get("informal_statement", None)
        if isinstance(ins, str) and ins.strip():
            text_lst.append(ins)

    for ex in data2["train"]:
        fp = ex.get("full_proof", None)
        if isinstance(fp, str) and fp.strip():
            text_lst.append(fp)

    df = pd.DataFrame({"text": text_lst})
    return df


# =========================
# Token / merge utilities
# =========================
def merge_parts(m) -> Tuple[str, str]:
    if isinstance(m, str):
        a, b = m.split()
        return a, b
    return str(m[0]), str(m[1])

def merge_to_str(m) -> str:
    a, b = merge_parts(m)
    return f"{a} {b}"

def result_token_of_merge(ms: str) -> str:
    a, b = ms.split()
    return a + b

def strip_markers(tok: str) -> str:
    return tok.lstrip("Ġ▁")

def token_surface(tok: str) -> str:
    # GPT-style byte BPE vocab often uses Ġ/▁ to mark "space + token"
    if tok.startswith("Ġ") or tok.startswith("▁"):
        return " " + tok[1:]
    return tok

def is_repeating_chunk(s: str) -> bool:
    n = len(s)
    if n < 4:
        return False
    for k in range(1, min(4, n // 2) + 1):
        if n % k == 0 and s == s[:k] * (n // k):
            return True
    return False

def is_weird_repeat(tok: str) -> bool:
    body = strip_markers(tok)
    if not body:
        return False
    if len(body) >= 3 and len(set(body)) == 1:
        return True
    if re.fullmatch(r"([\-_=])\1{2,}", body):
        return True
    if is_repeating_chunk(body):
        return True
    return False

def is_garbage(tok: str) -> bool:
    # Force tokens are never garbage
    if tok in FORCE_TOKENS or strip_markers(tok) in OPERATOR_ALLOWLIST:
        return False

    if "\ufffd" in tok or "\x00" in tok:
        return True
    if tok.endswith("\n"):
        return True

    body = strip_markers(tok)
    if body == "":
        return len(tok) > 1

    if len(body) >= 3 and len(set(body)) == 1:
        return True
    if re.fullmatch(r"([\-_=])\1{2,}", body):
        return True
    if is_repeating_chunk(body):
        return True

    # letters/numbers -> keep
    if any(ud.category(ch)[0] in ("L", "N") for ch in body):
        return False

    # single math symbol -> keep
    if len(body) == 1 and ud.category(body) == "Sm":
        return False

    if body in OPERATOR_ALLOWLIST:
        return False

    # pure punctuation/symbol soup -> garbage
    if all(ud.category(ch)[0] in ("P", "S") for ch in body):
        return True

    return False


def compute_producible_tokens(vocab_keys: Set[str], merges: Sequence) -> Set[str]:
    """
    base merges로 만들 수 있는 토큰 근사 집합을 계산.
    (BPE merges를 순서대로 적용하면서, 양쪽 파트가 이미 가능하면 결과도 가능하다고 보는 단순 클로저)
    """
    merges_str = [merge_to_str(m) for m in merges]
    results = set(result_token_of_merge(ms) for ms in merges_str)
    atoms = set(vocab_keys) - results
    producible = set(atoms)

    for ms in merges_str:
        a, b = ms.split()
        res = a + b
        if a in producible and b in producible:
            producible.add(res)
    return producible


# =========================
# Main pipeline
# =========================
def main() -> None:
    # 1) Build df corpus (your exact 방식)
    print("[corpus] building df from datasets ...")
    df = build_df_corpus()
    print("[corpus] df rows:", len(df))

    # Drop invalid
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    ALL_CORPUS = df["text"].to_list()

    def batch_iterator() -> Iterator[List[str]]:
        for i in range(0, len(ALL_CORPUS), BATCH_SIZE):
            yield ALL_CORPUS[i:i+BATCH_SIZE]

    # 2) Save base tokenizer (if not already)
    Path(BASE_DIR).mkdir(parents=True, exist_ok=True)
    if not Path(BASE_DIR, "tokenizer.json").exists():
        print("[base] saving base tokenizer to", BASE_DIR)
        AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True).save_pretrained(BASE_DIR)

    base_tok = AutoTokenizer.from_pretrained(BASE_DIR, use_fast=True)

    # 3) Train candidate tokenizer from your df corpus (or reuse if already exists)
    if Path(NEW_DIR, "tokenizer.json").exists():
        print("[cand] reuse existing candidate tokenizer:", NEW_DIR)
        cand_tok = AutoTokenizer.from_pretrained(NEW_DIR, use_fast=True)
    else:
        print("[cand] training candidate tokenizer (train_new_from_iterator) ...")
        cand_tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True).train_new_from_iterator(
            batch_iterator(), CANDIDATE_VOCAB_SIZE
        )
        cand_tok.save_pretrained(NEW_DIR)
        print("[cand] saved to", NEW_DIR)

    # 4) Load tokenizer.json (base + candidate)
    with open(Path(BASE_DIR) / "tokenizer.json", "r", encoding="utf-8") as f:
        base_info = json.load(f)
    with open(Path(NEW_DIR) / "tokenizer.json", "r", encoding="utf-8") as f:
        new_info = json.load(f)

    base_vocab: Dict[str, int] = base_info["model"]["vocab"]
    base_merges = base_info["model"]["merges"]
    base_vocab_keys = set(base_vocab.keys())

    new_vocab: Dict[str, int] = new_info["model"]["vocab"]
    new_merges = new_info["model"]["merges"]

    base_repeat_cnt = sum(is_weird_repeat(t) for t in base_vocab_keys)
    base_producible = compute_producible_tokens(base_vocab_keys, base_merges)

    print(f"[base] vocab={len(base_vocab)} merges={len(base_merges)} weird_repeat={base_repeat_cnt}")
    print(f"[base] producible≈{len(base_producible)}")

    # 5) Frequency stats on corpus with candidate tokenizer
    limit = len(ALL_CORPUS) if FREQ_TEXT_LIMIT is None else min(FREQ_TEXT_LIMIT, len(ALL_CORPUS))
    print(f"[freq] tokenizing corpus for freq... (N={limit})")
    freq_counter: Counter = Counter()

    for text in tqdm(ALL_CORPUS[:limit], desc="cand_tok.tokenize"):
        toks = cand_tok.tokenize(text)
        freq_counter.update(toks)

    print(f"[freq] unique_tokens_seen={len(freq_counter)}")

    # 6) Score tokens (freq × savings)
    candidates: List[Tuple[float, str, int, int]] = []  # (score, tok, freq, savings)
    skipped = Counter()

    for tok, freq in freq_counter.items():
        if freq < MIN_NEW_TOKEN_FREQ:
            skipped["lowfreq"] += 1
            continue
        if FORCE_TOKENS_AS_ADDED_ONLY and tok in FORCE_TOKENS:
            skipped["force"] += 1
            continue
        if tok in base_vocab_keys:
            skipped["in_base"] += 1
            continue
        if is_garbage(tok):
            skipped["garbage"] += 1
            continue

        surf = token_surface(tok)
        base_pieces = base_tok.tokenize(surf)
        savings = len(base_pieces) - 1
        if savings < MIN_SAVINGS:
            skipped["nosave"] += 1
            continue

        # 길이 페널티 (너무 긴 토큰 방지)
        score = float(freq * savings) / (1.0 + 0.02 * max(0, len(strip_markers(tok)) - 20))
        candidates.append((score, tok, freq, savings))

    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = candidates[:MAX_TARGET_NEW_TOKENS]
    selected_tokens = [t for _, t, _, _ in selected]
    selected_set = set(selected_tokens)

    print(f"[select] candidates={len(candidates)} selected={len(selected_tokens)} skipped={dict(skipped)}")

    # 7) Dependency-closure merges (candidate merges 기반)
    new_merges_str: List[str] = [merge_to_str(m) for m in new_merges]

    # map result_token -> first merge producing it
    merge_map: Dict[str, Tuple[str, str, str]] = {}
    for ms in new_merges_str:
        a, b = ms.split()
        res = a + b
        if res not in merge_map:
            merge_map[res] = (a, b, ms)

    needed_merges: Set[str] = set()
    needed_tokens: Set[str] = set()
    dead_tokens: Set[str] = set()
    allowed_dead_base_intermediates: Set[str] = set()
    pruned_by_base_producible = 0

    def ensure_token(t: str) -> None:
        nonlocal pruned_by_base_producible

        # 중복 제거 핵심: base merges로 이미 생성 가능한 토큰이면 여기서 컷
        if t in base_producible:
            pruned_by_base_producible += 1
            return

        if t in needed_tokens:
            return
        needed_tokens.add(t)

        if t in merge_map:
            a, b, ms = merge_map[t]
            ensure_token(a)
            ensure_token(b)
            needed_merges.add(ms)

            if t in base_vocab_keys and t not in base_producible:
                allowed_dead_base_intermediates.add(t)
        else:
            dead_tokens.add(t)

    for t in tqdm(selected_tokens, desc="[closure] ensure selected tokens"):
        ensure_token(t)

    if dead_tokens:
        # 생성 경로 없는 토큰은 의미 없으니 제거
        selected_set -= dead_tokens
        selected_tokens = [t for t in selected_tokens if t not in dead_tokens]
        print(f"[warn] dropped dead selected tokens (no generating merge): {len(dead_tokens)}")

    base_merges_set = set(merge_to_str(m) for m in base_merges)

    # keep merges in candidate order (stable)
    kept_new_merges_ordered = [ms for ms in new_merges_str if ms in needed_merges and ms not in base_merges_set]

    # final dedup
    seen = set()
    kept_new_merges: List[str] = []
    for ms in kept_new_merges_ordered:
        if ms in seen:
            continue
        seen.add(ms)
        kept_new_merges.append(ms)

    print(f"[closure] needed_merges={len(needed_merges)} kept_new_merges={len(kept_new_merges)}")
    print(f"[closure] pruned_by_base_producible_calls={pruned_by_base_producible}")
    print(f"[closure] allowed_dead_base_intermediates={len(allowed_dead_base_intermediates)}")

    # 8) Expand vocab with selected + tokens required by kept merges
    required_tokens: Set[str] = set()
    for ms in kept_new_merges:
        a, b = ms.split()
        required_tokens.update([a, b, a + b])

    required_tokens.update(selected_tokens)

    # base가 이미 만들 수 있는 건 추가 불필요
    required_tokens = {t for t in required_tokens if t not in base_producible}

    # Force tokens are AddedToken-only
    if FORCE_TOKENS_AS_ADDED_ONLY:
        required_tokens -= set(FORCE_TOKENS)

    expanded_vocab = dict(base_vocab)
    next_id = max(base_vocab.values()) + 1

    add_cnt = 0
    drop_garbage_cnt = 0
    dup_cnt = 0

    # Add "selected first" then intermediates
    def sort_key(t: str):
        return (t not in selected_set, -freq_counter.get(t, 0), len(t))

    for tok in sorted(required_tokens, key=sort_key):
        if tok in expanded_vocab:
            dup_cnt += 1
            continue
        if is_garbage(tok):
            drop_garbage_cnt += 1
            continue
        expanded_vocab[tok] = next_id
        next_id += 1
        add_cnt += 1

    added_repeat_cnt = sum(is_weird_repeat(t) for t in required_tokens)

    print(f"[vocab] required_tokens={len(required_tokens)} added={add_cnt} dup={dup_cnt} dropped_garbage={drop_garbage_cnt}")
    print(f"[vocab] weird_repeat_in_required≈{added_repeat_cnt}")

    # 9) Write tokenizer.json (base copy + append merges)
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    shutil.copytree(BASE_DIR, OUT_DIR)

    out_info = copy.deepcopy(base_info)
    out_info["model"]["vocab"] = expanded_vocab

    base_merges_as_str = True
    if len(base_merges) > 0 and not isinstance(base_merges[0], str):
        base_merges_as_str = False

    if base_merges_as_str:
        out_info["model"]["merges"] = list(base_merges) + kept_new_merges
    else:
        out_info["model"]["merges"] = list(base_merges) + [[a, b] for (a, b) in (ms.split() for ms in kept_new_merges)]

    with open(Path(OUT_DIR) / "tokenizer.json", "w", encoding="utf-8") as f:
        json.dump(out_info, f, ensure_ascii=False, indent=2)

    # 10) Add AddedTokens for Lean essentials
    out_tok = AutoTokenizer.from_pretrained(OUT_DIR, use_fast=True)

    added_objs: List[AddedToken] = []
    for kw in FORCE_KEYWORDS:
        # single_word=True: 식별자 내부 매칭 줄이기
        added_objs.append(AddedToken(kw, single_word=True, lstrip=False, rstrip=False, normalized=True))
    for op in FORCE_OPERATORS:
        added_objs.append(AddedToken(op, single_word=False, lstrip=False, rstrip=False, normalized=False))

    before_len = len(out_tok)
    forced_added = out_tok.add_tokens(added_objs, special_tokens=False)
    out_tok.save_pretrained(OUT_DIR)
    after_len = len(out_tok)

    # 11) Sanity checks
    verify = json.load(open(Path(OUT_DIR) / "tokenizer.json", "r", encoding="utf-8"))
    out_merges = verify["model"]["merges"]
    assert out_merges[:len(base_merges)] == base_merges, "ERROR: base merges prefix가 유지되지 않았습니다!"

    test = "theorem foo : ∀ x, x -> x := by intro x; simp"
    print("[test]", test)
    print("[test tokens]", out_tok.tokenize(test))

    # 12) Summary
    print("\n========== SUMMARY ==========")
    print("base_vocab_size:", len(base_vocab))
    print("base_merges_size:", len(base_merges))
    print("base_weird_repeat_tokens:", base_repeat_cnt)
    print("candidate_vocab_size:", len(new_vocab))
    print("selected_new_tokens:", len(selected_tokens))
    print("dead_selected_tokens_dropped:", len(dead_tokens))
    print("vocab_added_count:", add_cnt)
    print("appended_merges:", len(kept_new_merges))
    print("forced_tokens_requested:", len(added_objs))
    print("forced_tokens_actually_added:", forced_added)
    print("tokenizer_len_before_forced:", before_len)
    print("tokenizer_len_after_forced:", after_len)
    print("OUT_DIR:", OUT_DIR)
    print("[important] model.resize_token_embeddings(len(tokenizer)) 꼭 호출하세요!")
    print("=============================\n")


if __name__ == "__main__":
    main()
