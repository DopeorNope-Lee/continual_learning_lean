#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tokenizer 확장 (DF 코퍼스 기반) — 안전/효율 "최종" 버전
======================================================
요구사항 반영:
- df['text'] 코퍼스를 그대로 사용 (Goedel-Pset-v1 + Lean-workbook-proofs)
- train_new_from_iterator 로 "후보 토크나이저" 학습 (완전 재학습) -> candidate
- candidate로 코퍼스를 토크나이즈해서 token frequency를 구함
- 새 토큰은 "freq × savings"로 top-K만 선별 (효율)
- merges는 기존(base) merges를 절대 교체하지 않고 그대로 두고,
  선별된 새 토큰을 만들기 위해 필요한 merges만 "의존성 클로저"로 뽑아 뒤에 append (안정)
- base merges로 이미 생성 가능한(base_producible) 토큰을 재생성하는 merges는 자동으로 배제 (중복 제거)
  단, base vocab에 있지만 base merges로는 생성이 안 되는 "죽은 토큰"이 중간 단계로 필요하면 예외적으로 허용
- Lean 핵심 토큰(키워드/연산자)은 "무조건 AddedToken"이 아니라:
    * base tokenizer가 이미 1토큰으로 잘 처리하면 AddedToken을 추가하지 않음
    * base가 분해하는 경우에만 AddedToken을 추가(선별)하여 토큰 길이 증가 부작용을 방지

실행:
  python expand_qwen3_tokenizer_from_df_depclosure_final.py

주의:
- 실행 후 모델을 쓰려면 반드시:
    model.resize_token_embeddings(len(tokenizer))
"""

from __future__ import annotations

import copy
import json
import re
import shutil
import unicodedata as ud
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

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

BASE_DIR = "./qwen3_tokenizer"   # old_tokenizer.save_pretrained 결과
NEW_DIR  = "./new_tokenizer_qwen3_new"  # candidate tokenizer save dir
OUT_DIR  = "./expanded_qwen3_tokenizer_safe"  # 최종 확장 토크나이저 출력 dir

# Candidate tokenizer training
CANDIDATE_VOCAB_SIZE = 50_000
BATCH_SIZE = 100_000  # train_new_from_iterator에서 yield batch 크기

# Efficient selection
MAX_TARGET_NEW_TOKENS = 10_000
MIN_NEW_TOKEN_FREQ = 5
MIN_SAVINGS = 1  # base tokenize 대비 최소 절감량

# Frequency computation sampling (너무 크면 조절)
FREQ_TEXT_LIMIT: Optional[int] = 200_000  # None이면 df 전체 사용

# Lean important tokens (candidate에서 선택하지 않고, 필요 시 AddedToken으로만 보강)
FORCE_KEYWORDS  = ["theorem", "lemma", "simp", "by", "have", "intro", "rw", "cases", "fun", "match"]
FORCE_OPERATORS = ["∀", "∃", "↦", "⊢", ":=", "->", "=>"]
FORCE_TOKENS    = FORCE_KEYWORDS + FORCE_OPERATORS

# Garbage filter allowlist
OPERATOR_ALLOWLIST = set(FORCE_OPERATORS) | {
    "::", "≠", "≤", "≥", "↔", "∧", "∨", "¬", "⟨", "⟩", "⟹", "⊆", "⊂",
    "∘", "⋆", "×", "⋅", "·", "=", "<", ">", "+", "-", "*", "/", "→"
}

# Force tokens: AddedToken-only 모드 (단, 선별적으로만 AddedToken 추가)
FORCE_TOKENS_AS_ADDED_ONLY = True


# =========================
# Corpus build (your code)
# =========================
def build_df_corpus() -> pd.DataFrame:
    """
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

    return pd.DataFrame({"text": text_lst})


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

    if any(ud.category(ch)[0] in ("L", "N") for ch in body):
        return False

    if len(body) == 1 and ud.category(body) == "Sm":
        return False

    if body in OPERATOR_ALLOWLIST:
        return False

    if all(ud.category(ch)[0] in ("P", "S") for ch in body):
        return True

    return False


def compute_producible_tokens(vocab_keys: Set[str], merges: Sequence) -> Set[str]:
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


def n_ids(tok, s: str) -> int:
    return len(tok(s, add_special_tokens=False)["input_ids"])

def need_added_token(base_tok, t: str) -> bool:
    if n_ids(base_tok, t) == 1:
        return False
    if n_ids(base_tok, " " + t) == 1:
        return False
    return True


# =========================
# Main
# =========================
def main() -> None:
    print("[corpus] building df from datasets ...")
    df = build_df_corpus()
    print("[corpus] df rows:", len(df))

    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    ALL_CORPUS = df["text"].to_list()

    def batch_iterator() -> Iterator[List[str]]:
        for i in range(0, len(ALL_CORPUS), BATCH_SIZE):
            yield ALL_CORPUS[i:i + BATCH_SIZE]

    Path(BASE_DIR).mkdir(parents=True, exist_ok=True)
    if not Path(BASE_DIR, "tokenizer.json").exists():
        print("[base] saving base tokenizer to", BASE_DIR)
        AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True).save_pretrained(BASE_DIR)

    base_tok = AutoTokenizer.from_pretrained(BASE_DIR, use_fast=True)

    FORCE_AS_ADDED = [t for t in FORCE_TOKENS if need_added_token(base_tok, t)]
    FORCE_AS_ADDED_SET = set(FORCE_AS_ADDED)
    print("[force] FORCE_TOKENS:", FORCE_TOKENS)
    print("[force] will add as AddedToken (base splits):", FORCE_AS_ADDED)

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

    limit = len(ALL_CORPUS) if FREQ_TEXT_LIMIT is None else min(FREQ_TEXT_LIMIT, len(ALL_CORPUS))
    print(f"[freq] tokenizing corpus for freq... (N={limit})")
    freq_counter: Counter = Counter()
    for text in tqdm(ALL_CORPUS[:limit], desc="cand_tok.tokenize"):
        freq_counter.update(cand_tok.tokenize(text))
    print(f"[freq] unique_tokens_seen={len(freq_counter)}")

    candidates: List[Tuple[float, str, int, int]] = []
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

        score = float(freq * savings) / (1.0 + 0.02 * max(0, len(strip_markers(tok)) - 20))
        candidates.append((score, tok, freq, savings))

    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = candidates[:MAX_TARGET_NEW_TOKENS]
    selected_tokens = [t for _, t, _, _ in selected]
    selected_set = set(selected_tokens)

    print(f"[select] candidates={len(candidates)} selected={len(selected_tokens)} skipped={dict(skipped)}")

    new_merges_str: List[str] = [merge_to_str(m) for m in new_merges]
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
        selected_set -= dead_tokens
        selected_tokens = [t for t in selected_tokens if t not in dead_tokens]
        print(f"[warn] dropped dead selected tokens (no generating merge): {len(dead_tokens)}")

    base_merges_set = set(merge_to_str(m) for m in base_merges)
    kept_new_merges_ordered = [ms for ms in new_merges_str if ms in needed_merges and ms not in base_merges_set]

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

    required_tokens: Set[str] = set()
    for ms in kept_new_merges:
        a, b = ms.split()
        required_tokens.update([a, b, a + b])
    required_tokens.update(selected_tokens)

    required_tokens = {t for t in required_tokens if t not in base_producible}

    if FORCE_TOKENS_AS_ADDED_ONLY:
        required_tokens -= FORCE_AS_ADDED_SET

    expanded_vocab = dict(base_vocab)
    next_id = max(base_vocab.values()) + 1

    add_cnt = 0
    drop_garbage_cnt = 0
    dup_cnt = 0

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

    print(f"[vocab] required_tokens={len(required_tokens)} added={add_cnt} dup={dup_cnt} dropped_garbage={drop_garbage_cnt}")

    # merges 참조 토큰이 vocab에 없으면 로딩 에러가 나므로, 그런 merges는 드롭
    vocab_keys_final = set(expanded_vocab.keys())
    kept_ok = []
    dropped_merges_due_to_vocab = 0
    for ms in kept_new_merges:
        a, b = ms.split()
        res = a + b
        if a in vocab_keys_final and b in vocab_keys_final and res in vocab_keys_final:
            kept_ok.append(ms)
        else:
            dropped_merges_due_to_vocab += 1
    if dropped_merges_due_to_vocab > 0:
        print(f"[guard] dropped merges due to missing vocab tokens: {dropped_merges_due_to_vocab}")
    kept_new_merges = kept_ok

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

    out_tok = AutoTokenizer.from_pretrained(OUT_DIR, use_fast=True)

    added_objs: List[AddedToken] = []
    for kw in FORCE_KEYWORDS:
        if kw not in FORCE_AS_ADDED_SET:
            continue
        added_objs.append(AddedToken(kw, single_word=True, lstrip=False, rstrip=False, normalized=True))
    for op in FORCE_OPERATORS:
        if op not in FORCE_AS_ADDED_SET:
            continue
        added_objs.append(AddedToken(op, single_word=False, lstrip=False, rstrip=False, normalized=False))

    before_len = len(out_tok)
    forced_added = out_tok.add_tokens(added_objs, special_tokens=False)
    out_tok.save_pretrained(OUT_DIR)
    after_len = len(out_tok)

    verify = json.load(open(Path(OUT_DIR) / "tokenizer.json", "r", encoding="utf-8"))
    out_merges = verify["model"]["merges"]
    assert out_merges[:len(base_merges)] == base_merges, "ERROR: base merges prefix가 유지되지 않았습니다!"

    test = "theorem foo : ∀ x, x -> x := by intro x; simp"
    print("[test]", test)
    print("[test tokens]", out_tok.tokenize(test))

    print("\n========== SUMMARY ==========")
    print("base_vocab_size:", len(base_vocab))
    print("base_merges_size:", len(base_merges))
    print("base_weird_repeat_tokens:", base_repeat_cnt)
    print("candidate_vocab_size:", len(new_vocab))
    print("selected_new_tokens:", len(selected_tokens))
    print("dead_selected_tokens_dropped:", len(dead_tokens))
    print("vocab_added_count:", add_cnt)
    print("appended_merges:", len(kept_new_merges))
    print("forced_tokens_total:", len(FORCE_TOKENS))
    print("forced_tokens_selected_for_added:", len(FORCE_AS_ADDED_SET))
    print("forced_tokens_requested:", len(added_objs))
    print("forced_tokens_actually_added:", forced_added)
    print("tokenizer_len_before_forced:", before_len)
    print("tokenizer_len_after_forced:", after_len)
    print("OUT_DIR:", OUT_DIR)
    print("[important] model.resize_token_embeddings(len(tokenizer)) 꼭 호출하세요!")
    print("=============================\n")


if __name__ == "__main__":
    main()
