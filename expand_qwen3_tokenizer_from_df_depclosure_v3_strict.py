#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tokenizer 확장 (DF 코퍼스 기반) — v3 STRICT (이상한 토큰 "추가 금지" 중심)
======================================================================
v1에서 에러가 난 근본 이유:
- merges가 어떤 토큰을 참조하면, 그 토큰은 vocab에 반드시 있어야 함.
- v1은 "garbage 토큰"을 vocab에서 drop했는데, merges 쪽엔 남아 있어서 로딩 단계에서 폭발.

v2는 "필요하면 vocab에 강제 추가"로 로딩을 보장했지만,
너 말대로 "너무 이상한 토큰은 추가하면 안 됨" 목표엔 맞지 않을 수 있음.

v3의 정책:
- '추가 금지(HARD BAN)' 토큰은 절대 vocab에 넣지 않음.
- 대신 그런 토큰을 필요로 하는 merges는 버리고,
  그 merges에 의존하는 '선택된 신규 토큰'도 같이 버림.
- 즉, "이상한 토큰 추가"가 아니라 "그 경로 자체를 포기"하는 방식.
  (안정/안전 최우선)

추가로:
- candidate merges에서 result-token마다 여러 merges가 있을 수 있으니,
  가능한 경우 "금지 토큰을 피하는 대체 merge 경로"를 찾아서 채택(backtracking)함.

실행:
  python expand_qwen3_tokenizer_from_df_depclosure_v3_strict.py

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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

from tqdm import tqdm

import datasets
import pandas as pd
from transformers import AutoTokenizer

try:
    from tokenizers import AddedToken
except Exception:
    from transformers import AddedToken


# =========================
# Config
# =========================
BASE_MODEL = "Qwen/Qwen3-8B"

BASE_DIR = "./pangea_qwen3_tokenizer"
NEW_DIR  = "./new_tokenizer_qwen3_new"
OUT_DIR  = "./pangea_qwen3_tokenizer_safe"

CANDIDATE_VOCAB_SIZE = 30_000
BATCH_SIZE = 100_000

MAX_TARGET_NEW_TOKENS = 5_000
MIN_NEW_TOKEN_FREQ = 5
MIN_SAVINGS = 1

FREQ_TEXT_LIMIT: Optional[int] = 200_000  # None이면 전체

# Lean important tokens (guarantee)
FORCE_KEYWORDS  = ["theorem", "lemma", "simp", "by", "have", "intro", "rw", "cases", "fun", "match"]
FORCE_OPERATORS = ["∀", "∃", "↦", "⊢", ":=", "->", "=>"]
FORCE_TOKENS    = FORCE_KEYWORDS + FORCE_OPERATORS

# Operators allowed even if "symbol-only"
OPERATOR_ALLOWLIST = set(FORCE_OPERATORS) | {
    "::", "≠", "≤", "≥", "↔", "∧", "∨", "¬", "⟨", "⟩", "⟹", "⊆", "⊂",
    "∘", "⋆", "×", "⋅", "·", "=", "<", ">", "+", "-", "*", "/", "→"
}

# 정책: TeX/backslash 토큰은 "이상한 토큰"으로 보고 아예 금지할지
DISALLOW_BACKSLASH_TOKENS = True

# 너무 긴 신규 토큰 금지(안전)
MAX_NEW_TOKEN_CHARS = 80  # strip_markers 후 길이 기준

# 신규 토큰에서 "문자/숫자"가 전혀 없으면 금지(연산자 allowlist 제외)
REQUIRE_ALNUM_FOR_NEW_TOKENS = True

# Lean 핵심 토큰은 AddedToken으로만 보장(=BPE vocab에 안 넣음)
FORCE_TOKENS_AS_ADDED_ONLY = True


# =========================
# Corpus build (your exact 방식)
# =========================
def build_df_corpus() -> pd.DataFrame:
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

def is_fatal(tok: str) -> bool:
    # 정말 추가하면 안 되는 수준
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
    return False

def has_alnum(body: str) -> bool:
    return any(ud.category(ch)[0] in ("L", "N") for ch in body)

def is_allowed_new_token(tok: str) -> bool:
    """
    v3 STRICT 핵심:
    - base vocab에 없는 "신규 토큰"을 추가할지 여부 판단.
    """
    if tok in FORCE_TOKENS:
        return True  # (실제로는 AddedToken-only로 처리할 수도 있음)

    if is_fatal(tok):
        return False

    body = strip_markers(tok)

    if len(body) == 0:
        return False

    if len(body) > MAX_NEW_TOKEN_CHARS:
        return False

    if DISALLOW_BACKSLASH_TOKENS and ("\\" in body):
        return False

    if body in OPERATOR_ALLOWLIST:
        return True

    if REQUIRE_ALNUM_FOR_NEW_TOKENS and (not has_alnum(body)):
        return False

    # 과도한 기호 비율도 컷 (문자/숫자 포함이어도 너무 기호 위주면 컷)
    # ex) "\mathrm{...}" 같은 걸 걸러내고 싶을 때 유효
    # (backslash는 위에서 이미 컷)
    non_alnum = sum(1 for ch in body if ud.category(ch)[0] not in ("L", "N"))
    if non_alnum / max(1, len(body)) > 0.7 and body not in OPERATOR_ALLOWLIST:
        return False

    return True


def compute_producible_tokens(vocab_keys: Set[str], merges: Sequence) -> Set[str]:
    merges_str = [merge_to_str(m) for m in merges]
    results = set()
    for ms in merges_str:
        a, b = ms.split()
        results.add(a + b)
    atoms = set(vocab_keys) - results
    producible = set(atoms)
    for ms in merges_str:
        a, b = ms.split()
        res = a + b
        if a in producible and b in producible:
            producible.add(res)
    return producible


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

    corpus = df["text"].to_list()

    def batch_iterator() -> Iterator[List[str]]:
        for i in range(0, len(corpus), BATCH_SIZE):
            yield corpus[i:i + BATCH_SIZE]

    # base tokenizer
    Path(BASE_DIR).mkdir(parents=True, exist_ok=True)
    if not Path(BASE_DIR, "tokenizer.json").exists():
        print("[base] saving base tokenizer to", BASE_DIR)
        AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True).save_pretrained(BASE_DIR)

    base_tok = AutoTokenizer.from_pretrained(BASE_DIR, use_fast=True)

    # candidate tokenizer
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

    # load tokenizer.json
    with open(Path(BASE_DIR) / "tokenizer.json", "r", encoding="utf-8") as f:
        base_info = json.load(f)
    with open(Path(NEW_DIR) / "tokenizer.json", "r", encoding="utf-8") as f:
        new_info = json.load(f)

    base_vocab: Dict[str, int] = base_info["model"]["vocab"]
    base_merges = base_info["model"]["merges"]
    base_vocab_keys = set(base_vocab.keys())

    new_merges = new_info["model"]["merges"]
    new_merges_str = [merge_to_str(m) for m in new_merges]

    base_producible = compute_producible_tokens(base_vocab_keys, base_merges)
    print(f"[base] vocab={len(base_vocab)} merges={len(base_merges)} producible≈{len(base_producible)}")

    # freq
    limit = len(corpus) if FREQ_TEXT_LIMIT is None else min(FREQ_TEXT_LIMIT, len(corpus))
    print(f"[freq] tokenizing corpus for freq... (N={limit})")
    freq_counter: Counter = Counter()
    for text in tqdm(corpus[:limit], desc="cand_tok.tokenize"):
        freq_counter.update(cand_tok.tokenize(text))
    print(f"[freq] unique_tokens_seen={len(freq_counter)}")

    # score tokens
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

        if not is_allowed_new_token(tok):
            skipped["disallowed_token"] += 1
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

    print(f"[select] candidates={len(candidates)} selected(pre)={len(selected_tokens)} skipped={dict(skipped)}")

    # --- Build merges_by_result (multiple options) ---
    merges_by_result: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    for ms in new_merges_str:
        a, b = ms.split()
        res = a + b
        merges_by_result[res].append((a, b, ms))

    # --- Constrained buildability with backtracking ---
    memo_can: Dict[str, bool] = {}
    choice: Dict[str, str] = {}  # token -> chosen merge string
    blocked_deps_counter: Counter = Counter()

    def can_build(t: str) -> bool:
        # if base can already produce it, treat as available
        if t in base_producible:
            return True

        if t in memo_can:
            return memo_can[t]

        # This is a *new* token candidate or intermediate:
        # It must be allowed to add.
        if not is_allowed_new_token(t):
            blocked_deps_counter[t] += 1
            memo_can[t] = False
            return False

        # must have a merge path in candidate merges
        options = merges_by_result.get(t, [])
        for a, b, ms in options:
            if can_build(a) and can_build(b):
                choice[t] = ms
                memo_can[t] = True
                return True

        memo_can[t] = False
        return False

    buildable_selected: List[str] = []
    dropped_selected = []
    for t in tqdm(selected_tokens, desc="[closure] check buildable"):
        if can_build(t):
            buildable_selected.append(t)
        else:
            dropped_selected.append(t)

    print(f"[closure] selected(buildable)={len(buildable_selected)} dropped_unbuildable={len(dropped_selected)}")

    if dropped_selected:
        # show a few examples
        print("[closure] dropped examples:", dropped_selected[:10])

    # --- Collect dependency-closure merges for buildable selected tokens ---
    needed_merges_set: Set[str] = set()

    def collect(t: str) -> None:
        if t in base_producible:
            return
        ms = choice.get(t)
        if ms is None:
            return
        if ms in needed_merges_set:
            return
        needed_merges_set.add(ms)
        a, b = ms.split()
        collect(a)
        collect(b)

    for t in tqdm(buildable_selected, desc="[closure] collect merges"):
        collect(t)

    base_merges_set = set(merge_to_str(m) for m in base_merges)

    # keep merges in candidate order, dedup, exclude those already in base
    kept_new_merges: List[str] = []
    seen = set()
    for ms in new_merges_str:
        if ms in needed_merges_set and ms not in base_merges_set and ms not in seen:
            kept_new_merges.append(ms)
            seen.add(ms)

    print(f"[closure] kept_new_merges={len(kept_new_merges)} (append)")

    # --- Build required tokens from kept merges ---
    required_tokens: Set[str] = set()
    for ms in kept_new_merges:
        a, b = ms.split()
        required_tokens.update([a, b, a + b])

    # Remove base-producible (already handled by base)
    required_tokens = {t for t in required_tokens if t not in base_producible}

    # Remove forced tokens if AddedToken-only
    if FORCE_TOKENS_AS_ADDED_ONLY:
        required_tokens -= set(FORCE_TOKENS)

    # HARD BAN enforcement: required tokens must be allowed_new_token
    banned_required = [t for t in required_tokens if not is_allowed_new_token(t)]
    if banned_required:
        # This should rarely happen because can_build already enforced,
        # but keep a guard anyway.
        print(f"[guard] banned tokens still in required_tokens: {len(banned_required)}")
        print("[guard] examples:", banned_required[:10])
        # Drop merges referencing banned_required (and recompute required_tokens)
        banned_set = set(banned_required)
        kept_before = len(kept_new_merges)
        kept_new_merges = [
            ms for ms in kept_new_merges
            if (ms.split()[0] not in banned_set and ms.split()[1] not in banned_set and (ms.split()[0] + ms.split()[1]) not in banned_set)
        ]
        print(f"[guard] dropped merges due to banned tokens: {kept_before - len(kept_new_merges)}")

        required_tokens.clear()
        for ms in kept_new_merges:
            a, b = ms.split()
            required_tokens.update([a, b, a + b])
        required_tokens = {t for t in required_tokens if t not in base_producible}
        if FORCE_TOKENS_AS_ADDED_ONLY:
            required_tokens -= set(FORCE_TOKENS)

    # Expand vocab with required tokens
    expanded_vocab = dict(base_vocab)
    next_id = max(base_vocab.values()) + 1

    add_cnt = 0
    for tok in sorted(required_tokens, key=lambda t: (-freq_counter.get(t, 0), len(t))):
        if tok in expanded_vocab:
            continue
        if not is_allowed_new_token(tok):
            # HARD BAN: 절대 추가 안 함
            continue
        expanded_vocab[tok] = next_id
        next_id += 1
        add_cnt += 1

    print(f"[vocab] required_tokens={len(required_tokens)} added={add_cnt}")

    # Write tokenizer.json (append merges)
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

    # Load and add AddedTokens
    out_tok = AutoTokenizer.from_pretrained(OUT_DIR, use_fast=True)

    added_objs: List[AddedToken] = []
    for kw in FORCE_KEYWORDS:
        added_objs.append(AddedToken(kw, single_word=True, lstrip=False, rstrip=False, normalized=True))
    for op in FORCE_OPERATORS:
        added_objs.append(AddedToken(op, single_word=False, lstrip=False, rstrip=False, normalized=False))

    before_len = len(out_tok)
    forced_added = out_tok.add_tokens(added_objs, special_tokens=False)
    out_tok.save_pretrained(OUT_DIR)
    after_len = len(out_tok)

    # Verify base merges prefix
    verify = json.load(open(Path(OUT_DIR) / "tokenizer.json", "r", encoding="utf-8"))
    out_merges = verify["model"]["merges"]
    assert out_merges[:len(base_merges)] == base_merges, "ERROR: base merges prefix가 유지되지 않았습니다!"

    # Quick test
    test = "theorem foo : ∀ x, x -> x := by intro x; simp"
    print("[test]", test)
    print("[test tokens]", out_tok.tokenize(test))

    # Report blocked dependency tokens (top)
    if blocked_deps_counter:
        top_blocked = blocked_deps_counter.most_common(20)
        print("\n[blocked deps] top 20 tokens that were DISALLOWED during closure:")
        for t, c in top_blocked:
            print(f"  {repr(t)}  (count={c})")

    print("\n========== SUMMARY ==========")
    print("base_vocab_size:", len(base_vocab))
    print("base_merges_size:", len(base_merges))
    print("selected_pre:", len(selected_tokens))
    print("selected_buildable:", len(buildable_selected))
    print("selected_dropped_unbuildable:", len(dropped_selected))
    print("vocab_added_count:", add_cnt)
    print("appended_merges:", len(kept_new_merges))
    print("forced_tokens_actually_added:", forced_added)
    print("tokenizer_len_before_forced:", before_len)
    print("tokenizer_len_after_forced:", after_len)
    print("OUT_DIR:", OUT_DIR)
    print("[important] model.resize_token_embeddings(len(tokenizer)) 꼭 호출하세요!")
    print("=============================\n")


if __name__ == "__main__":
    main()
