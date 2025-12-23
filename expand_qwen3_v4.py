#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tokenizer 확장 (DF 코퍼스 기반) — v4 STRICT + REPORT
====================================================
요구사항 추가:
1) "드랍된 토큰 중 점수 상위 TOP-N" 리포트 자동 생성
2) "차단된 deps 때문에 드랍된 토큰이 어떤 유형인지" (원인 유형/예시) 리포트 자동 생성

핵심 정책(STRICT):
- 이상한 토큰(예: backslash 포함, alnum 없음 등)은 절대 vocab에 추가하지 않음.
- 따라서 그런 토큰을 필요로 하는 merge 경로는 버리고,
  그 merge 경로에 의존하는 신규 토큰도 같이 drop.
- 가능한 경우, 같은 결과 토큰을 만드는 "대체 merge 경로"가 있으면 자동으로 선택(backtracking).

출력:
- OUT_DIR/_reports/ 아래에 CSV/JSON 리포트 생성
  - dropped_selected_topN.csv
  - dropped_selected_reason_summary.csv
  - blocked_dependency_reason_summary.csv
  - report_summary.json

실행:
  python expand_qwen3_tokenizer_from_df_depclosure_v4_strict_report.py

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
from functools import lru_cache
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

# Report
REPORT_TOP_N_DROPPED = 80          # "드랍된 토큰 중 점수 상위 TOP-N"
REPORT_EXPLAIN_TOP_N = 80          # 설명(블로커 추적)할 드랍 토큰 수(상위 N개)
REPORT_MAX_BLOCKERS_PER_TOKEN = 10 # 한 토큰에 대해 보여줄 blocker 토큰 최대 개수
REPORT_MAX_EXAMPLES_PER_REASON = 12

# Lean important tokens (guarantee)
FORCE_KEYWORDS  = ["theorem", "lemma", "simp", "by", "have", "intro", "rw", "cases", "fun", "match"]
FORCE_OPERATORS = ["∀", "∃", "↦", "⊢", ":=", "->", "=>"]
FORCE_TOKENS    = FORCE_KEYWORDS + FORCE_OPERATORS

# Operators allowed even if "symbol-only"
OPERATOR_ALLOWLIST = set(FORCE_OPERATORS) | {
    "::", "≠", "≤", "≥", "↔", "∧", "∨", "¬", "⟨", "⟩", "⟹", "⊆", "⊂",
    "∘", "⋆", "×", "⋅", "·", "=", "<", ">", "+", "-", "*", "/", "→"
}

# STRICT token policy
DISALLOW_BACKSLASH_TOKENS = True          # \ 포함 토큰 금지 (LaTeX 흔적)
MAX_NEW_TOKEN_CHARS = 80                  # strip_markers 길이 기준
REQUIRE_ALNUM_FOR_NEW_TOKENS = True       # 연산자 allowlist 제외하고 L/N 없으면 금지
MAX_SYMBOL_RATIO = 0.70                   # non-alnum 비율이 너무 높으면 금지(연산자 제외)

# Force tokens: selection에서는 제외(불필요한 중복 확장 방지) + AddedToken으로 보장
SKIP_FORCE_TOKENS_IN_SELECTION = True


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

def why_disallowed(tok: str) -> Optional[str]:
    """
    STRICT 정책 기준으로 이 토큰이 "신규 토큰으로서 추가 금지"인 이유를 문자열로 반환.
    허용이면 None.
    """
    if tok in FORCE_TOKENS:
        return None  # force는 허용(실제로는 AddedToken으로 보장)

    if is_fatal(tok):
        return "fatal_invalid_or_repeat"

    body = strip_markers(tok)
    if len(body) == 0:
        return "empty_after_strip"

    if len(body) > MAX_NEW_TOKEN_CHARS:
        return "too_long"

    if body in OPERATOR_ALLOWLIST:
        return None

    if DISALLOW_BACKSLASH_TOKENS and ("\\" in body):
        return "contains_backslash"

    if REQUIRE_ALNUM_FOR_NEW_TOKENS and (not has_alnum(body)):
        return "no_alnum"

    non_alnum = sum(1 for ch in body if ud.category(ch)[0] not in ("L", "N"))
    if (non_alnum / max(1, len(body))) > MAX_SYMBOL_RATIO and body not in OPERATOR_ALLOWLIST:
        return "symbol_ratio_high"

    return None

def is_allowed_new_token(tok: str) -> bool:
    return why_disallowed(tok) is None


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
    cand_info: Dict[str, Dict[str, float]] = {}
    skipped = Counter()
    skipped_disallow_reasons = Counter()

    for tok, freq in freq_counter.items():
        if freq < MIN_NEW_TOKEN_FREQ:
            skipped["lowfreq"] += 1
            continue

        if SKIP_FORCE_TOKENS_IN_SELECTION and tok in FORCE_TOKENS:
            skipped["force"] += 1
            continue

        if tok in base_vocab_keys:
            skipped["in_base"] += 1
            continue

        dis_r = why_disallowed(tok)
        if dis_r is not None:
            skipped["disallowed_token"] += 1
            skipped_disallow_reasons[dis_r] += 1
            continue

        surf = token_surface(tok)
        base_pieces = base_tok.tokenize(surf)
        savings = len(base_pieces) - 1
        if savings < MIN_SAVINGS:
            skipped["nosave"] += 1
            continue

        score = float(freq * savings) / (1.0 + 0.02 * max(0, len(strip_markers(tok)) - 20))
        candidates.append((score, tok, freq, savings))
        cand_info[tok] = {"score": score, "freq": float(freq), "savings": float(savings), "len": float(len(strip_markers(tok)))}

    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = candidates[:MAX_TARGET_NEW_TOKENS]
    selected_tokens = [t for _, t, _, _ in selected]

    print(f"[select] candidates={len(candidates)} selected(pre)={len(selected_tokens)} skipped={dict(skipped)}")
    if skipped_disallow_reasons:
        print("[select] disallowed reasons (top):", skipped_disallow_reasons.most_common(6))

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
        if t in base_producible:
            return True
        if t in memo_can:
            return memo_can[t]

        dis_r = why_disallowed(t)
        if dis_r is not None:
            blocked_deps_counter[t] += 1
            memo_can[t] = False
            return False

        options = merges_by_result.get(t, [])
        # no merge path => cannot "create" this token via appended merges
        if not options:
            memo_can[t] = False
            return False

        for a, b, ms in options:
            if can_build(a) and can_build(b):
                choice[t] = ms
                memo_can[t] = True
                return True

        memo_can[t] = False
        return False

    buildable_selected: List[str] = []
    dropped_selected: List[str] = []
    for t in tqdm(selected_tokens, desc="[closure] check buildable"):
        if can_build(t):
            buildable_selected.append(t)
        else:
            dropped_selected.append(t)

    print(f"[closure] selected(buildable)={len(buildable_selected)} dropped_unbuildable={len(dropped_selected)}")

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

    # Remove base-producible (already in base vocab)
    required_tokens = {t for t in required_tokens if t not in base_producible}

    # HARD BAN guard: required tokens must be allowed
    banned_required = [t for t in required_tokens if not is_allowed_new_token(t)]
    if banned_required:
        banned_set = set(banned_required)
        kept_before = len(kept_new_merges)
        kept_new_merges = [
            ms for ms in kept_new_merges
            if (ms.split()[0] not in banned_set and ms.split()[1] not in banned_set and (ms.split()[0] + ms.split()[1]) not in banned_set)
        ]
        print(f"[guard] banned required tokens found: {len(banned_required)} -> dropped merges: {kept_before - len(kept_new_merges)}")

        required_tokens.clear()
        for ms in kept_new_merges:
            a, b = ms.split()
            required_tokens.update([a, b, a + b])
        required_tokens = {t for t in required_tokens if t not in base_producible}
        required_tokens = {t for t in required_tokens if is_allowed_new_token(t)}

    # Expand vocab with required tokens
    expanded_vocab = dict(base_vocab)
    next_id = max(base_vocab.values()) + 1

    add_cnt = 0
    for tok in sorted(required_tokens, key=lambda t: (-freq_counter.get(t, 0), len(t))):
        if tok in expanded_vocab:
            continue
        if not is_allowed_new_token(tok):
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

    # =========================
    # REPORT
    # =========================
    report_dir = Path(OUT_DIR) / "_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) dropped tokens TOP-N by score ----
    dropped_rows = []
    for t in dropped_selected:
        info = cand_info.get(t, None)
        if info is None:
            # should not happen, but keep robust
            info = {"score": float("nan"), "freq": float(freq_counter.get(t, 0)), "savings": float("nan"), "len": float(len(strip_markers(t)))}
        dropped_rows.append({
            "token": t,
            "score": info["score"],
            "freq": info["freq"],
            "savings": info["savings"],
            "len": info["len"],
        })

    dropped_df = pd.DataFrame(dropped_rows)
    dropped_df = dropped_df.sort_values("score", ascending=False)

    topN = dropped_df.head(REPORT_TOP_N_DROPPED).copy()
    topN.to_csv(report_dir / "dropped_selected_topN.csv", index=False)

    # ---- 2) explain: why dropped? (blocked deps 유형) ----
    # For explanation, we compute minimal blockers for top-N dropped (score 기준)
    top_to_explain = topN.head(REPORT_EXPLAIN_TOP_N)["token"].tolist()

    @lru_cache(maxsize=None)
    def explain_failure(token: str) -> Tuple[bool, Tuple[str, ...], Tuple[str, ...]]:
        """
        Returns (success, blockers, blocker_reasons).
        - success True means buildable (shouldn't happen for dropped tokens, but cache re-use)
        - blockers: tuple of "disallowed" tokens that block building (unique, order-stable)
        - blocker_reasons: tuple of reasons aligned with blockers (same length)
        Special case:
        - if no merge options: blockers empty, blocker_reasons contains ('no_merge_path',)
        """
        if token in base_producible:
            return True, tuple(), tuple()

        r = why_disallowed(token)
        if r is not None:
            return False, (token,), (r,)

        options = merges_by_result.get(token, [])
        if not options:
            return False, tuple(), ("no_merge_path",)

        best: Optional[Tuple[bool, Tuple[str, ...], Tuple[str, ...]]] = None

        for a, b, ms in options:
            sa, ba, ra = explain_failure(a)
            sb, bb, rb = explain_failure(b)
            if sa and sb:
                return True, tuple(), tuple()

            blockers: List[str] = []
            reasons: List[str] = []

            if not sa:
                # (ba,ra) or special
                if len(ba) == 0 and len(ra) == 1 and ra[0] == "no_merge_path":
                    blockers.append(a)
                    reasons.append("no_merge_path")
                else:
                    blockers.extend(list(ba))
                    reasons.extend(list(ra))

            if not sb:
                if len(bb) == 0 and len(rb) == 1 and rb[0] == "no_merge_path":
                    blockers.append(b)
                    reasons.append("no_merge_path")
                else:
                    blockers.extend(list(bb))
                    reasons.extend(list(rb))

            # dedup blockers while keeping first reason
            dedup_blockers = []
            dedup_reasons = []
            seen_local = set()
            for t, rr in zip(blockers, reasons):
                if t in seen_local:
                    continue
                seen_local.add(t)
                dedup_blockers.append(t)
                dedup_reasons.append(rr)

            # choose best (fewest blockers), tie-breaker: fewer "no_merge_path"
            def score_tuple(bl: List[str], rs: List[str]) -> Tuple[int, int]:
                return (len(bl), sum(1 for x in rs if x == "no_merge_path"))

            cand = (False, tuple(dedup_blockers), tuple(dedup_reasons))
            if best is None:
                best = cand
            else:
                if score_tuple(list(cand[1]), list(cand[2])) < score_tuple(list(best[1]), list(best[2])):
                    best = cand

        assert best is not None
        return best

    explained_rows = []
    drop_reason_counter = Counter()
    drop_reason_examples: Dict[str, List[str]] = defaultdict(list)

    for t in tqdm(top_to_explain, desc="[report] explain dropped tokens"):
        ok, blockers, reasons = explain_failure(t)
        if ok:
            # shouldn't happen, but keep
            reason_key = "unexpected_buildable"
            blockers_out = ""
            reasons_out = ""
        else:
            # main reason classification: most common blocker reason, or no_merge_path
            if len(blockers) == 0 and len(reasons) == 1 and reasons[0] == "no_merge_path":
                reason_key = "no_merge_path"
                blockers_out = ""
                reasons_out = "no_merge_path"
            else:
                # pick first reason as primary
                reason_key = reasons[0] if len(reasons) > 0 else "blocked_deps_unknown"
                # truncate
                blockers_list = list(blockers)[:REPORT_MAX_BLOCKERS_PER_TOKEN]
                reasons_list = list(reasons)[:REPORT_MAX_BLOCKERS_PER_TOKEN]
                blockers_out = " | ".join([repr(x) for x in blockers_list])
                reasons_out = " | ".join(reasons_list)

        drop_reason_counter[reason_key] += 1
        if len(drop_reason_examples[reason_key]) < REPORT_MAX_EXAMPLES_PER_REASON:
            drop_reason_examples[reason_key].append(t)

        info = cand_info.get(t, {"score": float("nan"), "freq": float(freq_counter.get(t, 0)), "savings": float("nan"), "len": float(len(strip_markers(t)))})
        explained_rows.append({
            "token": t,
            "score": info["score"],
            "freq": info["freq"],
            "savings": info["savings"],
            "len": info["len"],
            "primary_reason": reason_key,
            "blockers": blockers_out,
            "blocker_reasons": reasons_out,
        })

    explained_df = pd.DataFrame(explained_rows).sort_values("score", ascending=False)
    explained_df.to_csv(report_dir / "dropped_selected_topN_with_blockers.csv", index=False)

    # Reason summary for dropped selected
    reason_summary = pd.DataFrame([
        {"reason": k, "count": v, "examples": " | ".join([repr(x) for x in drop_reason_examples.get(k, [])[:8]])}
        for k, v in drop_reason_counter.most_common()
    ])
    reason_summary.to_csv(report_dir / "dropped_selected_reason_summary.csv", index=False)

    # ---- 3) blocked deps overall summary ----
    blocked_reason_counter = Counter()
    blocked_reason_examples: Dict[str, List[str]] = defaultdict(list)

    for tok, cnt in blocked_deps_counter.items():
        r = why_disallowed(tok) or "unknown"
        blocked_reason_counter[r] += cnt
        if len(blocked_reason_examples[r]) < REPORT_MAX_EXAMPLES_PER_REASON:
            blocked_reason_examples[r].append(tok)

    blocked_summary = pd.DataFrame([
        {"reason": r, "blocked_calls": c, "examples": " | ".join([repr(x) for x in blocked_reason_examples.get(r, [])[:8]])}
        for r, c in blocked_reason_counter.most_common()
    ])
    blocked_summary.to_csv(report_dir / "blocked_dependency_reason_summary.csv", index=False)

    # Summary JSON
    summary_obj = {
        "base_vocab_size": len(base_vocab),
        "base_merges_size": len(base_merges),
        "freq_text_limit": limit,
        "unique_tokens_seen": len(freq_counter),
        "candidates_scored": len(candidates),
        "selected_pre": len(selected_tokens),
        "selected_buildable": len(buildable_selected),
        "selected_dropped": len(dropped_selected),
        "vocab_added_count": add_cnt,
        "appended_merges": len(kept_new_merges),
        "forced_tokens_requested": len(added_objs),
        "forced_tokens_actually_added": forced_added,
        "tokenizer_len_before_forced": before_len,
        "tokenizer_len_after_forced": after_len,
        "report_dir": str(report_dir),
        "top_disallowed_reasons_in_scoring": skipped_disallow_reasons.most_common(20),
        "dropped_primary_reasons_top": drop_reason_counter.most_common(20),
        "blocked_dependency_reasons_top": blocked_reason_counter.most_common(20),
    }
    with open(report_dir / "report_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_obj, f, ensure_ascii=False, indent=2)

    # Console highlights
    print("\n========== REPORT HIGHLIGHTS ==========")
    print("[report] dir:", report_dir)
    print("[report] dropped_selected:", len(dropped_selected), " (top file: dropped_selected_topN.csv)")
    if drop_reason_counter:
        print("[report] dropped primary reasons top:", drop_reason_counter.most_common(6))
    if blocked_reason_counter:
        print("[report] blocked deps reasons top:", blocked_reason_counter.most_common(6))
    print("=======================================\n")

    print("\n========== SUMMARY ==========")
    print("base_vocab_size:", len(base_vocab))
    print("base_merges_size:", len(base_merges))
    print("selected_pre:", len(selected_tokens))
    print("selected_buildable:", len(buildable_selected))
    print("selected_dropped:", len(dropped_selected))
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
