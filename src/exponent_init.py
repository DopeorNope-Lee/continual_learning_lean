#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BUILD + VERIFY in one shot:
- base model: Qwen/Qwen3-8B (weights)
- expanded tokenizer: e.g., DopeorNope/FFT-expanded-naive (merge-based expanded tokenizer)
- output: aligned model+tokenizer directory

Build phase:
  1) resize embeddings to len(expanded tokenizer)
  2) for shared tokens (same token string): copy base row -> ext row (fixes shifted special tokens)
  3) for new tokens: AdaptiVocab exponential init using base-token decomposition
      - input emb : softmax(+alpha*pos)  (emphasize last)
      - lm_head   : softmax(-alpha*pos)  (emphasize first)
  4) sync config/generation_config bos/eos/pad to tokenizer ids
  5) save model+tokenizer to output_dir

Verify phase (immediately after build):
  - checks tokenizer ids + config alignment
  - forces full checks for core special tokens
  - shared tokens: sample + (all mismatches) + (core specials), prints top-k worst
  - new tokens: sample, compares against expected init, prints top-k worst + skipped

Usage:
  python build_and_verify_qwen3_expansion.py \
    --base_model Qwen/Qwen3-8B \
    --base_tokenizer Qwen/Qwen3-8B \
    --ext_tokenizer DopeorNope/FFT-expanded-naive \
    --output_dir ./qwen3-8b-fft-aligned \
    --alpha 2.0 \
    --device_map auto \
    --dtype auto \
    --dump_json ./exponent_new_token_decomp.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


# -------------------------
# Helpers
# -------------------------
def _parse_dtype(s: str):
    s = s.lower()
    if s == "auto":
        return "auto"
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s}")


def _resize_embeddings(model, new_size: int):
    # Transformers가 mean/cov 기반 초기화 안내문을 찍는 걸 피하고 싶으면 mean_resizing=False
    try:
        model.resize_token_embeddings(new_size, mean_resizing=False)
    except TypeError:
        model.resize_token_embeddings(new_size)


def _safe_decode_single(ext_tok, token_id: int) -> str:
    # "단일 토큰이 실제로 소비/생성하는 surface"를 얻는 가장 안전한 방법
    return ext_tok.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _softmax_weights(k: int, alpha: float, sign: float, device) -> torch.Tensor:
    # pos = 0..k-1 (1..k로 해도 softmax는 동일)
    pos = torch.arange(k, device=device, dtype=torch.float32)
    logits = sign * alpha * pos
    return torch.softmax(logits, dim=0)  # float32


def _gather_rows_with_buffer(
    weight: torch.Tensor,                 # [V, D]
    ids: List[int],                       # ids to gather
    buf_ids: List[int],                   # buffered base ids
    buf_tensor: torch.Tensor,             # [len(buf_ids), D] original rows for buf_ids
    buf_index: Dict[int, int],            # id -> index in buf_tensor
) -> torch.Tensor:
    """
    Efficiently gather rows for ids:
      - if id in buf_index: take from buf_tensor
      - else: take from weight[id]
    Returns: [N, D] on same device/dtype as weight/buf_tensor
    """
    device = weight.device
    dtype = weight.dtype
    N = len(ids)
    D = weight.shape[1]
    out = torch.empty((N, D), device=device, dtype=dtype)

    # split ids
    pos_buf = []
    ids_buf = []
    pos_nbuf = []
    ids_nbuf = []

    for p, tid in enumerate(ids):
        bi = buf_index.get(int(tid), -1)
        if bi >= 0:
            pos_buf.append(p)
            ids_buf.append(bi)
        else:
            pos_nbuf.append(p)
            ids_nbuf.append(int(tid))

    if pos_nbuf:
        t_ids = torch.tensor(ids_nbuf, device=device, dtype=torch.long)
        out[torch.tensor(pos_nbuf, device=device)] = weight.index_select(0, t_ids)

    if pos_buf:
        b_ids = torch.tensor(ids_buf, device=device, dtype=torch.long)
        out[torch.tensor(pos_buf, device=device)] = buf_tensor.index_select(0, b_ids)

    return out


@torch.no_grad()
def _exp_init_from_parts(
    part_rows: torch.Tensor,  # [K, D]
    alpha: float,
    mode: str,                # "input" or "output"
) -> torch.Tensor:
    """
    AdaptiVocab init:
      input  -> sign +1 (emphasize last)
      output -> sign -1 (emphasize first)
    """
    K = part_rows.shape[0]
    if K == 1:
        return part_rows[0].clone()

    sign = +1.0 if mode == "input" else -1.0
    w = _softmax_weights(K, alpha=alpha, sign=sign, device=part_rows.device).to(part_rows.dtype)  # [K]
    return (w[:, None] * part_rows).sum(dim=0)  # [D]


def _sync_qwen_special_ids(model, tok):
    """
    Qwen3 관례:
      bos/pad: <|endoftext|>
      eos    : <|im_end|>
      generation eos: [<|im_end|>, <|endoftext|>]
    """
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if eot is None or eot < 0:
        raise RuntimeError("Expanded tokenizer missing <|endoftext|>")
    if im_end is None or im_end < 0:
        raise RuntimeError("Expanded tokenizer missing <|im_end|>")

    model.config.bos_token_id = int(eot)
    model.config.pad_token_id = int(eot)
    model.config.eos_token_id = int(im_end)
    model.config.vocab_size = len(tok)

    if getattr(model, "generation_config", None) is None:
        model.generation_config = GenerationConfig.from_model_config(model.config)

    model.generation_config.bos_token_id = int(eot)
    model.generation_config.pad_token_id = int(eot)
    model.generation_config.eos_token_id = [int(im_end), int(eot)]


def _parse_core_special_tokens(arg: Optional[str]) -> List[str]:
    # 기본 핵심 토큰 + 자주 등장하는 qwen 컨트롤 토큰을 조금 포함
    default = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|image_pad|>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
        "<|fim_pad|>",
        "<|quad_start|>",
        "<|quad_end|>",
    ]
    if not arg:
        return default
    toks = [t.strip() for t in arg.split(",") if t.strip()]
    return toks if toks else default


def _check_config_files(output_dir: Path, tok) -> Tuple[bool, bool]:
    cfg_path = output_dir / "config.json"
    gen_path = output_dir / "generation_config.json"

    ok_cfg = True
    ok_gen = True

    if not cfg_path.exists():
        print("[FAIL] config.json missing in output_dir")
        ok_cfg = False
    else:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        bos = cfg.get("bos_token_id")
        pad = cfg.get("pad_token_id")
        eos = cfg.get("eos_token_id")
        vsz = cfg.get("vocab_size")
        eot = tok.convert_tokens_to_ids("<|endoftext|>")
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        print("\n[check:file] config.json:")
        print(f"  bos={bos} pad={pad} eos={eos} vocab_size={vsz}")
        if eot is not None and eot >= 0:
            ok_cfg &= (bos == eot) and (pad == eot)
        if im_end is not None and im_end >= 0:
            ok_cfg &= (eos == im_end)
        ok_cfg &= (vsz == len(tok))

    if not gen_path.exists():
        print("[WARN] generation_config.json missing in output_dir (recommended)")
        ok_gen = False
    else:
        gen = json.loads(gen_path.read_text(encoding="utf-8"))
        gbos = gen.get("bos_token_id")
        gpad = gen.get("pad_token_id")
        geos = gen.get("eos_token_id")
        eot = tok.convert_tokens_to_ids("<|endoftext|>")
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        print("\n[check:file] generation_config.json:")
        print(f"  bos={gbos} pad={gpad} eos={geos}")
        if eot is not None and eot >= 0:
            ok_gen &= (gbos == eot) and (gpad == eot)
        if im_end is not None and im_end >= 0:
            if isinstance(geos, list) and len(geos) >= 1:
                ok_gen &= (geos[0] == im_end) and (geos[-1] == eot)
            else:
                ok_gen &= (geos == im_end)

    return ok_cfg, ok_gen


# -------------------------
# Main
# -------------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()

    # Build inputs
    ap.add_argument("--base_model", default="Qwen/Qwen3-8B")
    ap.add_argument("--base_tokenizer", default="Qwen/Qwen3-8B")
    ap.add_argument("--ext_tokenizer", required=True, help="Expanded tokenizer repo-id or local path")
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--device_map", default=None, help='e.g. "auto" (recommended) or None')
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--dump_json", default=None, help="Optional: dump new-token decomposition records")
    ap.add_argument("--max_new_init", type=int, default=None, help="Debug: only init first N new tokens (by ext_id order)")

    # Verify options
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_shared", type=int, default=2000)
    ap.add_argument("--max_new", type=int, default=200)
    ap.add_argument("--top_k_shared", type=int, default=20)
    ap.add_argument("--top_k_new", type=int, default=20)
    ap.add_argument("--top_k_skipped_new", type=int, default=20)
    ap.add_argument("--core_special_tokens", type=str, default=None,
                    help="comma-separated list; if omitted uses a sensible default set")
    ap.add_argument("--skip_verify", action="store_true",
                    help="Build + save only (skip verification)")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # LOAD
    # -------------------------
    print("[LOAD] tokenizers ...")
    base_tok = AutoTokenizer.from_pretrained(args.base_tokenizer, use_fast=True, trust_remote_code=args.trust_remote_code)
    ext_tok = AutoTokenizer.from_pretrained(args.ext_tokenizer, use_fast=True, trust_remote_code=args.trust_remote_code)

    base_vocab: Dict[str, int] = base_tok.get_vocab()
    ext_vocab: Dict[str, int] = ext_tok.get_vocab()

    print(f"[tok] base len={len(base_tok)}")
    print(f"[tok] ext  len={len(ext_tok)}")

    # Identify mismatches + new tokens
    common = set(base_vocab).intersection(ext_vocab)
    mismatches = [(t, base_vocab[t], ext_vocab[t]) for t in common if base_vocab[t] != ext_vocab[t]]
    mismatches.sort(key=lambda x: x[1])
    new_tokens = [(t, ext_vocab[t]) for t in ext_vocab.keys() if t not in base_vocab]
    new_tokens.sort(key=lambda x: x[1])

    print(f"[analyze] shared={len(common)}  mismatches={len(mismatches)}  new_tokens={len(new_tokens)}")
    if mismatches:
        print("[analyze] first mismatches (token, base_id -> ext_id):")
        for t, bi, ei in mismatches[:20]:
            print(f"  {t!r}: {bi} -> {ei}")

    # Load model
    print("\n[LOAD] base model weights ...")
    torch_dtype = _parse_dtype(args.dtype)
    kwargs = dict(low_cpu_mem_usage=True, trust_remote_code=args.trust_remote_code)
    if torch_dtype != "auto":
        kwargs["torch_dtype"] = torch_dtype
    if args.device_map:
        kwargs["device_map"] = args.device_map

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **kwargs)
    model.eval()

    # Grab weights
    W_in = model.get_input_embeddings().weight
    out_emb = model.get_output_embeddings()
    if out_emb is None:
        raise RuntimeError("model.get_output_embeddings() is None; cannot align lm_head.")
    W_out = out_emb.weight

    tied = (W_in.data_ptr() == W_out.data_ptr())
    if tied:
        print("[WARN] input/output embeddings are tied. Output init will match input (cannot do separate + / -).")

    old_V = int(W_in.shape[0])
    new_V = int(len(ext_tok))
    D = int(W_in.shape[1])
    print(f"[model] old_V={old_V} -> new_V={new_V}  D={D}  dtype={W_in.dtype}  device={W_in.device}")

    # -------------------------
    # BUILD: resize
    # -------------------------
    print("\n[BUILD] resize_token_embeddings ...")
    _resize_embeddings(model, new_V)
    # refresh pointers after resize
    W_in = model.get_input_embeddings().weight
    W_out = model.get_output_embeddings().weight

    # -------------------------
    # BUILD: buffer base rows that might get overwritten but are needed as sources
    # -------------------------
    # We will overwrite some ext_ids that are < old_V (e.g., where new tokens occupy old special-token slots).
    overwritten_ids = {ext_id for _, ext_id in new_tokens if ext_id < old_V}
    src_ids = {bi for _, bi, _ in mismatches}
    buf_ids = sorted(overwritten_ids.union(src_ids))
    buf_index = {tid: i for i, tid in enumerate(buf_ids)}
    buf_in = W_in.index_select(0, torch.tensor(buf_ids, device=W_in.device, dtype=torch.long)).clone() if buf_ids else None
    buf_out = W_out.index_select(0, torch.tensor(buf_ids, device=W_out.device, dtype=torch.long)).clone() if buf_ids else None

    print(f"[BUILD] buffered base rows: {len(buf_ids)} (overwritten_ids={len(overwritten_ids)}, mismatch_sources={len(src_ids)})")

    # For fallback init
    mean_in = W_in[:min(old_V, W_in.shape[0])].mean(dim=0).clone()
    mean_out = W_out[:min(old_V, W_out.shape[0])].mean(dim=0).clone()

    # -------------------------
    # BUILD: init new tokens
    # -------------------------
    print("\n[BUILD] init NEW tokens (AdaptiVocab exponential init) ...")
    dump_records = []
    inited = 0
    skipped = 0

    # optional debug cap
    new_tokens_to_init = new_tokens
    if args.max_new_init is not None:
        new_tokens_to_init = new_tokens_to_init[: args.max_new_init]

    for tok_str, ext_id in new_tokens_to_init:
        surface = _safe_decode_single(ext_tok, ext_id)
        part_ids = base_tok.encode(surface, add_special_tokens=False)

        if not part_ids:
            # fallback
            W_in[ext_id].copy_(mean_in)
            if not tied:
                W_out[ext_id].copy_(mean_out)
            else:
                W_out[ext_id].copy_(W_in[ext_id])
            skipped += 1
            continue

        # gather constituent rows from "base weights" (use buffer if needed)
        parts_in = _gather_rows_with_buffer(
            W_in, part_ids, buf_ids, buf_in, buf_index
        ) if buf_in is not None else W_in.index_select(0, torch.tensor(part_ids, device=W_in.device, dtype=torch.long))

        vec_in = _exp_init_from_parts(parts_in, alpha=args.alpha, mode="input")
        W_in[ext_id].copy_(vec_in)

        if not tied:
            parts_out = _gather_rows_with_buffer(
                W_out, part_ids, buf_ids, buf_out, buf_index
            ) if buf_out is not None else W_out.index_select(0, torch.tensor(part_ids, device=W_out.device, dtype=torch.long))
            vec_out = _exp_init_from_parts(parts_out, alpha=args.alpha, mode="output")
            W_out[ext_id].copy_(vec_out)
        else:
            W_out[ext_id].copy_(W_in[ext_id])

        inited += 1

        if args.dump_json:
            dump_records.append(
                {
                    "token": tok_str,
                    "ext_id": int(ext_id),
                    "surface": surface,
                    "base_part_ids": [int(i) for i in part_ids],
                    "base_part_tokens": base_tok.convert_ids_to_tokens(part_ids),
                }
            )

    print(f"[BUILD] new tokens: considered={len(new_tokens_to_init)}  inited={inited}  fallback_mean={skipped}")

    # -------------------------
    # BUILD: copy mismatched shared tokens (important for shifted special tokens)
    # -------------------------
    print("\n[BUILD] remap shared mismatched tokens (copy by token string) ...")
    if mismatches:
        for tok_str, base_id, ext_id in mismatches:
            # source from buffered base row (must exist)
            if base_id in buf_index and buf_in is not None and buf_out is not None:
                bi = buf_index[base_id]
                W_in[ext_id].copy_(buf_in[bi])
                if not tied:
                    W_out[ext_id].copy_(buf_out[bi])
                else:
                    W_out[ext_id].copy_(W_in[ext_id])
            else:
                # fallback: read current (should still be base row if not overwritten)
                W_in[ext_id].copy_(W_in[base_id])
                if not tied:
                    W_out[ext_id].copy_(W_out[base_id])
                else:
                    W_out[ext_id].copy_(W_in[ext_id])

        print(f"[BUILD] remapped mismatch rows: {len(mismatches)}")
    else:
        print("[BUILD] no mismatches found (pure extension or already aligned).")

    # -------------------------
    # BUILD: sync config/generation_config
    # -------------------------
    print("\n[BUILD] sync special token ids in config ...")
    _sync_qwen_special_ids(model, ext_tok)
    print(f"[BUILD] synced bos/pad/eos + vocab_size={model.config.vocab_size}")

    # -------------------------
    # SAVE
    # -------------------------
    print("\n[SAVE] saving model+tokenizer ...")
    model.save_pretrained(out_dir, safe_serialization=True)
    model.generation_config.save_pretrained(out_dir)
    ext_tok.save_pretrained(out_dir)
    print(f"[SAVE] wrote: {out_dir}")

    if args.dump_json:
        dump_path = Path(args.dump_json)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(json.dumps(dump_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[SAVE] wrote decomposition dump: {dump_path}")

    if args.skip_verify:
        print("\n[VERIFY] skipped by --skip_verify")
        return

    # -------------------------
    # VERIFY
    # -------------------------
    print("\n" + "=" * 80)
    print("[VERIFY] start")
    print("=" * 80)

    core_special = _parse_core_special_tokens(args.core_special_tokens)

    # Load tokenizer back from saved dir (ensures we verify what was saved)
    aligned_tok = AutoTokenizer.from_pretrained(str(out_dir), use_fast=True)
    aligned_vocab = aligned_tok.get_vocab()

    print(f"[verify] aligned tokenizer len={len(aligned_tok)} (from output_dir)")

    # (A) Check special token ids existence
    print("\n[verify] CORE special tokens ids:")
    for t in core_special:
        tid = aligned_tok.convert_tokens_to_ids(t)
        print(f"  {t!r}: id={tid}")

    # (B) Check file configs match tokenizer ids
    ok_cfg, ok_gen = _check_config_files(out_dir, aligned_tok)
    print("\n[verify:result] config.json alignment:", "PASS" if ok_cfg else "FAIL")
    print("[verify:result] generation_config.json alignment:", "PASS" if ok_gen else "FAIL (or missing)")

    # (C) Shared token copy check (sample + forced)
    # Build forced set: all mismatches + all core specials
    forced_shared = set([t for (t, _, _) in mismatches])
    forced_shared.update(core_special)

    common2 = list(set(base_vocab).intersection(aligned_vocab))
    mism2 = [(t, base_vocab[t], aligned_vocab[t]) for t in common2 if base_vocab[t] != aligned_vocab[t]]
    mism2_set = set([t for (t, _, _) in mism2])

    # Sample shared tokens (excluding forced to diversify)
    rest = [t for t in common2 if t not in forced_shared]
    sample_n = min(args.max_shared, len(rest))
    sampled = rng.sample(rest, sample_n) if sample_n > 0 else []
    to_check_shared = list(set(sampled).union(forced_shared).union(mism2_set))
    # Keep only tokens present in both vocabs
    to_check_shared = [t for t in to_check_shared if (t in base_vocab and t in aligned_vocab)]

    print(f"\n[verify] shared tokens: common={len(common2)} mismatches={len(mism2)} will_check={len(to_check_shared)}")

    # tolerance
    atol_shared = 5e-3 if W_in.dtype in (torch.float16, torch.bfloat16) else 1e-6

    shared_rows = []
    for t in to_check_shared:
        bi = int(base_vocab[t])
        ai = int(aligned_vocab[t])

        # expected = base row (use buffer if base id got overwritten)
        exp_in = buf_in[buf_index[bi]] if (buf_in is not None and bi in buf_index) else W_in[bi]
        exp_out = buf_out[buf_index[bi]] if (buf_out is not None and bi in buf_index) else W_out[bi]

        act_in = W_in[ai]
        act_out = W_out[ai]

        din = float((exp_in.float() - act_in.float()).abs().max().item())
        dout = float((exp_out.float() - act_out.float()).abs().max().item())
        score = max(din, dout)

        shared_rows.append({
            "token": t,
            "base_id": bi,
            "aligned_id": ai,
            "din": din,
            "dout": dout,
            "score": score,
            "critical": (t in core_special),
        })

    shared_rows.sort(key=lambda x: x["score"], reverse=True)
    max_in = max((r["din"] for r in shared_rows), default=0.0)
    max_out = max((r["dout"] for r in shared_rows), default=0.0)

    print("\n[verify] shared-token copy summary:")
    print(f"  overall max_diff_in ={max_in:.6g}  (atol~{atol_shared})")
    print(f"  overall max_diff_out={max_out:.6g}  (atol~{atol_shared})")
    print("  =>", "PASS" if (max_in <= atol_shared and max_out <= atol_shared) else "FAIL")

    # Core specials full report
    print("\n[verify] CORE special tokens (full check):")
    for t in core_special:
        if t not in base_vocab or t not in aligned_vocab:
            print(f"  {t!r}: SKIP (missing in vocab: base={t in base_vocab}, aligned={t in aligned_vocab})")
            continue
        bi = int(base_vocab[t])
        ai = int(aligned_vocab[t])
        exp_in = buf_in[buf_index[bi]] if (buf_in is not None and bi in buf_index) else W_in[bi]
        exp_out = buf_out[buf_index[bi]] if (buf_out is not None and bi in buf_index) else W_out[bi]
        din = float((exp_in.float() - W_in[ai].float()).abs().max().item())
        dout = float((exp_out.float() - W_out[ai].float()).abs().max().item())
        ok = (din <= atol_shared and dout <= atol_shared)
        print(f"  {t!r}: base_id={bi} -> aligned_id={ai}  max_in={din:.6g} max_out={dout:.6g} => {'PASS' if ok else 'FAIL'}")

    # Top-k shared
    print(f"\n[top-k] shared worst (k={min(args.top_k_shared, len(shared_rows))}):")
    for i, r in enumerate(shared_rows[: args.top_k_shared], start=1):
        flag = "CRIT" if r["critical"] else "    "
        bad = " !!" if (r["din"] > atol_shared or r["dout"] > atol_shared) else ""
        print(f"  #{i:02d} [{flag}] score={r['score']:.6g} in={r['din']:.6g} out={r['dout']:.6g} "
              f"{r['token']!r} (base {r['base_id']} -> aligned {r['aligned_id']}){bad}")

    # (D) New token init check (sample)
    new_tokens2 = [t for t in aligned_vocab.keys() if t not in base_vocab]
    sample_new = rng.sample(new_tokens2, min(args.max_new, len(new_tokens2))) if new_tokens2 else []
    print(f"\n[verify] new tokens: total={len(new_tokens2)} sample_check={len(sample_new)}")

    new_rows = []
    skipped_new = []

    for tok_str in sample_new:
        tid = int(aligned_vocab[tok_str])

        surface = aligned_tok.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        part_ids = base_tok.encode(surface, add_special_tokens=False)

        if not part_ids:
            skipped_new.append((tok_str, tid, surface))
            continue

        # expected from base rows
        parts_in = _gather_rows_with_buffer(W_in, part_ids, buf_ids, buf_in, buf_index) if buf_in is not None else \
            W_in.index_select(0, torch.tensor(part_ids, device=W_in.device, dtype=torch.long))
        parts_out = _gather_rows_with_buffer(W_out, part_ids, buf_ids, buf_out, buf_index) if buf_out is not None else \
            W_out.index_select(0, torch.tensor(part_ids, device=W_out.device, dtype=torch.long))

        exp_in = _exp_init_from_parts(parts_in, alpha=args.alpha, mode="input").float()
        exp_out = _exp_init_from_parts(parts_out, alpha=args.alpha, mode="output").float()

        act_in = W_in[tid].float()
        act_out = W_out[tid].float()

        cos_in = float(F.cosine_similarity(exp_in, act_in, dim=0).item())
        cos_out = float(F.cosine_similarity(exp_out, act_out, dim=0).item())
        score = min(cos_in, cos_out)

        parts_preview = base_tok.convert_ids_to_tokens(part_ids)[:12]
        new_rows.append({
            "token": tok_str,
            "id": tid,
            "surface": surface,
            "k": len(part_ids),
            "parts_preview": parts_preview,
            "cos_in": cos_in,
            "cos_out": cos_out,
            "score": score,
        })

    new_rows.sort(key=lambda x: x["score"])  # worst first

    if new_rows:
        print("\n[verify] new-token init summary (sampled):")
        print(f"  checked={len(new_rows)} skipped(no decomposition)={len(skipped_new)}")
        print(f"  min_cos (input) ={min(r['cos_in'] for r in new_rows):.6f}  mean={sum(r['cos_in'] for r in new_rows)/len(new_rows):.6f}")
        print(f"  min_cos (output)={min(r['cos_out'] for r in new_rows):.6f}  mean={sum(r['cos_out'] for r in new_rows)/len(new_rows):.6f}")

        print(f"\n[top-k] new-token worst (k={min(args.top_k_new, len(new_rows))}):")
        for i, r in enumerate(new_rows[: args.top_k_new], start=1):
            surf = r["surface"].replace("\n", "\\n")
            if len(surf) > 120:
                surf = surf[:117] + "..."
            more = "" if r["k"] <= 12 else f" ... (+{r['k']-12} more)"
            print(f"  #{i:02d} score(mincos)={r['score']:.6f} cos_in={r['cos_in']:.6f} cos_out={r['cos_out']:.6f} "
                  f"id={r['id']} token={r['token']!r} surface={surf!r} parts={r['parts_preview']}{more}")
    else:
        print("\n[verify] new-token init: no decomposable tokens in the sample (all skipped).")

    if skipped_new:
        print(f"\n[top-k] new tokens skipped (no decomposition) (k={min(args.top_k_skipped_new, len(skipped_new))}):")
        for i, (tok_str, tid, surface) in enumerate(skipped_new[: args.top_k_skipped_new], start=1):
            surf = surface.replace("\n", "\\n")
            if len(surf) > 120:
                surf = surf[:117] + "..."
            print(f"  #{i:02d} token={tok_str!r} id={tid} surface={surf!r}")

    print("\n[VERIFY] done ✅")


if __name__ == "__main__":
    main()
