#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
확장 토크나이저에 맞춰 모델 임베딩(+ 출력 lm_head) 확장 + "서브워드 평균" 초기화
=========================================================================

목표:
- tokenizer가 확장되면 (len(tokenizer) 증가) 모델의 input embedding과 output layer(lm_head)도 확장 필요
- 새로 추가된 토큰의 weight는 "기존(base) 토크나이저로 그 토큰 문자열을 분해한 subword들의 평균"으로 초기화
- 초기화 후 모델 + 새 tokenizer를 save_pretrained로 저장

주의:
- 이 스크립트는 "확장된 tokenizer"가 이미 만들어져 있다는 가정:
    - base tokenizer dir: ./pangea_qwen3_tokenizer
    - expanded tokenizer dir: ./pangea_qwen3_tokenizer_safe
- 모델은 기본적으로 BASE_MODEL(=Qwen/Qwen3-8B)에서 로드하지만,
  파인튜닝 체크포인트 경로를 --model_path로 넣어도 됨.

실행 예시:
  python init_new_token_embeddings_by_subword_mean.py \
    --model_path Qwen/Qwen3-8B \
    --base_tokenizer ./pangea_qwen3_tokenizer \
    --new_tokenizer ./pangea_qwen3_tokenizer_safe \
    --save_dir ./qwen3_8b_with_expanded_tokenizer \
    --dtype bfloat16

중요:
- 모델/토크나이저를 "반드시 같은 save_dir"에 저장해두면 이후 학습이 편함.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-8B", help="Base model or checkpoint path.")
    p.add_argument("--base_tokenizer", type=str, default="./pangea_qwen3_tokenizer", help="Old/base tokenizer dir.")
    p.add_argument("--new_tokenizer", type=str, default="./pangea_qwen3_tokenizer_safe", help="Expanded tokenizer dir.")
    p.add_argument("--save_dir", type=str, default="./qwen3_8b_expanded_init", help="Where to save model+tokenizer.")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device", type=str, default=None, help="e.g. cuda, cuda:0, cpu. Default: auto.")
    p.add_argument("--trust_remote_code", action="store_true", help="Enable if model/tokenizer needs it.")
    p.add_argument("--max_shard_size", type=str, default="5GB", help="Sharding size for save_pretrained.")
    p.add_argument("--report_file", type=str, default=None, help="Optional JSON report path (default: save_dir/_init_report.json).")
    return p.parse_args()


def dtype_from_str(s: str):
    if s == "float32":
        return torch.float32
    if s == "float16":
        return torch.float16
    if s == "bfloat16":
        return torch.bfloat16
    raise ValueError(s)


def get_device(user_device: Optional[str]) -> torch.device:
    if user_device:
        return torch.device(user_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def token_to_piece_text(tokenizer, tok: str) -> str:
    """
    Fast tokenizer의 내부 토큰 문자열(tok)을 사람이 쓰는 문자열 조각(piece)로 변환.
    가장 안전한 방법은 convert_tokens_to_string 사용.
    """
    try:
        s = tokenizer.convert_tokens_to_string([tok])
        return s
    except Exception:
        # fallback: handle common whitespace markers
        if tok.startswith("Ġ") or tok.startswith("▁"):
            return " " + tok[1:]
        return tok


def mean_of_rows(weight: torch.Tensor, ids: List[int]) -> torch.Tensor:
    """
    weight[ids] 평균 (float32로 계산한 뒤 원 dtype로)
    """
    rows = weight.index_select(0, torch.tensor(ids, device=weight.device))
    return rows.float().mean(dim=0)


@torch.no_grad()
def main():
    args = parse_args()
    device = get_device(args.device)
    dtype = dtype_from_str(args.dtype)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_file) if args.report_file else (save_dir / "_init_report.json")

    print("[load] base tokenizer:", args.base_tokenizer)
    base_tok = AutoTokenizer.from_pretrained(args.base_tokenizer, use_fast=True, trust_remote_code=args.trust_remote_code)

    print("[load] new tokenizer:", args.new_tokenizer)
    new_tok = AutoTokenizer.from_pretrained(args.new_tokenizer, use_fast=True, trust_remote_code=args.trust_remote_code)

    print("[load] model:", args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(device)
    model.eval()

    in_emb = model.get_input_embeddings()
    if in_emb is None:
        raise RuntimeError("model.get_input_embeddings() returned None.")
    old_vocab_size = in_emb.weight.shape[0]
    new_vocab_size = len(new_tok)

    print(f"[sizes] old_vocab_size(model)={old_vocab_size}  new_vocab_size(tokenizer)={new_vocab_size}")

    if new_vocab_size <= old_vocab_size:
        print("[info] tokenizer size did not increase. Nothing to resize/init.")
        # Still save tokenizer+model for convenience
        model.save_pretrained(save_dir, max_shard_size=args.max_shard_size)
        new_tok.save_pretrained(save_dir)
        return

    # Resize (this will add rows initialized randomly)
    print("[resize] resize_token_embeddings ...")
    model.resize_token_embeddings(new_vocab_size)
    model.tie_weights()  # if tied, lm_head will share with embeddings

    in_emb = model.get_input_embeddings()
    out_emb = model.get_output_embeddings()  # could be tied or separate
    in_w = in_emb.weight

    # Detect tying
    tied = False
    if out_emb is not None and hasattr(out_emb, "weight") and out_emb.weight is not None:
        tied = (out_emb.weight.data_ptr() == in_w.data_ptr())

    if out_emb is not None and hasattr(out_emb, "bias") and out_emb.bias is not None:
        out_bias = out_emb.bias
    else:
        out_bias = None

    base_special_ids = set(getattr(base_tok, "all_special_ids", []))
    new_special_ids = set(getattr(new_tok, "all_special_ids", []))

    # We'll init only newly-added ids (from old_vocab_size to new_vocab_size-1)
    new_ids = list(range(old_vocab_size, new_vocab_size))

    stats = {
        "old_vocab_size_model": int(old_vocab_size),
        "new_vocab_size_tokenizer": int(new_vocab_size),
        "num_new_ids": int(len(new_ids)),
        "tied_input_output": bool(tied),
        "skipped_special": 0,
        "skipped_empty_decomp": 0,
        "skipped_contains_unk": 0,
        "initialized": 0,
        "examples": [],
    }

    # For unknown token id (if any)
    unk_id = getattr(base_tok, "unk_token_id", None)

    print(f"[init] initializing {len(new_ids)} new token rows by subword-mean ...")
    for idx, new_id in enumerate(new_ids):
        tok_str = new_tok.convert_ids_to_tokens(new_id)
        if tok_str is None:
            stats["skipped_empty_decomp"] += 1
            continue

        # Skip special tokens (if any were appended)
        if new_id in new_special_ids or tok_str in getattr(new_tok, "all_special_tokens", []):
            stats["skipped_special"] += 1
            continue

        piece = token_to_piece_text(new_tok, tok_str)
        if piece == "":
            stats["skipped_empty_decomp"] += 1
            continue

        # Decompose using *base* tokenizer
        base_ids = base_tok(piece, add_special_tokens=False)["input_ids"]

        # Safety: filter out special ids, and any ids >= old_vocab_size (shouldn't happen)
        base_ids = [i for i in base_ids if i not in base_special_ids and i < old_vocab_size]

        if len(base_ids) == 0:
            stats["skipped_empty_decomp"] += 1
            continue

        if unk_id is not None and (unk_id in base_ids):
            # If decomposition contains UNK, the average is meaningless; skip (keep random init)
            stats["skipped_contains_unk"] += 1
            continue

        # Input embedding init
        vec_in = mean_of_rows(in_w, base_ids)
        in_w[new_id].copy_(vec_in.to(dtype=in_w.dtype))

        # Output embedding init (if not tied)
        if out_emb is not None and hasattr(out_emb, "weight") and out_emb.weight is not None and not tied:
            out_w = out_emb.weight
            vec_out = mean_of_rows(out_w, base_ids)
            out_w[new_id].copy_(vec_out.to(dtype=out_w.dtype))

        # Output bias init (if exists)
        if out_bias is not None and out_bias.numel() >= new_vocab_size:
            # bias shape [vocab]
            b = out_bias.index_select(0, torch.tensor(base_ids, device=out_bias.device)).float().mean()
            out_bias[new_id].copy_(b.to(dtype=out_bias.dtype))

        stats["initialized"] += 1

        # store a few examples
        if len(stats["examples"]) < 20:
            stats["examples"].append({
                "new_id": int(new_id),
                "token": tok_str,
                "piece": piece,
                "base_ids": base_ids[:50],
                "base_tokens": base_tok.convert_ids_to_tokens(base_ids)[:50],
            })

        if (idx + 1) % 1000 == 0:
            print(f"  ... {idx+1}/{len(new_ids)}")

    print("[save] saving model + tokenizer to:", str(save_dir))
    model.save_pretrained(save_dir, max_shard_size=args.max_shard_size)
    new_tok.save_pretrained(save_dir)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("[done] init report saved:", str(report_path))
    print("[stats]", {k: stats[k] for k in ["num_new_ids", "initialized", "skipped_special", "skipped_empty_decomp", "skipped_contains_unk", "tied_input_output"]})
    print("[important] 이후 학습/추론에서는 이 save_dir에서 model/tokenizer 같이 로드하세요.")
    print("           예: model=AutoModelForCausalLM.from_pretrained(save_dir), tok=AutoTokenizer.from_pretrained(save_dir)")


if __name__ == "__main__":
    main()
