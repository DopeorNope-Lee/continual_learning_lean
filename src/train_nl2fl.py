#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
)

# -----------------------------
# 1) Dataclasses: model/data/train/wandb
# -----------------------------
@dataclass
class ModelConfig:
    model_name_or_path: str = field(default="Qwen/Qwen3-8B")
    trust_remote_code: bool = field(default=False)
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={"help": "가능하면 'flash_attention_2' 지정. 없으면 None."},
    )


@dataclass
class DataConfig:
    dataset_name: str = field(default="Goedel-LM/Goedel-Pset-v1")
    dataset_split: str = field(default="train")

    # split / sampling
    eval_ratio: float = field(default=0.001, metadata={"help": "eval 비율(0이면 eval off)"})
    seed: int = field(default=42)

    # formatting
    max_seq_length: int = field(default=4096)
    wrap_in_code_block: bool = field(default=True)
    enable_thinking: bool = field(
        default=False,
        metadata={"help": "훈련 시 think 생성 허용 여부. Lean 코드만 원하면 False 권장."},
    )
    system_prompt: str = field(
        default=(
            "You are an expert Lean 4 formalizer. "
            "Convert the given natural-language math problem into Lean 4 code that compiles with Mathlib. "
            "Return ONLY Lean 4 code (no extra explanations)."
        )
    )

    # preprocessing performance
    preprocessing_num_proc: int = field(default=8)
    preprocessing_batch_size: int = field(default=256)

    # optional truncation of dataset (debug)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)


@dataclass
class TrainConfig:
    output_dir: str = field(default="./qwen3-8b-goedel-pset-fft")

    # core hyperparams
    num_train_epochs: float = field(default=1.0)
    learning_rate: float = field(default=2e-5)
    weight_decay: float = field(default=0.1)
    warmup_ratio: float = field(default=0.03)
    lr_scheduler_type: str = field(default="cosine")  # linear, cosine, etc.
    optim: str = field(default="adamw_torch_fused")

    # batch
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=8)

    # stability
    max_grad_norm: float = field(default=1.0)
    gradient_checkpointing: bool = field(default=True)

    # precision
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    tf32: bool = field(default=True)

    # logging / save / eval
    logging_steps: int = field(default=10)
    logging_first_step: bool = field(default=True)

    evaluation_strategy: str = field(default="steps")  # steps/no
    eval_steps: int = field(default=2000)

    save_strategy: str = field(default="steps")
    save_steps: int = field(default=2000)
    save_total_limit: int = field(default=2)

    # dataloader
    dataloader_num_workers: int = field(default=4)
    dataloader_pin_memory: bool = field(default=True)

    # distributed
    ddp_find_unused_parameters: bool = field(default=False)

    # deepspeed config path (네가 가진 json 경로 넣으면 됨)
    deepspeed: Optional[str] = field(default="ds_zero3_bf16.json")

    # resume
    resume_from_checkpoint: Optional[str] = field(default=None)


@dataclass
class WandbConfig:
    wandb_enabled: bool = field(default=True)
    wandb_project: str = field(default="qwen3-fft")
    wandb_entity: Optional[str] = field(default=None)
    wandb_group: Optional[str] = field(default=None)
    wandb_name: Optional[str] = field(default=None)
    wandb_name_prefix: str = field(default="")
    wandb_tags: Optional[str] = field(default=None, metadata={"help": "comma-separated, ex: fft,goedel,qwen3"})
    wandb_notes: Optional[str] = field(default=None)
    wandb_mode: str = field(default="online", metadata={"help": "online/offline/disabled"})


# -----------------------------
# 2) Collator: pad + label pad(-100)
# -----------------------------
class CausalLMCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int] = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_inputs = {
            "input_ids": [f["input_ids"] for f in features],
            "attention_mask": [f["attention_mask"] for f in features],
        }
        batch = self.tokenizer.pad(
            batch_inputs,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]
        labels = []
        for f in features:
            l = f["labels"]
            if len(l) < max_len:
                l = l + [-100] * (max_len - len(l))
            labels.append(l)

        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


# -----------------------------
# 3) Helpers
# -----------------------------
_THEOREM_RE = re.compile(r"\btheorem\s+([A-Za-z0-9_']+)")
_LEMMA_RE = re.compile(r"\blemma\s+([A-Za-z0-9_']+)")
_DEF_RE = re.compile(r"\bdef\s+([A-Za-z0-9_']+)")


def extract_decl_name(formal_stmt: str) -> Optional[str]:
    if not isinstance(formal_stmt, str):
        return None
    for pat in (_THEOREM_RE, _LEMMA_RE, _DEF_RE):
        m = pat.search(formal_stmt)
        if m:
            return m.group(1)
    return None


def short_id(x: str) -> str:
    return (x or "").strip().split("/")[-1]


def is_main_process() -> bool:
    # torchrun / deepspeed 모두 RANK를 보통 넣어줌
    return int(os.environ.get("RANK", "0")) == 0


def build_run_name(model_cfg: ModelConfig, data_cfg: DataConfig, train_cfg: TrainConfig, wb_cfg: WandbConfig) -> str:
    if wb_cfg.wandb_name:
        return wb_cfg.wandb_name

    model_s = short_id(model_cfg.model_name_or_path)
    data_s = short_id(data_cfg.dataset_name)
    bs = train_cfg.per_device_train_batch_size
    ga = train_cfg.gradient_accumulation_steps

    # 너무 길지 않게 핵심만
    name = (
        f"{wb_cfg.wandb_name_prefix}"
        f"{model_s}__{data_s}"
        f"__seq{data_cfg.max_seq_length}"
        f"__lr{train_cfg.learning_rate:g}"
        f"__bs{bs}ga{ga}"
        f"__ep{train_cfg.num_train_epochs:g}"
        f"__think{int(data_cfg.enable_thinking)}"
        f"__cb{int(data_cfg.wrap_in_code_block)}"
    )
    # W&B name 길이 과도 방지(대충)
    return name[:180]


def build_group_name(model_cfg: ModelConfig, data_cfg: DataConfig, wb_cfg: WandbConfig) -> str:
    if wb_cfg.wandb_group:
        return wb_cfg.wandb_group
    return f"{short_id(model_cfg.model_name_or_path)}__{short_id(data_cfg.dataset_name)}"


def setup_wandb(wb_cfg: WandbConfig, run_name: str, group_name: str, flat_config: Dict[str, Any]):
    if not wb_cfg.wandb_enabled or wb_cfg.wandb_mode.lower() == "disabled":
        os.environ["WANDB_MODE"] = "disabled"
        return

    # 멀티 GPU에서 중복 run 생성 방지: non-main은 wandb 비활성
    if not is_main_process():
        os.environ["WANDB_MODE"] = "disabled"
        return

    os.environ["WANDB_MODE"] = wb_cfg.wandb_mode
    os.environ["WANDB_PROJECT"] = wb_cfg.wandb_project
    if wb_cfg.wandb_entity:
        os.environ["WANDB_ENTITY"] = wb_cfg.wandb_entity

    try:
        import wandb
    except Exception as e:
        raise RuntimeError(
            "wandb가 설치되어 있지 않거나 import 실패. `pip install wandb` 후 다시 시도해줘."
        ) from e

    tags = None
    if wb_cfg.wandb_tags:
        tags = [t.strip() for t in wb_cfg.wandb_tags.split(",") if t.strip()]

    wandb.init(
        project=wb_cfg.wandb_project,
        entity=wb_cfg.wandb_entity,
        group=group_name,
        name=run_name,
        tags=tags,
        notes=wb_cfg.wandb_notes,
        config=flat_config,
    )


def flatten_config(model_cfg: ModelConfig, data_cfg: DataConfig, train_cfg: TrainConfig, wb_cfg: WandbConfig) -> Dict[str, Any]:
    # W&B config에 보기 좋게 기록될 하이퍼파라미터들
    return {
        # model
        "model_name_or_path": model_cfg.model_name_or_path,
        "attn_implementation": model_cfg.attn_implementation,
        # data
        "dataset_name": data_cfg.dataset_name,
        "eval_ratio": data_cfg.eval_ratio,
        "max_seq_length": data_cfg.max_seq_length,
        "wrap_in_code_block": data_cfg.wrap_in_code_block,
        "enable_thinking": data_cfg.enable_thinking,
        # train
        "output_dir": train_cfg.output_dir,
        "num_train_epochs": train_cfg.num_train_epochs,
        "learning_rate": train_cfg.learning_rate,
        "weight_decay": train_cfg.weight_decay,
        "warmup_ratio": train_cfg.warmup_ratio,
        "lr_scheduler_type": train_cfg.lr_scheduler_type,
        "optim": train_cfg.optim,
        "per_device_train_batch_size": train_cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": train_cfg.gradient_accumulation_steps,
        "max_grad_norm": train_cfg.max_grad_norm,
        "bf16": train_cfg.bf16,
        "fp16": train_cfg.fp16,
        "tf32": train_cfg.tf32,
        "gradient_checkpointing": train_cfg.gradient_checkpointing,
        "deepspeed": train_cfg.deepspeed,
        # wandb
        "wandb_project": wb_cfg.wandb_project,
        "wandb_group": wb_cfg.wandb_group,
        "wandb_mode": wb_cfg.wandb_mode,
    }


# -----------------------------
# 4) Main
# -----------------------------
def main():
    parser = HfArgumentParser((ModelConfig, DataConfig, TrainConfig, WandbConfig))
    model_cfg, data_cfg, train_cfg, wb_cfg = parser.parse_args_into_dataclasses()

    # seed / tf32
    set_seed(data_cfg.seed)
    if train_cfg.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # run/group naming
    run_name = build_run_name(model_cfg, data_cfg, train_cfg, wb_cfg)
    group_name = build_group_name(model_cfg, data_cfg, wb_cfg)

    # build training arguments
    eval_enabled = data_cfg.eval_ratio and data_cfg.eval_ratio > 0.0
    evaluation_strategy = train_cfg.evaluation_strategy if eval_enabled else "no"

    report_to = ["wandb"] if (wb_cfg.wandb_enabled and wb_cfg.wandb_mode.lower() != "disabled") else ["none"]

    # deepspeed path 처리
    ds_cfg = train_cfg.deepspeed
    if ds_cfg is not None:
        ds_cfg = ds_cfg.strip()
        if ds_cfg == "" or ds_cfg.lower() == "none":
            ds_cfg = None

    training_args = TrainingArguments(
        output_dir=train_cfg.output_dir,
        overwrite_output_dir=True,

        run_name=run_name,
        report_to=report_to,

        num_train_epochs=train_cfg.num_train_epochs,
        learning_rate=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        warmup_ratio=train_cfg.warmup_ratio,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        optim=train_cfg.optim,

        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=train_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,

        max_grad_norm=train_cfg.max_grad_norm,
        gradient_checkpointing=train_cfg.gradient_checkpointing,

        bf16=train_cfg.bf16,
        fp16=train_cfg.fp16,
        tf32=train_cfg.tf32,

        logging_steps=train_cfg.logging_steps,
        logging_first_step=train_cfg.logging_first_step,

        eval_strategy=evaluation_strategy,
        eval_steps=train_cfg.eval_steps,

        save_strategy=train_cfg.save_strategy,
        save_steps=train_cfg.save_steps,
        save_total_limit=train_cfg.save_total_limit,
        save_safetensors=True,

        dataloader_num_workers=train_cfg.dataloader_num_workers,
        dataloader_pin_memory=train_cfg.dataloader_pin_memory,

        ddp_find_unused_parameters=train_cfg.ddp_find_unused_parameters,

        deepspeed=ds_cfg,
    )

    # W&B init(메인 프로세스에서만)
    flat_cfg = flatten_config(model_cfg, data_cfg, train_cfg, wb_cfg)
    setup_wandb(wb_cfg, run_name=run_name, group_name=group_name, flat_config=flat_cfg)

    # tokenizer/model
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.model_name_or_path,
        trust_remote_code=model_cfg.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name_or_path,
        trust_remote_code=model_cfg.trust_remote_code,
        torch_dtype=torch.bfloat16 if train_cfg.bf16 else None,
        attn_implementation=model_cfg.attn_implementation,
    )
    model.config.use_cache = False  # training 안정

    if train_cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # dataset
    ds = load_dataset(data_cfg.dataset_name, split=data_cfg.dataset_split)

    # 디버그용 샘플 제한
    if data_cfg.max_train_samples is not None:
        ds = ds.select(range(min(len(ds), data_cfg.max_train_samples)))

    # eval split
    if eval_enabled:
        split = ds.train_test_split(test_size=data_cfg.eval_ratio, seed=data_cfg.seed, shuffle=True)
        train_ds, eval_ds = split["train"], split["test"]
        if data_cfg.max_eval_samples is not None:
            eval_ds = eval_ds.select(range(min(len(eval_ds), data_cfg.max_eval_samples)))
    else:
        train_ds, eval_ds = ds, None

    eos_text = tokenizer.eos_token or ""

    def preprocess_batch(batch):
        input_ids_list, attention_mask_list, labels_list = [], [], []

        for informal, formal, pid in zip(
            batch["informal_statement"],
            batch["formal_statement"],
            batch["problem_id"],
        ):
            informal = (informal or "").strip()
            formal = (formal or "").strip()

            decl_name = extract_decl_name(formal) or pid

            user_prompt = (
                "Please autoformalize the following natural language problem statement in Lean 4.\n"
                f"Use the following theorem name: {decl_name}\n"
                "The natural language statement is:\n"
                f"{informal}\n"
                "Return ONLY Lean 4 code."
            )

            messages = []
            if data_cfg.system_prompt:
                messages.append({"role": "system", "content": data_cfg.system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,   # assistant 시작 토큰까지 포함
                enable_thinking=data_cfg.enable_thinking,
            )

            if data_cfg.wrap_in_code_block:
                completion_text = f"```lean4\n{formal}\n```{eos_text}"
            else:
                completion_text = f"{formal}{eos_text}"

            prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
            completion_ids = tokenizer(completion_text, add_special_tokens=False).input_ids

            # 길이 초과 처리: completion 우선, prompt는 왼쪽에서 잘라냄
            max_len = data_cfg.max_seq_length
            if len(prompt_ids) + len(completion_ids) > max_len:
                if len(completion_ids) >= max_len:
                    completion_ids = completion_ids[:max_len]
                    if tokenizer.eos_token_id is not None:
                        completion_ids[-1] = tokenizer.eos_token_id
                    prompt_ids = []
                else:
                    keep_prompt = max_len - len(completion_ids)
                    prompt_ids = prompt_ids[-keep_prompt:]

            input_ids = prompt_ids + completion_ids
            labels = ([-100] * len(prompt_ids)) + completion_ids
            attention_mask = [1] * len(input_ids)

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
        }

    remove_cols = train_ds.column_names

    # 멀티 GPU에서 map 중복 작업 줄이기(메인 먼저 캐시 생성)
    with training_args.main_process_first(desc="Tokenizing dataset"):
        train_ds = train_ds.map(
            preprocess_batch,
            batched=True,
            batch_size=data_cfg.preprocessing_batch_size,
            num_proc=data_cfg.preprocessing_num_proc,
            remove_columns=remove_cols,
            desc="Tokenizing train",
        )
        if eval_ds is not None:
            eval_ds = eval_ds.map(
                preprocess_batch,
                batched=True,
                batch_size=data_cfg.preprocessing_batch_size,
                num_proc=data_cfg.preprocessing_num_proc,
                remove_columns=remove_cols,
                desc="Tokenizing eval",
            )

    data_collator = CausalLMCollator(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=train_cfg.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
