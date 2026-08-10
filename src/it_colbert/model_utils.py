"""shared helpers for building pylate colbert models and trainers."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformerTrainingArguments

from pylate import models

logger = logging.getLogger(__name__)


def bf16_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def enable_cuda_fast_kernels() -> None:
    """enable tf32 / cudnn benchmark for faster matmul on ampere+ gpus."""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def resolve_attn_implementation(requested: str | None = "auto") -> str | None:
    """pick an attention backend.

    note: flash-attn-2 helps long sequences / vram, but on doc180 kd it was
    slower than pytorch sdpa on a 3090 (~1.11 vs ~1.3 it/s). default to sdpa.
    """
    if requested is not None and requested != "auto":
        return requested
    return "sdpa"


def build_colbert(
    model_name_or_path: str,
    document_length: int = 256,
    query_length: int = 32,
    dim: int = 128,
    compile_model: bool = False,
    attn_implementation: str | None = "auto",
) -> models.ColBERT:
    """initialize a colbert model from a base encoder or an existing colbert checkpoint."""
    kwargs: dict = {
        "model_name_or_path": model_name_or_path,
        "document_length": document_length,
        "query_length": query_length,
    }
    impl = resolve_attn_implementation(attn_implementation)
    if impl:
        model_kwargs: dict = {"attn_implementation": impl}
        # flash-attn-2 requires fp16/bf16 weights at load time
        if impl == "flash_attention_2" and bf16_available():
            model_kwargs["dtype"] = torch.bfloat16
        kwargs["model_kwargs"] = model_kwargs
        logger.info("loading colbert with attn_implementation=%s", impl)

    # dim is used when converting a non-colbert backbone
    try:
        model = models.ColBERT(**kwargs, embedding_size=dim)
    except TypeError:
        try:
            model = models.ColBERT(**kwargs, output_dim=dim)
        except TypeError:
            model = models.ColBERT(**kwargs)

    if compile_model and hasattr(torch, "compile"):
        logger.info("compiling model with torch.compile")
        model = torch.compile(model)
    return model


def make_training_args(
    *,
    output_dir: str,
    run_name: str,
    num_train_epochs: float,
    per_device_train_batch_size: int,
    learning_rate: float,
    warmup_ratio: float = 0.05,
    logging_steps: int = 50,
    eval_steps: int | None = None,
    save_steps: int = 1000,
    save_total_limit: int = 2,
    per_device_eval_batch_size: int | None = None,
    bf16: bool = True,
    fp16: bool = False,
    seed: int = 42,
    max_steps: int | None = None,
    eval_strategy: str = "no",
    load_best_model_at_end: bool = False,
    metric_for_best_model: str | None = None,
    greater_is_better: bool | None = None,
) -> SentenceTransformerTrainingArguments:
    use_bf16 = bool(bf16 and bf16_available())
    use_fp16 = bool(fp16 and not use_bf16 and torch.cuda.is_available())
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size
        or per_device_train_batch_size,
        learning_rate=learning_rate,
        # transformers v5: warmup_steps accepts a float ratio
        warmup_steps=float(warmup_ratio),
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        bf16=use_bf16,
        fp16=use_fp16,
        seed=seed,
        # 0 = main-process loading; avoids rare wsl dataloader worker hangs mid-epoch
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        report_to="none",
        # tee/nohup makes tqdm emit one newline per step (~100k lines); loss still logs via logging_steps
        disable_tqdm=True,
    )
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    if eval_steps is not None and eval_strategy != "no":
        kwargs["eval_strategy"] = eval_strategy
        kwargs["eval_steps"] = eval_steps
        kwargs["save_strategy"] = "steps"
        # load_best requires save when we evaluate; align intervals
        if load_best_model_at_end:
            kwargs["save_steps"] = eval_steps
            kwargs["load_best_model_at_end"] = True
            if metric_for_best_model:
                kwargs["metric_for_best_model"] = metric_for_best_model
            if greater_is_better is not None:
                kwargs["greater_is_better"] = greater_is_better
    else:
        kwargs["eval_strategy"] = "no"
        kwargs["save_strategy"] = "steps"

    return SentenceTransformerTrainingArguments(**kwargs)
