"""load and normalize training configs from toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Phase1Config:
    model_name: str = "DeepMount00/Italian-ModernBERT-base"
    output_dir: str = "outputs/phase1"
    run_name: str = "it-colbert-phase1"
    dim: int = 128
    document_length: int = 256
    query_length: int = 32
    mmarco_samples: int = 2_000_000
    mmarco_eval_samples: int = 2_000
    include_wiki_hn: bool = True
    wiki_max_hard_negatives: int = 2
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 128
    per_device_eval_batch_size: int = 32
    mini_batch_size: int = 16
    learning_rate: float = 3e-6
    temperature: float = 0.02
    warmup_ratio: float = 0.05
    logging_steps: int = 50
    eval_steps: int = 1000
    save_steps: int = 1000
    save_total_limit: int = 2
    seed: int = 42
    bf16: bool = True
    fp16: bool = False
    compile_model: bool = False
    max_steps: int | None = None
    resume_from_checkpoint: str | None = None


@dataclass
class Phase2Config:
    model_name_or_path: str = "outputs/phase1/final"
    output_dir: str = "outputs/phase2"
    run_name: str = "it-colbert-phase2"
    document_length: int = 256
    query_length: int = 32
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 8
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.05
    logging_steps: int = 20
    save_steps: int = 500
    save_total_limit: int = 2
    seed: int = 42
    bf16: bool = True
    fp16: bool = False
    compile_model: bool = False
    max_steps: int | None = None
    # caps lighton kd rows (none = all). when mix is on, this is the lighton budget only.
    max_train_samples: int | None = None
    resume_from_checkpoint: str | None = None
    # option b: mix lighton kd with mmarco-it hard-negatives already scored by a ce teacher
    include_mmarco_hn: bool = False
    mmarco_hn_max_samples: int | None = None
    mmarco_hn_dataset: str = "hotchpotch/mmarco-hard-negatives-reranker-filtered"
    mmarco_hn_config: str = "italian-hard-negatives"
    kd_n_ways: int = 11
    # cheap overfitting check on mixed/lighton kd (0 = disabled)
    kd_eval_samples: int = 512
    # if false, eval may overlap train (needed for mid-run resume with same max_steps)
    kd_eval_exclude_from_train: bool = True
    eval_steps: int = 2000
    per_device_eval_batch_size: int = 32
    # stop if val kl does not improve for this many evals; 0 = monitor only
    early_stopping_patience: int = 3
    load_best_model_at_end: bool = True
    # kl lower is better
    metric_for_best_model: str = "kd-holdout_kl_divergence"


@dataclass
class EvalConfig:
    model_name_or_path: str = "outputs/phase2/final"
    output_dir: str = "outputs/eval"
    mmarco_eval_samples: int = 2000
    document_length: int = 256
    query_length: int = 32
    batch_size: int = 16
    seed: int = 42


@dataclass
class SmokeConfig:
    """short run overrides for validating the pipeline on gpu."""

    phase1: dict[str, Any] = field(default_factory=dict)
    phase2: dict[str, Any] = field(default_factory=dict)


def load_toml(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def phase1_from_toml(path: str | Path) -> Phase1Config:
    data = load_toml(path)
    section = data.get("phase1", data)
    known = {f.name for f in Phase1Config.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return Phase1Config(**{k: v for k, v in section.items() if k in known})


def phase2_from_toml(path: str | Path) -> Phase2Config:
    data = load_toml(path)
    section = data.get("phase2", data)
    known = {f.name for f in Phase2Config.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return Phase2Config(**{k: v for k, v in section.items() if k in known})


def eval_from_toml(path: str | Path) -> EvalConfig:
    data = load_toml(path)
    section = data.get("eval", data)
    known = {f.name for f in EvalConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return EvalConfig(**{k: v for k, v in section.items() if k in known})


def apply_overrides(cfg: Any, overrides: dict[str, Any]) -> Any:
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg
