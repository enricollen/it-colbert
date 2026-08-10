"""load and normalize training configs from toml."""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
    # reranker-mined hard negatives; much harder than the official bm25 triples
    include_mmarco_hn: bool = False
    mmarco_hn_samples: int | None = None
    mmarco_hn_negatives_per_query: int = 4
    # miracl-ita / squad-ita, to widen phase 1 past machine-translated mmarco
    include_italian_sources: bool = False
    italian_source_max_samples: int | None = None
    # round-2 negatives from scripts/mine_hard_negatives.py
    mined_negatives_path: str | None = None
    mined_negatives_per_query: int = 4
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
    # bge probs -> logits span ~[-13.8, 13.8]; divide to match lighton label sharpness
    mmarco_hn_score_temperature: float = 1.0
    mmarco_hn_score_clip: float | None = None
    kd_n_ways: int = 11
    # how max_train_samples is spent across lighton kd splits:
    # "proportional" (every split, sized by row count) or "sequential" (legacy: drain in order)
    kd_split_sampling: str = "proportional"
    kd_split_min_share: float = 0.02
    # explicit per-split row counts; overrides kd_split_sampling when set
    kd_split_quotas: dict[str, int] | None = None
    # cheap overfitting check on mixed/lighton kd (0 = disabled)
    kd_eval_samples: int = 512
    # if false, eval may overlap train (needed for mid-run resume with same max_steps)
    kd_eval_exclude_from_train: bool = True
    eval_steps: int = 2000
    per_device_eval_batch_size: int = 32
    # stop if val kl does not improve for this many evals; 0 = monitor only
    early_stopping_patience: int = 3
    load_best_model_at_end: bool = True
    # kl lower is better; overridden automatically when ir_eval_enabled
    metric_for_best_model: str = "kd-holdout_kl_divergence"
    greater_is_better: bool | None = None
    # select checkpoints on real retrieval metrics instead of hold-out kl
    ir_eval_enabled: bool = False
    ir_eval_name: str = "it-ir"
    ir_eval_mldr_queries: int = 200
    ir_eval_mldr_docs: int = 3_000
    ir_eval_mmarco_queries: int = 500
    ir_eval_mmarco_docs: int = 5_000
    ir_eval_mmarco_pool_docs: int = 50_000
    ir_eval_weights: tuple[float, float] = (0.5, 0.5)
    # anti-forgetting: replay phase-1 contrastive alongside kd (experimental)
    contrastive_anchor_enabled: bool = False
    contrastive_anchor_samples: int = 100_000
    contrastive_anchor_mini_batch_size: int = 16
    contrastive_anchor_temperature: float = 0.02


def load_toml(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _build(cls: Any, section: dict[str, Any], path: str | Path) -> Any:
    """instantiate a config dataclass, warning about keys it does not recognize.

    unknown keys used to be dropped silently, so a typo in a toml turned into a
    run with the default value and no way to tell from the logs.
    """
    known = {f.name for f in cls.__dataclass_fields__.values()}
    unknown = sorted(set(section) - known)
    if unknown:
        logger.warning("%s: ignoring unknown config keys %s", path, unknown)
    return cls(**{k: v for k, v in section.items() if k in known})


def phase1_from_toml(path: str | Path) -> Phase1Config:
    data = load_toml(path)
    return _build(Phase1Config, data.get("phase1", data), path)


def phase2_from_toml(path: str | Path) -> Phase2Config:
    data = load_toml(path)
    return _build(Phase2Config, data.get("phase2", data), path)


def apply_overrides(cfg: Any, overrides: dict[str, Any]) -> Any:
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg
