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
    # an already-IR-tuned checkpoint, not the raw MLM: the raw backbone scores
    # 0.004 nDCG@10 on MLDR and cannot be turned into a retriever in one pass
    model_name: str = "nickprock/Italian-ModernBERT-base-embed-mmarco-mnrl"
    output_dir: str = "outputs/phase1"
    run_name: str = "it-colbert-phase1"
    dim: int = 128
    # train at the length you intend to index at
    document_length: int = 512
    query_length: int = 32
    mmarco_samples: int = 1_500_000
    mmarco_eval_samples: int = 2_000
    include_wiki_hn: bool = True
    wiki_max_hard_negatives: int = 2
    # reranker-mined hard negatives; much harder than the official bm25 triples
    include_mmarco_hn: bool = True
    mmarco_hn_samples: int | None = 200_000
    mmarco_hn_negatives_per_query: int = 4
    # miracl-ita / squad-ita, to widen phase 1 past machine-translated mmarco
    include_italian_sources: bool = True
    italian_source_max_samples: int | None = 50_000
    # round-2 negatives from scripts/mine_hard_negatives.py
    mined_negatives_path: str | None = None
    mined_negatives_per_query: int = 4
    num_train_epochs: float = 1.0
    # CachedContrastive decouples these: the batch sets how many in-batch
    # negatives each query sees, mini_batch_size sets peak VRAM. keeping them
    # equal (the old default) pays for GradCache and gets nothing back.
    per_device_train_batch_size: int = 512
    per_device_eval_batch_size: int = 32
    mini_batch_size: int = 32
    learning_rate: float = 1e-5
    temperature: float = 0.02
    warmup_ratio: float = 0.05
    logging_steps: int = 10
    eval_steps: int = 250
    save_steps: int = 250
    save_total_limit: int = 3
    # retrieval metrics during phase 1: triplet accuracy against BM25-sampled
    # negatives saturates above ~0.95 within a few hundred steps and cannot tell
    # you whether more training would still help. nDCG/MRR can.
    ir_eval_enabled: bool = True
    ir_eval_name: str = "it-ir"
    ir_eval_mldr_queries: int = 150
    ir_eval_mldr_docs: int = 2_000
    ir_eval_mmarco_queries: int = 300
    ir_eval_mmarco_docs: int = 3_000
    ir_eval_mmarco_pool_docs: int = 50_000
    # deliberately no early stopping and no load_best_model_at_end here:
    #  - one epoch over ~2.4M triplets cannot overfit, so there is
    #    nothing for early stopping to protect against;
    #  - the schedule is warmup + linear decay, so stopping early leaves the
    #    model at a high learning rate, usually worse than the annealed end.
    # if the IR curve is still climbing at the last eval, train LONGER
    # (2 epochs or batch 1024) rather than stopping sooner.
    early_stopping_patience: int = 0
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "it-ir_score"
    seed: int = 42
    bf16: bool = True
    fp16: bool = False
    compile_model: bool = False
    max_steps: int | None = None
    resume_from_checkpoint: str | None = None
    # pick up the newest complete checkpoint automatically, so re-running the
    # same command after an overnight stop continues instead of starting over
    auto_resume: bool = True
    # built dataset cache; joining the mmarco tsvs indexes 8.8M passages, which
    # would otherwise be re-paid on every resume
    dataset_cache_dir: str | None = "outputs/dataset_cache"


@dataclass
class Phase2Config:
    model_name_or_path: str = "outputs/phase1/final"
    output_dir: str = "outputs/phase2"
    run_name: str = "it-colbert-phase2"
    document_length: int = 512
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
    auto_resume: bool = True
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
    ir_eval_enabled: bool = True
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
