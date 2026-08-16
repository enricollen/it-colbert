"""phase 1: supervised contrastive colbert training with cached in-batch negatives."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers.training_args import BatchSamplers

from pylate import evaluation, losses, utils

from it_colbert.config import Phase1Config
from it_colbert.data import build_phase1_dataset
from it_colbert.ir_eval import CombinedEvaluator, ItIREvaluator
from it_colbert.callbacks import EarlyStoppingCallback
from it_colbert.checkpoints import resolve_resume
from it_colbert.model_utils import (
    build_colbert,
    enable_cuda_fast_kernels,
    make_training_args,
)

logger = logging.getLogger(__name__)


def run_phase1(cfg: Phase1Config) -> Path:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    enable_cuda_fast_kernels()
    logger.info("starting phase 1 contrastive training")
    logger.info("config: %s", cfg)

    train_ds, eval_ds = build_phase1_dataset(
        mmarco_samples=cfg.mmarco_samples,
        include_wiki_hn=cfg.include_wiki_hn,
        wiki_max_hard_negatives=cfg.wiki_max_hard_negatives,
        eval_samples=cfg.mmarco_eval_samples,
        seed=cfg.seed,
        include_mmarco_hn=cfg.include_mmarco_hn,
        mmarco_hn_samples=cfg.mmarco_hn_samples,
        mmarco_hn_negatives_per_query=cfg.mmarco_hn_negatives_per_query,
        include_italian_sources=cfg.include_italian_sources,
        italian_source_max_samples=cfg.italian_source_max_samples,
        mined_negatives_path=cfg.mined_negatives_path,
        mined_negatives_per_query=cfg.mined_negatives_per_query,
        cache_dir=cfg.dataset_cache_dir,
    )

    model = build_colbert(
        model_name_or_path=cfg.model_name,
        document_length=cfg.document_length,
        query_length=cfg.query_length,
        dim=cfg.dim,
        compile_model=cfg.compile_model,
    )

    train_loss = losses.CachedContrastive(
        model=model,
        mini_batch_size=cfg.mini_batch_size,
        temperature=cfg.temperature,
    )

    # triplet accuracy is a cheap divergence check, not a quality signal: against
    # BM25-sampled negatives it saturates above ~0.95 within a few hundred steps.
    evaluators: list[Any] = [
        evaluation.ColBERTTripletEvaluator(
            anchors=eval_ds["query"],
            positives=eval_ds["positive"],
            negatives=eval_ds["negative"],
            name="mmarco-it-wiki-eval",
        )
    ]
    metric_for_best_model = None
    if cfg.ir_eval_enabled:
        # real retrieval metrics during phase 1, so you can see whether the curve
        # has flattened before deciding to train longer
        ir_evaluator = ItIREvaluator(
            name=cfg.ir_eval_name,
            mldr_queries=cfg.ir_eval_mldr_queries,
            mldr_docs=cfg.ir_eval_mldr_docs,
            mmarco_queries=cfg.ir_eval_mmarco_queries,
            mmarco_docs=cfg.ir_eval_mmarco_docs,
            mmarco_pool_docs=cfg.ir_eval_mmarco_pool_docs,
            batch_size=cfg.per_device_eval_batch_size,
            seed=cfg.seed,
            query_half=cfg.ir_eval_query_half,
        )
        evaluators.append(ir_evaluator)
        metric_for_best_model = ir_evaluator.primary_metric
        logger.info("phase1 ir eval on, every %s steps", cfg.eval_steps)

    evaluator: Any = (
        evaluators[0]
        if len(evaluators) == 1
        else CombinedEvaluator(evaluators, primary_metric=metric_for_best_model)
    )

    callbacks = []
    if cfg.load_best_model_at_end and int(cfg.early_stopping_patience or 0) > 0:
        # off by default; see the reasoning on Phase1Config
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=int(cfg.early_stopping_patience)
            )
        )
        logger.warning(
            "phase1 early stopping is on: it truncates the linear-decay schedule, "
            "so the kept checkpoint may sit at a high learning rate"
        )

    args = make_training_args(
        output_dir=cfg.output_dir,
        run_name=cfg.run_name,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        seed=cfg.seed,
        max_steps=cfg.max_steps,
        eval_strategy="steps",
        load_best_model_at_end=bool(cfg.load_best_model_at_end),
        metric_for_best_model=metric_for_best_model if cfg.load_best_model_at_end else None,
        greater_is_better=True if cfg.load_best_model_at_end else None,
    )
    # avoid duplicate queries/docs colliding as false negatives in-batch
    args.batch_sampler = BatchSamplers.NO_DUPLICATES

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        loss=train_loss,
        evaluator=evaluator,
        callbacks=callbacks or None,
        data_collator=utils.ColBERTCollator(model.tokenize),
    )

    resume_from = resolve_resume(
        cfg.output_dir, explicit=cfg.resume_from_checkpoint, auto=cfg.auto_resume
    )
    trainer.train(resume_from_checkpoint=resume_from)

    final_dir = Path(cfg.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    # unwrap compiled module if needed
    to_save = model
    if hasattr(model, "_orig_mod"):
        to_save = model._orig_mod
    to_save.save_pretrained(str(final_dir))
    logger.info("phase 1 model saved to %s", final_dir)
    return final_dir
