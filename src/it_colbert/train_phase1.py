"""phase 1: supervised contrastive colbert training with cached in-batch negatives."""

from __future__ import annotations

import logging
from pathlib import Path

from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers.training_args import BatchSamplers

from pylate import evaluation, losses, utils

from it_colbert.config import Phase1Config
from it_colbert.data import build_phase1_dataset
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

    evaluator = evaluation.ColBERTTripletEvaluator(
        anchors=eval_ds["query"],
        positives=eval_ds["positive"],
        negatives=eval_ds["negative"],
        name="mmarco-it-wiki-eval",
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
        data_collator=utils.ColBERTCollator(model.tokenize),
    )

    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)

    final_dir = Path(cfg.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    # unwrap compiled module if needed
    to_save = model
    if hasattr(model, "_orig_mod"):
        to_save = model._orig_mod
    to_save.save_pretrained(str(final_dir))
    logger.info("phase 1 model saved to %s", final_dir)
    return final_dir
