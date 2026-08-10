"""phase 2: knowledge distillation from cross-encoder scores into colbert."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from datasets import DatasetDict
from sentence_transformers import SentenceTransformerTrainer
from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from pylate import evaluation, losses, utils

from it_colbert.config import Phase2Config
from it_colbert.data import build_phase1_dataset, load_kd_italian_train_eval
from it_colbert.ir_eval import CombinedEvaluator, ItIREvaluator
from it_colbert.model_utils import (
    build_colbert,
    enable_cuda_fast_kernels,
    make_training_args,
)

logger = logging.getLogger(__name__)


class KdEarlyStoppingCallback(TrainerCallback):
    """early stop on val metric without ExportableState.

    hf EarlyStoppingCallback is ExportableState and crashes on resume when the
    old checkpoint's trainer_state.json lacks that callback key.
    """

    def __init__(self, early_stopping_patience: int = 3, early_stopping_threshold: float = 0.0):
        self.early_stopping_patience = int(early_stopping_patience)
        self.early_stopping_threshold = float(early_stopping_threshold)
        self.early_stopping_patience_counter = 0

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict,
        **kwargs,
    ) -> TrainerControl:
        metric_to_check = args.metric_for_best_model
        if metric_to_check is None:
            return control
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)
        if metric_value is None:
            logger.warning("early stopping: missing metric %s", metric_to_check)
            return control

        # runs before trainer updates best_metric; same rule as hf EarlyStoppingCallback
        operator = (lambda a, b: a > b) if args.greater_is_better else (lambda a, b: a < b)
        if state.best_metric is None or (
            operator(metric_value, state.best_metric)
            and abs(metric_value - state.best_metric) > self.early_stopping_threshold
        ):
            self.early_stopping_patience_counter = 0
            logger.info(
                "kd early-stop: improve %s=%.6f (prev_best=%s) at step %s",
                metric_to_check,
                float(metric_value),
                state.best_metric,
                state.global_step,
            )
        else:
            self.early_stopping_patience_counter += 1
            logger.info(
                "kd early-stop: no improve (%s=%.6f, best=%.6f, patience %s/%s)",
                metric_to_check,
                float(metric_value),
                float(state.best_metric),
                self.early_stopping_patience_counter,
                self.early_stopping_patience,
            )
            if self.early_stopping_patience_counter >= self.early_stopping_patience:
                logger.info("kd early-stop: stopping training")
                control.should_training_stop = True
        return control


def run_phase2(cfg: Phase2Config) -> Path:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    enable_cuda_fast_kernels()
    logger.info("starting phase 2 distillation training")
    logger.info("config: %s", cfg)

    # mid-run resume: keep full train size so max_steps/scheduler match the original job
    exclude_eval = bool(cfg.kd_eval_exclude_from_train)
    if cfg.resume_from_checkpoint and exclude_eval and int(cfg.kd_eval_samples or 0) > 0:
        logger.warning(
            "resume + kd_eval: keeping full train set (eval may overlap) so step counts match"
        )
        exclude_eval = False

    train_ds, eval_ds = load_kd_italian_train_eval(
        max_samples=cfg.max_train_samples,
        eval_samples=cfg.kd_eval_samples,
        seed=cfg.seed,
        n_ways=cfg.kd_n_ways,
        exclude_eval_from_train=exclude_eval,
        include_mmarco_hn=cfg.include_mmarco_hn,
        mmarco_hn_max_samples=cfg.mmarco_hn_max_samples,
        mmarco_hn_dataset=cfg.mmarco_hn_dataset,
        mmarco_hn_config=cfg.mmarco_hn_config,
        mmarco_hn_score_temperature=cfg.mmarco_hn_score_temperature,
        mmarco_hn_score_clip=cfg.mmarco_hn_score_clip,
        split_sampling=cfg.kd_split_sampling,
        split_min_share=cfg.kd_split_min_share,
        split_quotas=cfg.kd_split_quotas,
    )
    if cfg.include_mmarco_hn:
        logger.info(
            "phase2 kd mix: lighton_cap=%s + mmarco_hn_cap=%s (%s[%s])",
            cfg.max_train_samples,
            cfg.mmarco_hn_max_samples,
            cfg.mmarco_hn_dataset,
            cfg.mmarco_hn_config,
        )

    model = build_colbert(
        model_name_or_path=cfg.model_name_or_path,
        document_length=cfg.document_length,
        query_length=cfg.query_length,
        compile_model=cfg.compile_model,
    )

    train_loss: Any = losses.Distillation(model=model)
    train_data: Any = train_ds

    # anti-forgetting replay: keep a contrastive stream on mmarco next to the kd
    # stream so long kd runs cannot drift off the short-passage distribution
    if cfg.contrastive_anchor_enabled:
        anchor_ds, _ = build_phase1_dataset(
            mmarco_samples=cfg.contrastive_anchor_samples,
            include_wiki_hn=False,
            eval_samples=0,
            seed=cfg.seed,
        )
        train_data = DatasetDict({"kd": train_ds, "contrastive": anchor_ds})
        train_loss = {
            "kd": train_loss,
            "contrastive": losses.CachedContrastive(
                model=model,
                mini_batch_size=cfg.contrastive_anchor_mini_batch_size,
                temperature=cfg.contrastive_anchor_temperature,
            ),
        }
        logger.info(
            "phase2 contrastive anchor on: %s replay triplets alongside %s kd rows",
            len(anchor_ds),
            len(train_ds),
        )

    # eval: hold-out KL says the student copies the teacher; the IR evaluator says
    # it actually retrieves better. select on the latter when it is enabled.
    evaluators: list[Any] = []
    callbacks = []
    kl_eval = eval_ds is not None and len(eval_ds) > 0
    if kl_eval:
        kd_evaluator = evaluation.ColBERTDistillationEvaluator(
            queries=list(eval_ds["query"]),
            documents=list(eval_ds["documents"]),
            scores=list(eval_ds["scores"]),
            name="kd-holdout",
            batch_size=cfg.per_device_eval_batch_size,
            show_progress_bar=False,
            write_csv=True,
        )
        # pylate evaluator leaves primary_metric=None; st 5.x prefix_name_to_metrics crashes
        kd_evaluator.primary_metric = "kl_divergence"
        evaluators.append(kd_evaluator)
        logger.info(
            "phase2 kd val enabled: %s examples every %s steps (kl, exclude_from_train=%s)",
            len(eval_ds),
            cfg.eval_steps,
            exclude_eval,
        )

    metric_for_best_model = cfg.metric_for_best_model
    greater_is_better = cfg.greater_is_better
    if cfg.ir_eval_enabled:
        ir_evaluator = ItIREvaluator(
            name=cfg.ir_eval_name,
            mldr_queries=cfg.ir_eval_mldr_queries,
            mldr_docs=cfg.ir_eval_mldr_docs,
            mmarco_queries=cfg.ir_eval_mmarco_queries,
            mmarco_docs=cfg.ir_eval_mmarco_docs,
            mmarco_pool_docs=cfg.ir_eval_mmarco_pool_docs,
            batch_size=cfg.per_device_eval_batch_size,
            seed=cfg.seed,
            weights=tuple(cfg.ir_eval_weights),
        )
        evaluators.append(ir_evaluator)
        metric_for_best_model = ir_evaluator.primary_metric
        greater_is_better = True
        logger.info("phase2 selecting checkpoints on %s (higher is better)", metric_for_best_model)
    elif greater_is_better is None:
        greater_is_better = False  # kl

    do_eval = bool(evaluators)
    evaluator: Any = None
    if len(evaluators) == 1:
        evaluator = evaluators[0]
    elif evaluators:
        evaluator = CombinedEvaluator(evaluators, primary_metric=metric_for_best_model)

    if do_eval and int(cfg.early_stopping_patience or 0) > 0:
        callbacks.append(
            KdEarlyStoppingCallback(
                early_stopping_patience=int(cfg.early_stopping_patience)
            )
        )
        logger.info(
            "early stopping on %s (patience=%s, greater_is_better=%s)",
            metric_for_best_model,
            cfg.early_stopping_patience,
            greater_is_better,
        )

    # keep more ckpts when selecting by val kl
    save_total_limit = cfg.save_total_limit
    if do_eval and cfg.load_best_model_at_end:
        save_total_limit = max(save_total_limit, 5)

    args = make_training_args(
        output_dir=cfg.output_dir,
        run_name=cfg.run_name,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        eval_steps=cfg.eval_steps if do_eval else None,
        save_steps=cfg.save_steps,
        save_total_limit=save_total_limit,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        seed=cfg.seed,
        max_steps=cfg.max_steps,
        eval_strategy="steps" if do_eval else "no",
        load_best_model_at_end=bool(do_eval and cfg.load_best_model_at_end),
        metric_for_best_model=metric_for_best_model if do_eval else None,
        greater_is_better=greater_is_better if do_eval else None,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        loss=train_loss,
        evaluator=evaluator,
        callbacks=callbacks or None,
        data_collator=utils.ColBERTCollator(tokenize_fn=model.tokenize),
    )

    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)

    final_dir = Path(cfg.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    to_save = model
    if hasattr(model, "_orig_mod"):
        to_save = model._orig_mod
    to_save.save_pretrained(str(final_dir))
    if do_eval and cfg.load_best_model_at_end:
        logger.info(
            "phase 2 final is the best-val checkpoint (metric=%s, best=%s, step=%s)",
            metric_for_best_model,
            getattr(trainer.state, "best_metric", None),
            getattr(trainer.state, "best_global_step", None),
        )
    logger.info("phase 2 model saved to %s", final_dir)
    return final_dir
