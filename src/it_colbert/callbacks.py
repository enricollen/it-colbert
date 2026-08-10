"""training callbacks shared by both phases."""

from __future__ import annotations

import logging

from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

logger = logging.getLogger(__name__)


class EarlyStoppingCallback(TrainerCallback):
    """stop when the selection metric stops improving.

    written instead of using `transformers.EarlyStoppingCallback` because that one
    is an ExportableState: resuming from a checkpoint whose `trainer_state.json`
    predates the callback raises on load.

    note that early stopping truncates a warmup+decay schedule, leaving the model
    at a higher learning rate than a completed run would. it earns its place when
    a long distillation run can drift away from retrieval quality; it does not
    when a single epoch cannot overfit in the first place.
    """

    def __init__(
        self,
        early_stopping_patience: int = 3,
        early_stopping_threshold: float = 0.0,
    ):
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

        # runs before the trainer updates best_metric; same rule as hf's callback
        operator = (
            (lambda a, b: a > b) if args.greater_is_better else (lambda a, b: a < b)
        )
        if state.best_metric is None or (
            operator(metric_value, state.best_metric)
            and abs(metric_value - state.best_metric) > self.early_stopping_threshold
        ):
            self.early_stopping_patience_counter = 0
            logger.info(
                "early-stop: improve %s=%.6f (prev_best=%s) at step %s",
                metric_to_check,
                float(metric_value),
                state.best_metric,
                state.global_step,
            )
        else:
            self.early_stopping_patience_counter += 1
            logger.info(
                "early-stop: no improve (%s=%.6f, best=%.6f, patience %s/%s)",
                metric_to_check,
                float(metric_value),
                float(state.best_metric),
                self.early_stopping_patience_counter,
                self.early_stopping_patience,
            )
            if self.early_stopping_patience_counter >= self.early_stopping_patience:
                logger.info("early-stop: stopping training")
                control.should_training_stop = True
        return control
