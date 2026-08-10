"""evaluation helpers for italian colbert."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pylate import evaluation, models

from it_colbert.config import EvalConfig
from it_colbert.data import build_mmarco_eval_triplet

logger = logging.getLogger(__name__)


def run_eval(cfg: EvalConfig) -> dict:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("evaluating model at %s", cfg.model_name_or_path)

    model = models.ColBERT(
        model_name_or_path=cfg.model_name_or_path,
        document_length=cfg.document_length,
        query_length=cfg.query_length,
    )

    eval_ds = build_mmarco_eval_triplet(
        n_samples=cfg.mmarco_eval_samples,
        seed=cfg.seed,
    )

    evaluator = evaluation.ColBERTTripletEvaluator(
        anchors=eval_ds["query"],
        positives=eval_ds["positive"],
        negatives=eval_ds["negative"],
        name="mmarco-it-heldout",
    )
    metrics = evaluator(model)
    logger.info("metrics: %s", metrics)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.json"
    payload = {
        "model": cfg.model_name_or_path,
        "n_eval": len(eval_ds),
        "metrics": metrics if isinstance(metrics, dict) else {"score": metrics},
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("wrote %s", out_path)
    return payload
