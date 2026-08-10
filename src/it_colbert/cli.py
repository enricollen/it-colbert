"""cli entrypoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from it_colbert.config import (
    apply_overrides,
    eval_from_toml,
    load_toml,
    phase1_from_toml,
    phase2_from_toml,
)
from it_colbert.benchmark.run import BenchmarkConfig, run_benchmark
from it_colbert.evaluate import run_eval
from it_colbert.infer import run_infer
from it_colbert.train_phase1 import run_phase1
from it_colbert.train_phase2 import run_phase2


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def phase1(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="phase 1 contrastive training")
    parser.add_argument(
        "--config",
        default=str(_repo_root() / "configs/phase1_contrastive.toml"),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="apply smoke.toml overrides for a short gpu validation run",
    )
    args = parser.parse_args(argv)
    cfg = phase1_from_toml(args.config)
    if args.smoke:
        smoke = load_toml(_repo_root() / "configs/smoke.toml")
        cfg = apply_overrides(cfg, smoke.get("phase1", {}))
    run_phase1(cfg)


def phase2(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="phase 2 distillation training")
    parser.add_argument(
        "--config",
        default=str(_repo_root() / "configs/phase2_distill.toml"),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--resume",
        default=None,
        help="resume from a trainer checkpoint directory",
    )
    args = parser.parse_args(argv)
    cfg = phase2_from_toml(args.config)
    if args.smoke:
        smoke = load_toml(_repo_root() / "configs/smoke.toml")
        cfg = apply_overrides(cfg, smoke.get("phase2", {}))
    if args.resume:
        cfg.resume_from_checkpoint = args.resume
    run_phase2(cfg)


def evaluate(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="evaluate italian colbert")
    parser.add_argument(
        "--config",
        default=str(_repo_root() / "configs/eval.toml"),
    )
    parser.add_argument("--model", default=None, help="override model path")
    args = parser.parse_args(argv)
    cfg = eval_from_toml(args.config)
    if args.model:
        cfg.model_name_or_path = args.model
    run_eval(cfg)


def infer(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="italian colbert demo")
    parser.add_argument("--model", default="outputs/phase2/final")
    parser.add_argument("--query", default="Qual è la capitale d'Italia?")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args(argv)
    run_infer(args.model, args.query, top_k=args.top_k)


def benchmark(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="italian ir sota comparison")
    parser.add_argument("--output-dir", default="outputs/benchmark")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["mldr-it", "mmarco-it"],
        choices=["mldr-it", "mmarco-it"],
    )
    parser.add_argument("--mmarco-max-corpus-docs", type=int, default=200_000)
    parser.add_argument("--mmarco-max-queries", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--dense-batch-size", type=int, default=64)
    parser.add_argument("--colbert-batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    cfg = BenchmarkConfig(
        output_dir=args.output_dir,
        index_root=str(Path(args.output_dir) / "indexes"),
        benchmarks=list(args.benchmarks),
        mmarco_max_corpus_docs=args.mmarco_max_corpus_docs,
        mmarco_max_queries=args.mmarco_max_queries,
        top_k=args.top_k,
        dense_batch_size=args.dense_batch_size,
        colbert_batch_size=args.colbert_batch_size,
        only_models=args.only,
    )
    run_benchmark(cfg)
