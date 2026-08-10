#!/usr/bin/env python3
# average pylate/colbert checkpoints (backbone + 1_dense) into a new folder
"""merge two or more ItColBERT checkpoints by uniform (or weighted) parameter average.

example:
  uv run python scripts/merge_checkpoints.py \\
    --inputs outputs/final outputs/final_fullkd \\
    --output outputs/final_merge_80k_fullkd \\
    --weights 0.5 0.5
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


WEIGHT_FILES = ("model.safetensors",)
DENSE_DIR = "1_Dense"
COPY_ALWAYS = (
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
)


def _average_safetensors(paths: list[Path], weights: list[float], out: Path) -> None:
    states = [load_file(str(p), device="cpu") for p in paths]
    keys = list(states[0].keys())
    for s in states[1:]:
        if set(s.keys()) != set(keys):
            missing = set(keys) ^ set(s.keys())
            raise SystemExit(f"key mismatch between {paths[0]} and tensors: {sorted(missing)[:8]}")
    merged: dict[str, torch.Tensor] = {}
    for k in keys:
        acc = None
        for st, w in zip(states, weights):
            t = st[k].float() * w
            acc = t if acc is None else acc + t
        assert acc is not None
        merged[k] = acc.to(states[0][k].dtype)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(out))


def merge_models(inputs: list[Path], output: Path, weights: list[float] | None) -> None:
    if len(inputs) < 2:
        raise SystemExit("need at least two --inputs")
    for p in inputs:
        if not (p / "model.safetensors").is_file():
            raise SystemExit(f"missing backbone weights: {p / 'model.safetensors'}")
        if not (p / DENSE_DIR / "model.safetensors").is_file():
            raise SystemExit(f"missing dense weights: {p / DENSE_DIR / 'model.safetensors'}")

    if weights is None:
        weights = [1.0 / len(inputs)] * len(inputs)
    if len(weights) != len(inputs):
        raise SystemExit("--weights length must match --inputs")
    s = float(sum(weights))
    if abs(s - 1.0) > 1e-5:
        weights = [w / s for w in weights]
        print(f"normalized weights -> {weights}", flush=True)

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    # copy non-weight metadata from first model
    src0 = inputs[0]
    for name in COPY_ALWAYS:
        src = src0 / name
        if src.is_file():
            shutil.copy2(src, output / name)
    dense_cfg = src0 / DENSE_DIR / "config.json"
    (output / DENSE_DIR).mkdir(parents=True, exist_ok=True)
    if dense_cfg.is_file():
        shutil.copy2(dense_cfg, output / DENSE_DIR / "config.json")

    print("merging backbone...", flush=True)
    _average_safetensors(
        [p / "model.safetensors" for p in inputs],
        weights,
        output / "model.safetensors",
    )
    print("merging 1_Dense...", flush=True)
    _average_safetensors(
        [p / DENSE_DIR / "model.safetensors" for p in inputs],
        weights,
        output / DENSE_DIR / "model.safetensors",
    )

    meta = {
        "merge": {
            "inputs": [str(p) for p in inputs],
            "weights": weights,
            "method": "uniform_parameter_average",
        }
    }
    (output / "merge_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    readme = (
        "# ItColBERT checkpoint merge\n\n"
        f"averaged: {', '.join(str(p) for p in inputs)}\n"
        f"weights: {weights}\n"
    )
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(f"wrote {output}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    args = ap.parse_args()
    merge_models(args.inputs, args.output, args.weights)


if __name__ == "__main__":
    main()
