#!/usr/bin/env python3
"""optional push of the final colbert checkpoint to the hugging face hub."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from pylate import models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="outputs/final",
        help="local path to the trained colbert model",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="hub repo id, e.g. username/Italian-ModernBERT-ColBERT",
    )
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    username = os.environ.get("HF_USERNAME", "").strip()
    repo_id = args.repo_id
    if not repo_id:
        if not username:
            raise SystemExit(
                "set --repo-id or HF_USERNAME (and HF_TOKEN) to push to the hub"
            )
        repo_id = f"{username}/Italian-ModernBERT-ColBERT"

    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is required to push")

    path = Path(args.model)
    if not path.exists():
        raise SystemExit(f"model path not found: {path}")

    model = models.ColBERT(model_name_or_path=str(path))
    model.push_to_hub(repo_id, private=args.private)
    print(f"pushed to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
