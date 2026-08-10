# prefetch only needed weight files for remaining baselines
from __future__ import annotations

import time

from huggingface_hub import snapshot_download

SPECS: list[tuple[str, list[str]]] = [
    (
        "BAAI/bge-m3",
        ["*.json", "*.txt", "*.model", "pytorch_model.bin", "sentencepiece*"],
    ),
    (
        "intfloat/multilingual-e5-large",
        ["*.json", "*.txt", "model.safetensors", "sentencepiece*", "tokenizer*"],
    ),
    (
        "jinaai/jina-colbert-v2",
        [
            "*.json",
            "*.txt",
            "*.model",
            "model.safetensors",
            "pytorch_model.bin",
            "tokenizer*",
            "sentencepiece*",
        ],
    ),
    (
        "DeepMount00/Ita-Search",
        [
            "*.json",
            "*.txt",
            "*.model",
            "model.safetensors",
            "pytorch_model.bin",
            "tokenizer*",
            "sentencepiece*",
        ],
    ),
]


def main() -> None:
    for mid, patterns in SPECS:
        print(f"START {mid}", flush=True)
        t0 = time.time()
        try:
            path = snapshot_download(mid, allow_patterns=patterns, max_workers=4)
            print(f"DONE {mid} in {time.time() - t0:.0f}s -> {path}", flush=True)
        except Exception as e:
            print(f"FAIL {mid}: {type(e).__name__}: {e}", flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
