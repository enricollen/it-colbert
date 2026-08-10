#!/usr/bin/env bash
# install a torch2.6+cu124-compatible flash-attn wheel (optional; sdpa is default).
# usage: bash scripts/install_flash_attn.sh
set -euo pipefail
cd "$(dirname "$0")/.."

ABI=$(uv run python -c 'import torch; print("TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")')
WHEEL="flash_attn-2.7.4.post1+cu12torch2.6cxx11abi${ABI}-cp311-cp311-linux_x86_64.whl"
URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/${WHEEL}"
mkdir -p /tmp/wheels
if [[ ! -f "/tmp/wheels/$WHEEL" ]]; then
  wget -c --tries=8 --timeout=60 -O "/tmp/wheels/$WHEEL" "$URL"
fi
uv pip install "/tmp/wheels/$WHEEL"
uv run python -c 'import flash_attn; from transformers.utils import is_flash_attn_2_available; print(flash_attn.__version__, is_flash_attn_2_available())'
