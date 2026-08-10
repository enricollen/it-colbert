#!/usr/bin/env bash
# resume large baseline weights into hf hub blob cache, then link via hub api
set -u
LOGDIR=/home/enricollen/PROJECTS/it-colbert/outputs/logs
VENV_PY=/home/enricollen/PROJECTS/it-colbert/.venv/bin/python
mkdir -p "$LOGDIR"

link_file() {
  local repo="$1" file="$2"
  # unset deprecated hf_transfer flag; prefer xet when available
  env -u HF_HUB_ENABLE_HF_TRANSFER HF_XET_HIGH_PERFORMANCE=1 \
    "$VENV_PY" -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('$repo','$file'))"
}

download_one() {
  local repo="$1" file="$2" sha="$3" expected="$4" dir="$5"
  mkdir -p "$dir"
  local final="$dir/$sha"
  if [ -f "$final" ] && [ "$(stat -c%s "$final")" -eq "$expected" ]; then
    echo "[$(date -Is)] skip blob exists $repo/$file"
    link_file "$repo" "$file" || true
    return 0
  fi
  local inc
  inc=$(ls -1S "$dir"/$sha.*.incomplete 2>/dev/null | head -1 || true)
  [ -n "${inc:-}" ] || inc="$dir/${sha}.wget.incomplete"
  echo "[$(date -Is)] START $repo/$file have=$(stat -c%s "$inc" 2>/dev/null || echo 0)/$expected"
  # keep retrying until size matches
  while true; do
    wget -c --timeout=120 --tries=0 --retry-connrefused --progress=dot:mega \
      -O "$inc" "https://huggingface.co/$repo/resolve/main/$file" \
      >>"$LOGDIR/wget_${repo//\//_}.log" 2>&1 || true
    local sz
    sz=$(stat -c%s "$inc" 2>/dev/null || echo 0)
    echo "[$(date -Is)] progress $repo/$file size=$sz"
    if [ "$sz" -eq "$expected" ]; then
      mv -f "$inc" "$final"
      echo "[$(date -Is)] DONE blob $final"
      link_file "$repo" "$file"
      return 0
    fi
    echo "[$(date -Is)] retrying $repo/$file after short pause"
    sleep 5
  done
}

# priority: smaller / higher-value first
download_one jinaai/jina-colbert-v2 model.safetensors \
  741387d37db6027ada13c4705b8bb32719a09dc71873f54814d1182cd8943806 \
  1119027888 \
  /home/enricollen/.cache/huggingface/hub/models--jinaai--jina-colbert-v2/blobs

download_one intfloat/multilingual-e5-large model.safetensors \
  020afdebf2762b29fcaf286629a96c3b3b65af241f6a08226b1cfee60a21def6 \
  2239611368 \
  /home/enricollen/.cache/huggingface/hub/models--intfloat--multilingual-e5-large/blobs

download_one BAAI/bge-m3 pytorch_model.bin \
  b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38 \
  2271145830 \
  /home/enricollen/.cache/huggingface/hub/models--BAAI--bge-m3/blobs

download_one DeepMount00/Ita-Search model.safetensors \
  ce7eeb7cb3ff8a6407eb5498e034b6f15bbe23c1d6f2c5b5f3d5c783f574577d \
  2383139480 \
  /home/enricollen/.cache/huggingface/hub/models--DeepMount00--Ita-Search/blobs

echo "[$(date -Is)] ALL_WEIGHTS_DONE"

# run remaining benchmark models
cd /home/enricollen/PROJECTS/it-colbert
unset HF_HUB_ENABLE_HF_TRANSFER
export HF_XET_HIGH_PERFORMANCE=1
for only in jina-colbert-v2 multilingual-e5-large bge-m3 Ita-Search; do
  echo "[$(date -Is)] BENCH $only"
  uv run python scripts/run_benchmark.py \
    --benchmarks mldr-it mmarco-it \
    --mmarco-max-corpus-docs 100000 \
    --only "$only" \
    --top-k 100 \
    --output-dir outputs/benchmark \
    >>"$LOGDIR/benchmark_remaining.log" 2>&1 || echo "[$(date -Is)] BENCH_FAIL $only"
done
echo "[$(date -Is)] ALL_BENCH_DONE"
