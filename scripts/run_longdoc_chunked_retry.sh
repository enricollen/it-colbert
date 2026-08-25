#!/bin/bash
# retry of §13.3's chunked step with the §11.2 fix (--colbert-brute-force-limit 70000)
# that the documented §13.3 command omits, which OOM'd both arms via the broken
# PLAID->Voyager ANN fallback on 64780 chunks > the default 25000 threshold.
set -uo pipefail
cd "$(dirname "$0")/.."

echo "=== chunked retry started $(date -Iseconds) ==="
for arm in 512 1024; do
  echo
  echo "--- arm ${arm}: chunked mldr-it (brute-force fix), started $(date -Iseconds) ---"
  NUMBA_CACHE_DIR="" uv run python scripts/run_benchmark.py \
    --output-dir "outputs/benchmark_longdoc_${arm}_chunked" \
    --benchmarks mldr-it \
    --models-only-extra --extra-colbert "longdoc-${arm} chunked=outputs/phase1_longdoc_${arm}/final" \
    --colbert-doc-length "${arm}" --chunk-chars 2000 --chunk-overlap-chars 200 \
    --colbert-brute-force-limit 70000
done

echo
echo "=== held-out chunked summaries ==="
for arm in 512 1024; do
  uv run python scripts/report_query_half.py --benchmark-dir "outputs/benchmark_longdoc_${arm}_chunked" \
    --half report --benchmark mldr-it --metric ndcg@10
done

echo
echo "=== §13.4 gate, re-run with chunked data now available ==="
uv run python scripts/compare_longdoc_arms.py

echo
echo "=== chunked retry finished $(date -Iseconds) ==="
