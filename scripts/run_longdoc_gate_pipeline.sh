#!/bin/bash
# unattended TODO.md §13.3 step 3-4 + §13.4 gate, kicked off 2026-08-25 while
# arm B (document_length=1024) phase-1 training was still running.
#
# 1. waits for the arm B training process to exit
# 2. benchmarks BOTH arms (512 and 1024), unchunked (mldr-it/mmarco-it/miracl-ita)
#    and chunked (mldr-it only), exactly per §13.3
# 3. prints held-out (report-half) summaries for each
# 4. runs the §13.4 gate: paired bootstrap of arm B vs arm A on the report half
#
# everything is appended to outputs/logs/longdoc_pipeline.log. safe to launch
# detached (nohup + disown) since every step here is read-only against the two
# arms' already-finished checkpoints -- it writes new outputs/benchmark_longdoc_*
# directories and touches nothing else.

set -uo pipefail
cd "$(dirname "$0")/.."

echo "=== pipeline started $(date -Iseconds) ==="
echo "waiting for arm B (phase1_longdoc_1024) training process to exit..."

while pgrep -f "it-colbert-phase1 --config configs/phase1_longdoc_1024.toml" > /dev/null; do
  sleep 20
done
echo "arm B process no longer running at $(date -Iseconds)"

if [ ! -d outputs/phase1_longdoc_1024/final ]; then
  echo "WARNING: outputs/phase1_longdoc_1024/final not found -- training may have"
  echo "crashed instead of finishing. Aborting the benchmark stage."
  echo "=== pipeline aborted $(date -Iseconds) ==="
  exit 1
fi

echo
echo "=== §13.3 step 3: benchmarking both arms ==="
for arm in 512 1024; do
  echo
  echo "--- arm ${arm}: unchunked mldr-it / mmarco-it / miracl-ita, started $(date -Iseconds) ---"
  NUMBA_CACHE_DIR="" uv run python scripts/run_benchmark.py \
    --output-dir "outputs/benchmark_longdoc_${arm}" \
    --benchmarks mldr-it mmarco-it miracl-ita \
    --models-only-extra --extra-colbert "longdoc-${arm}=outputs/phase1_longdoc_${arm}/final" \
    --colbert-doc-length "${arm}"

  echo
  echo "--- arm ${arm}: chunked mldr-it (2000/200 chars), started $(date -Iseconds) ---"
  NUMBA_CACHE_DIR="" uv run python scripts/run_benchmark.py \
    --output-dir "outputs/benchmark_longdoc_${arm}_chunked" \
    --benchmarks mldr-it \
    --models-only-extra --extra-colbert "longdoc-${arm} chunked=outputs/phase1_longdoc_${arm}/final" \
    --colbert-doc-length "${arm}" --chunk-chars 2000 --chunk-overlap-chars 200
done

echo
echo "=== §13.3 step 4: held-out (report-half) summaries ==="
for arm in 512 1024; do
  echo
  echo "--- arm ${arm} unchunked, mldr-it ---"
  uv run python scripts/report_query_half.py --benchmark-dir "outputs/benchmark_longdoc_${arm}" \
    --half report --benchmark mldr-it --metric ndcg@10
  echo "--- arm ${arm} chunked, mldr-it ---"
  uv run python scripts/report_query_half.py --benchmark-dir "outputs/benchmark_longdoc_${arm}_chunked" \
    --half report --benchmark mldr-it --metric ndcg@10
  echo "--- arm ${arm} unchunked, mmarco-it (guardrail) ---"
  uv run python scripts/report_query_half.py --benchmark-dir "outputs/benchmark_longdoc_${arm}" \
    --half report --benchmark mmarco-it --metric mrr@10
  echo "--- arm ${arm} unchunked, miracl-ita (guardrail) ---"
  uv run python scripts/report_query_half.py --benchmark-dir "outputs/benchmark_longdoc_${arm}" \
    --half report --benchmark miracl-ita --metric ndcg@10
done

echo
echo "=== §13.4 gate: arm B vs arm A, paired bootstrap ==="
uv run python scripts/compare_longdoc_arms.py

echo
echo "=== pipeline finished $(date -Iseconds) ==="
