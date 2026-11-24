#!/usr/bin/env bash
# Reproduce every result for one model, in order.
# Usage: scripts/run_all.sh <flux-schnell|flux-dev|sd3.5-medium> [out-dir]
# Every step is resume-safe: re-running skips images and classifications that exist.
set -euo pipefail

MODEL="${1:?usage: scripts/run_all.sh <flux-schnell|flux-dev|sd3.5-medium> [out-dir]}"
OUT="${2:-outputs}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run() { echo; echo "=== $1 ==="; python3 "$HERE/$1.py" --model "$MODEL" --out-dir "$OUT" "${@:2}"; }

run opsearch               # Sec. 4.4: pick one operating point per target type
run generate               # Sec. 4.1/4.3: baseline, ours, prompt-swap, negative-prompt images
run evaluate               # Sec. 4.2: LLaVA classifications + four-way audit
run tables                 # Tables 1-4
run figures                # qualitative grid + audit figures
run curves                 # Fig. 3 operating curves
run subspace_diagnostics   # Fig. 4 / App. E
run probes                 # App. F / G
if [[ "$MODEL" == flux-* ]]; then
  run redundancy_ablation --concept Towers --strength 1.5   # Fig. 5
else
  run redundancy_ablation --concept Dogs                    # Fig. 1
fi
echo; echo "done: results in $OUT/$MODEL/"
