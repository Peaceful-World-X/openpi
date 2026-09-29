#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

uv run --frozen --no-build-isolation --offline scripts/compute_norm_stats_ur7.py \
  --data-path "/16T-2/sudu/code/.data/bund_demo_data_0reset_0827_v002_v21_reconstructed" \
  --output-path "/16T-2/sudu/code/.data/bund_demo_data_0reset_0827_v002_v21_reconstructed/assets" \
  --action-horizon 50 \
  --action-dim 32 \
  --num-workers 8
