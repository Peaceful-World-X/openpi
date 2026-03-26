#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export LEROBOT_IMPL=fast_norm
export LEROBOT_WORKSPACE="$ROOT_DIR/lerobot_dataset"
export HF_DATASETS_CACHE=/shared_disk/users/yang.wang/cache/hf_datasets_cache
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH

export HF_LEROBOT_HOME=/mnt/pfs/users/yang.wang/data

# print config  repo_id
uv run python - <<'PY'
import openpi.training.config_fast_norm as cfg
c = cfg.get_config("pi05_aloha_piper_fold_shirt_exp1001_norm")
print("repo_id =", c.data.repo_id)
print("asset_id =", c.data.assets.asset_id)
print("assets_dir =", c.data.assets.assets_dir)
PY

# compute norm
JAX_PLATFORM_NAME=cpu uv run scripts/compute_norm_stats.py --config-name pi05_aloha_piper_fold_shirt_exp1001_norm
