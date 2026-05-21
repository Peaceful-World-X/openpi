#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda
export LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH:-}

# ==================================================================================================================

name=${name:-}
config_name=${config_name:-}
policy_dir=${policy_dir:-}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
BASE_PORT=${BASE_PORT:-8000}
gpu_list=(${CUDA_VISIBLE_DEVICES//,/ })
NUM_PORTS=${NUM_PORTS:-${#gpu_list[@]}}
SERVICES_PER_GPU=${SERVICES_PER_GPU:-$(((NUM_PORTS + ${#gpu_list[@]} - 1) / ${#gpu_list[@]}))}

time_str=$(date +%Y%m%d_%H%M)
LOG_DIR=/mnt/pfs/users/wenyao.xue/code/.logs/server/openpi_${time_str}
mkdir -p "${LOG_DIR}"

# ==================================================================================================================

if [ -z "${mem_fraction:-}" ]; then
  gpu_mem_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "${gpu_list[0]}" | head -n 1 | tr -d ' ')
  if [ "${gpu_mem_mib}" -ge 70000 ]; then
    mem_fraction=0.12
  elif [ "${gpu_mem_mib}" -ge 20000 ]; then
    mem_fraction=0.35
  else
    mem_fraction=0.90
  fi
fi

idx=0
pids=()
last_log_file=""
last_port=""

cleanup() {
  trap - EXIT INT TERM
  echo
  echo "Stopping services..."

  if [ "${#pids[@]}" -gt 0 ]; then
    kill "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

echo "name=${name}"
echo "config_name=${config_name}"
echo "policy_dir=${policy_dir}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NUM_PORTS=${NUM_PORTS}"
echo "SERVICES_PER_GPU=${SERVICES_PER_GPU}"
echo "BASE_PORT=${BASE_PORT}"
echo "mem_fraction=${mem_fraction}"

for gpu in "${gpu_list[@]}"; do
  for replica in $(seq 1 "${SERVICES_PER_GPU}"); do
    if [ "${idx}" -ge "${NUM_PORTS}" ]; then
      break 2
    fi

    port=$((BASE_PORT + idx))
    log_file="${LOG_DIR}/${name}_gpu${gpu}_replica${replica}_port${port}.log"
    echo "[${idx}] GPU ${gpu} replica ${replica}/${SERVICES_PER_GPU} at port ${port}, logging to ${log_file}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    XLA_PYTHON_CLIENT_MEM_FRACTION="${mem_fraction}" \
    uv run --offline scripts/serve_policy.py \
      --port "${port}" \
      policy:checkpoint \
      --policy.config="${config_name}" \
      --policy.dir="${policy_dir}" \
      > "${log_file}" 2>&1 &

    pids+=("$!")

    last_log_file="${log_file}"
    last_port="${port}"

    idx=$((idx + 1))
  done
done

if [ -z "${last_log_file}" ]; then
  echo "No service started."
  exit 1
fi
sleep 1

echo
echo "All services started."
echo "Log directory: ${LOG_DIR}"
echo "Showing logs for last port ${last_port}: ${last_log_file}"
echo

tail -n 200 -F "${last_log_file}"
