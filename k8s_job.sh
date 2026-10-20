#!/usr/bin/env bash
set -euo pipefail

# Capture initialization as well as training, before loading .env or installing Python.
if [[ "${_OPENPI_LOG_TRAIN:-0}" != 1 ]]; then
  job_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  bootstrap_python="${OPENPI_BOOTSTRAP_PYTHON:-}"
  if [[ -n "${bootstrap_python}" && ! -x "${bootstrap_python}" ]]; then
    bootstrap_python=""
  fi
  if [[ -z "${bootstrap_python}" ]]; then
    for candidate in python3 python /usr/local/bin/python3 /opt/conda/bin/python /usr/bin/python3; do
      if [[ "${candidate}" == */* ]]; then
        [[ -x "${candidate}" ]] || continue
        bootstrap_python="${candidate}"
      elif command -v "${candidate}" >/dev/null 2>&1; then
        bootstrap_python="$(command -v "${candidate}")"
      fi
      [[ -n "${bootstrap_python}" ]] && break
    done
  fi
  if [[ -n "${bootstrap_python}" ]]; then
    OPENPI_BOOTSTRAP_PYTHON="${bootstrap_python}" exec "${bootstrap_python}" "${job_root}/scripts/run_logged.py" train bash "${job_root}/k8s_job.sh" "$@"
  fi
  echo "No system Python is available for startup logging; continuing without the log wrapper." >&2
  export _OPENPI_LOG_TRAIN=1
fi

# Container/data/OSS setup is shared with manual runs.
# shellcheck source=init_cfg.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/init_cfg.sh"
[[ "${OPENPI_GPU_NUM:?Set OPENPI_GPU_NUM in .env}" =~ ^[1-9][0-9]*$ ]] || {
  echo "OPENPI_GPU_NUM must be a positive integer" >&2; exit 1;
}
for count in 2 4 8; do
  if [[ "${OPENPI_TRAIN_CONFIG:?}" == *"_80G_${count}gpu_full"* && "${OPENPI_GPU_NUM}" != "${count}" ]]; then
    echo "${OPENPI_TRAIN_CONFIG} requires ${count} GPUs; OPENPI_GPU_NUM=${OPENPI_GPU_NUM}" >&2
    exit 1
  fi
done
if [[ "${OPENPI_RESUME:-0}" == 1 && -z "${OPENPI_EXP_NAME:-}" ]]; then
  echo "Set OPENPI_EXP_NAME to the existing experiment when OPENPI_RESUME=1" >&2
  exit 1
fi
init_cfg_main
if [[ "${OPENPI_ENV_ONLY:-0}" == "1" ]]; then
  exit 0
fi
# init_env.sh runs as a child shell. Restore its user-space FFmpeg library
# path for the training process (system FFmpeg remains the default when it is
# available).
runtime_env="${PYTHON%/bin/python}/openpi-runtime.env"
if [[ -f "${runtime_env}" ]]; then
  # shellcheck disable=SC1090
  source "${runtime_env}"
fi
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_PLATFORMS=cuda
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 "$((${OPENPI_GPU_NUM:?} - 1))")}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

CONFIG_NAME="${OPENPI_TRAIN_CONFIG:?Set OPENPI_TRAIN_CONFIG in .env}"
EXP_NAME="${OPENPI_EXP_NAME:-${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S)}"
PROJECT_NAME="${OPENPI_PROJECT_NAME:-${CONFIG_NAME}}"
RUN_DIR="${CHECKPOINT_DIR}/${CONFIG_NAME}/${EXP_NAME}"
TRAIN_LOG="${CHECKPOINT_DIR}/${CONFIG_NAME}/${EXP_NAME}.train.log"
OSS_RUN_URI="${OSS_ROOT%/}/${PROJECT_NAME}/${EXP_NAME}"
UPLOADED_DIR="${OPENPI_UPLOAD_MARKERS_DIR:-/tmp/openpi-uploaded-checkpoints}/${CONFIG_NAME}/${EXP_NAME}"
mkdir -p "${CHECKPOINT_DIR}/${CONFIG_NAME}" "${UPLOADED_DIR}"

sync_training_log() {
  # 训练进程和 tee 退出后上传完整文件，不按 OSS 时间戳跳过最终版本。
  [[ -f "${TRAIN_LOG}" ]] || {
    echo "Training log missing; cannot upload ${OSS_RUN_URI}/train.log: ${TRAIN_LOG}" >&2
    return 1
  }
  echo "Uploading training log to ${OSS_RUN_URI}/train.log"
  ossutil_upload cp "${TRAIN_LOG}" "${OSS_RUN_URI}/train.log" -f || return $?
  echo "Training log saved: ${OSS_RUN_URI}/train.log"
}

sync_completed_checkpoints() {
  local checkpoint_path step marker
  [[ -d "${RUN_DIR}" ]] || return 0

  for checkpoint_path in "${RUN_DIR}"/*; do
    [[ -d "${checkpoint_path}" ]] || continue
    step="$(basename "${checkpoint_path}")"
    [[ "${step}" =~ ^[0-9]+$ ]] || continue
    marker="${UPLOADED_DIR}/${step}"
    # A resumed run may upload to a different OSS destination.
    [[ -f "${marker}" && "$(<"${marker}")" == "${OSS_RUN_URI}/${step}/" ]] && continue

    echo "Uploading completed checkpoint ${step} to ${OSS_RUN_URI}/${step}/"
    ossutil_upload cp -r "${checkpoint_path}/" "${OSS_RUN_URI}/${step}/" -j 4 --update || return $?
    printf '%s\n' "${OSS_RUN_URI}/${step}/" >"${marker}" || return $?
  done
}

upload_loop() {
  local sleep_pid=""
  # Finish an in-flight OSS command before the parent performs the final sync.
  trap - EXIT
  trap '' INT
  trap 'if [[ -n "${sleep_pid}" ]]; then kill "${sleep_pid}" 2>/dev/null || true; wait "${sleep_pid}" 2>/dev/null || true; fi; exit 0' TERM
  while true; do
    if ! sync_completed_checkpoints; then
      echo "Checkpoint upload failed; retrying in ${UPLOAD_INTERVAL_SECONDS}s" >&2
    fi
    sleep "${UPLOAD_INTERVAL_SECONDS}" &
    sleep_pid=$!
    wait "${sleep_pid}" || return $?
    sleep_pid=""
  done
}

uploader_pid=""
stop_uploader() {
  if [[ -n "${uploader_pid}" ]] && kill -0 "${uploader_pid}" 2>/dev/null; then
    kill "${uploader_pid}" 2>/dev/null || true
    wait "${uploader_pid}" 2>/dev/null || true
  fi
  uploader_pid=""
}
training_pid=""
stop_training() {
  if [[ -n "${training_pid}" ]]; then
    # The trainer, data-loader workers and tee share this private process group.
    kill -TERM -- "-${training_pid}" 2>/dev/null || true
    wait "${training_pid}" 2>/dev/null || true
    training_pid=""
  fi
}

# One exit path for success, training failure and unexpected errors.
cleanup() {
  local exit_status=$? upload_status=0
  trap - EXIT
  trap '' INT TERM
  stop_training
  # 优先保存含 loss/stderr 的完整日志，不依赖大 checkpoint 上传完成。
  sync_training_log || upload_status=$?
  stop_uploader
  sync_completed_checkpoints || upload_status=$?
  if (( exit_status != 0 )); then
    echo "Training/job failed (${exit_status}); final OSS sync status: ${upload_status}" >&2
  elif (( upload_status != 0 )); then
    echo "Training completed, but final OSS sync failed (${upload_status})" >&2
    exit_status=${upload_status}
  else
    echo "Training and OSS upload completed: ${OSS_RUN_URI}"
  fi
  exit "${exit_status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

echo "config=${CONFIG_NAME}"
echo "checkpoints=${RUN_DIR}"
echo "oss=${OSS_RUN_URI}"
echo "training_log=${TRAIN_LOG} -> ${OSS_RUN_URI}/train.log (uploaded on exit)"
upload_loop &
uploader_pid=$!

train_args=(--wandb-enabled)
[[ "${OPENPI_RESUME:-0}" != 1 ]] || train_args+=(--resume)

# Give the training pipeline its own process group so termination also stops
# its workers. Wait asynchronously so Bash handles INT/TERM immediately.
set -m
(
  set +m
  trap - EXIT INT TERM
  set +e
  "${PYTHON}" scripts/train.py \
    "${CONFIG_NAME}" \
    --exp-name "${EXP_NAME}" \
    --data.repo-id "${DATA_DIR}" \
    --data.assets.assets-dir "${ASSETS_DIR}" \
    --data.assets.asset-id assets \
    --weight-loader.params-path "${PARAMS_DIR}" \
    --checkpoint-base-dir "${CHECKPOINT_DIR}" \
    "${train_args[@]}" \
    2>&1 | tee -a "${TRAIN_LOG}"
  statuses=("${PIPESTATUS[@]}")
  # Preserve a training error; otherwise report a failed log write.
  exit "$(( statuses[0] != 0 ? statuses[0] : statuses[1] ))"
) &
training_pid=$!
set +m
set +e
wait "${training_pid}"
training_status=$?
training_pid=""
exit "${training_status}"
