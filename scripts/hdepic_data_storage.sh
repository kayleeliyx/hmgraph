#!/usr/bin/env bash
# Source before starting Python, CUDA, torchrun, or any HD-EPIC subprocess.
# Storage routing only: do not rewrite frozen model/config/checkpoint identities.
# Existing legacy /tmp Qwen paths resolve to /data; this file does not create,
# remove, or alter those aliases, model files, environments, or checkpoints.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf '%s\n' 'Source this file from a Bash launcher before starting Python.' >&2
  exit 2
fi

_hdepic_configure_data_storage() {
  local task_root=/data/yaxuanli/ActGround/.runtime/hdepic-storage
  local task_storage=/data/yaxuanli/qwen38-storage
  local task_assignment task_name task_path task_real
  local -a task_directories=(
    "HDEPIC_STORAGE_ROOT=$task_root"
    "TMPDIR=$task_root/tmp"
    "TMP=$task_root/tmp"
    "TEMP=$task_root/tmp"
    "XDG_CACHE_HOME=$task_root/cache"
    "XDG_CONFIG_HOME=$task_root/config"
    "XDG_DATA_HOME=$task_root/share"
    "XDG_RUNTIME_DIR=$task_root/xdg-runtime"
    "HF_HOME=$task_root/cache/huggingface"
    "HF_HUB_CACHE=$task_root/cache/huggingface/hub"
    "HUGGINGFACE_HUB_CACHE=$task_root/cache/huggingface/hub"
    "TRANSFORMERS_CACHE=$task_root/cache/huggingface/hub"
    "HF_DATASETS_CACHE=$task_root/cache/huggingface/datasets"
    "HF_MODULES_CACHE=$task_root/cache/huggingface/modules"
    "HF_ASSETS_CACHE=$task_root/cache/huggingface/assets"
    "HF_XET_CACHE=$task_root/cache/huggingface/xet"
    "PIP_CACHE_DIR=$task_root/cache/pip"
    "UV_CACHE_DIR=$task_root/cache/uv"
    "TORCH_HOME=$task_root/cache/torch"
    "TORCH_EXTENSIONS_DIR=$task_root/tmp/torch-extensions"
    "TORCHINDUCTOR_CACHE_DIR=$task_root/tmp/torchinductor"
    "TORCH_COMPILE_DEBUG_DIR=$task_root/tmp/torch-compile-debug"
    "TORCHELASTIC_LOG_DIR=$task_root/tmp/torchelastic"
    "PET_LOG_DIR=$task_root/tmp/torchelastic"
    "TRITON_CACHE_DIR=$task_root/tmp/triton/cache"
    "TRITON_DUMP_DIR=$task_root/tmp/triton/dump"
    "TRITON_OVERRIDE_DIR=$task_root/tmp/triton/override"
    "CUDA_CACHE_PATH=$task_root/tmp/cuda"
    "PYTORCH_KERNEL_CACHE_PATH=$task_root/tmp/pytorch-kernels"
    "NUMBA_CACHE_DIR=$task_root/tmp/numba"
    "PYTHONPYCACHEPREFIX=$task_root/cache/pycache"
    "MPLCONFIGDIR=$task_root/cache/matplotlib"
    "WANDB_DIR=$task_root/wandb"
    "WANDB_CACHE_DIR=$task_root/cache/wandb"
    "WANDB_CONFIG_DIR=$task_root/config/wandb"
    "WANDB_DATA_DIR=$task_root/share/wandb"
    "WANDB_ARTIFACT_DIR=$task_root/wandb/artifacts"
    "JUPYTER_RUNTIME_DIR=$task_root/tmp/jupyter"
    "IPYTHONDIR=$task_root/config/ipython"
    "PLAYWRIGHT_BROWSERS_PATH=$task_root/cache/playwright"
    "SE_CACHE_PATH=$task_root/cache/selenium"
  )
  local -a task_existing=(
    "HDEPIC_QWEN38_SFT_ENV=$task_storage/hdepic-qwen38-sft-env"
    "HDEPIC_QWEN38_BASELINE_ENV=$task_storage/hdepic-qwen38-27b-env"
    "HDEPIC_QWEN38_MODEL_CACHE=$task_storage/hdepic-qwen38-27b-hf"
    "HDEPIC_QWEN38_CHECKPOINT_ROOT=$task_storage/hdepic-qwen38-sft-checkpoints"
  )

  # realpath -m resolves existing symlink ancestors even for missing children.
  # Validate every destination before creating any runtime directory.
  for task_assignment in "${task_directories[@]}"; do
    task_path=${task_assignment#*=}
    task_real=$(realpath -m -- "$task_path") || return 1
    case "$task_real" in
      /data/*) ;;
      *) printf 'Refusing runtime storage outside /data: %s -> %s\n' "$task_path" "$task_real" >&2; return 1 ;;
    esac
  done
  # These are relocated existing resources, never installation destinations.
  for task_assignment in "${task_existing[@]}"; do
    task_path=${task_assignment#*=}
    task_real=$(realpath -e -- "$task_path") || return 1
    case "$task_real" in
      /data/*) ;;
      *) printf 'Refusing Qwen resource outside /data: %s -> %s\n' "$task_path" "$task_real" >&2; return 1 ;;
    esac
    [[ -d "$task_real" ]] || return 1
  done

  for task_assignment in "${task_directories[@]}"; do
    task_name=${task_assignment%%=*}
    task_path=${task_assignment#*=}
    # Private permissions for newly created temp/runtime/cache directories.
    (umask 077; mkdir -p -- "$task_path") || return 1
    task_real=$(realpath -e -- "$task_path") || return 1
    case "$task_real" in
      /data/*) ;;
      *) printf 'Runtime destination escaped /data: %s\n' "$task_path" >&2; return 1 ;;
    esac
    export "$task_name=$task_real"
  done
  for task_assignment in "${task_existing[@]}"; do
    task_name=${task_assignment%%=*}
    task_path=${task_assignment#*=}
    task_real=$(realpath -e -- "$task_path") || return 1
    export "$task_name=$task_real"
  done
  export HDEPIC_QWEN38_SFT_PYTHON="$HDEPIC_QWEN38_SFT_ENV/bin/python"
  export HDEPIC_QWEN38_BASELINE_PYTHON="$HDEPIC_QWEN38_BASELINE_ENV/bin/python"
  [[ -x "$HDEPIC_QWEN38_SFT_PYTHON" && -x "$HDEPIC_QWEN38_BASELINE_PYTHON" ]] || return 1

  export PYTHONDONTWRITEBYTECODE=1
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  export HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true
  # HOME remains untouched. All settings are inherited by subsequently launched
  # processes; already running interpreters may have cached their old temp dirs.
}

if _hdepic_configure_data_storage; then
  unset -f _hdepic_configure_data_storage
else
  unset -f _hdepic_configure_data_storage
  return 1
fi
