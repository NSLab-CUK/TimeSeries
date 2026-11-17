#!/usr/bin/env bash
set -euo pipefail
task_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
project_dir="$(CDPATH= cd -- "$task_dir/.." && pwd)"
env_dir="$project_dir/.venvs/jcou_revision_py311"
tools_dir="$project_dir/.venvs/jcou_tools"
mkdir -p "$tools_dir"
exec > >(tee -a "$tools_dir/setup.log") 2>&1
echo "[SETUP] Project: $project_dir"
echo "[SETUP] Environment: $env_dir"
if [[ ! -x "$tools_dir/uv" ]]; then
  echo '[SETUP] Installing project-local uv from the official Astral installer'
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --silent --show-error https://astral.sh/uv/install.sh -o "$tools_dir/install_uv.sh"
  else
    wget -O "$tools_dir/install_uv.sh" https://astral.sh/uv/install.sh
  fi
  UV_INSTALL_DIR="$tools_dir" UV_NO_MODIFY_PATH=1 sh "$tools_dir/install_uv.sh"
fi
export UV_PYTHON_INSTALL_DIR="$tools_dir/python"
export UV_CACHE_DIR="$tools_dir/cache"
"$tools_dir/uv" python install 3.11
if [[ ! -x "$env_dir/bin/python" ]]; then
  "$tools_dir/uv" venv --python 3.11 "$env_dir"
fi
"$env_dir/bin/python" -c 'import sys; assert sys.version_info[:2] == (3,11), sys.version'
echo '[SETUP] Installing CUDA PyTorch 2.5.1 and fixed data dependencies'
"$tools_dir/uv" pip install --python "$env_dir/bin/python" 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cu124
"$tools_dir/uv" pip install --python "$env_dir/bin/python" -r "$project_dir/requirements/training.txt"
"$tools_dir/uv" pip check --python "$env_dir/bin/python"
"$tools_dir/uv" pip freeze --python "$env_dir/bin/python" > "$tools_dir/requirements_installed.txt"
GPU_IDS="${GPU_IDS:-0,1,2,3}" "$env_dir/bin/python" -u "$task_dir/check_env.py" "$tools_dir/environment_check.json"
echo '[SETUP PASS] Environment installed and selected GPUs checked. Full training was not started.'
echo "[RUN] GPU_IDS=0,1,2,3 bash $task_dir/run_4gpu.sh --resume"
