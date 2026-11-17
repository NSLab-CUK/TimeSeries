#!/usr/bin/env bash
set -euo pipefail
task_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
project_dir="$(CDPATH= cd -- "$task_dir/.." && pwd)"
env_python="$project_dir/.venvs/jcou_revision_py311/bin/python"
if [[ ! -x "$env_python" ]]; then
  echo "Project environment missing. Run: bash $task_dir/setup_env.sh" >&2
  exit 1
fi
exec "$env_python" -u "$task_dir/run_4gpu.py" "$@"
