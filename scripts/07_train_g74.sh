#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-6000}"
TASK="Isaac-Sprite0825-Stage2-AMP-G74ShoulderRollCenter50Hz-Robust-v0"

cd "$ISAACLAB_ROOT"
exec "$ISAACLAB_SH" -p "$TRAIN_PY" \
  --task "$TASK" --num_envs "$NUM_ENVS" --max_iterations "$MAX_ITERATIONS" \
  --headless --device cuda:0 --seed 42
