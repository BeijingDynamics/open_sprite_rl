#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1000}"
RUN_NAME="${RUN_NAME:-stage2_G57_native_pm01_scratch_1000_env8192}"
TASK="Isaac-Sprite0825-Stage2-AMP-G57NativePM01Forward100Hz-Robust-v0"

echo "Training G57 from scratch: envs=$NUM_ENVS iterations=$MAX_ITERATIONS"
cd "$ISAACLAB_ROOT"
"$ISAACLAB_SH" -p "$TRAIN_PY" \
  --task "$TASK" \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERATIONS" \
  --headless \
  --device cuda:0 \
  agent.run_name="$RUN_NAME" \
  agent.save_interval=100
