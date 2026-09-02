#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

PLAY_PY="$ISAACLAB_ROOT/scripts/reinforcement_learning/rsl_rl/play.py"
TASK="${TASK:-Isaac-Sprite0825-Stage2-AMP-G58Preserve925GentleYaw100Hz-v0}"
DEFAULT_CHECKPOINT="$SPRITE_RL_ROOT/baselines/sprite0825_stage2_g58f_model1050_stage2_qualified/model_1050.pt"
CHECKPOINT="${CHECKPOINT:-$DEFAULT_CHECKPOINT}"
NUM_ENVS="${NUM_ENVS:-1}"

require_file "$PLAY_PY"
if [[ "$CHECKPOINT" == "$DEFAULT_CHECKPOINT" ]]; then
  verify_sha256 "$CHECKPOINT" deda0cc80023c00e6fd7723830bf3a9b9ceb4b33e72c9e1e446d9b0873f1d912
else
  require_file "$CHECKPOINT"
fi

cd "$ISAACLAB_ROOT"
exec "$ISAACLAB_SH" -p "$PLAY_PY" \
  --task "$TASK" \
  --num_envs "$NUM_ENVS" \
  --checkpoint "$CHECKPOINT" \
  --device cuda:0
