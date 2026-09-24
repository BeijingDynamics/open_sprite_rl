#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

PLAY_PY="$ISAACLAB_ROOT/scripts/reinforcement_learning/rsl_rl/play.py"
TASK="${TASK:-Isaac-Sprite0825-Stage2-AMP-G60TimeNormalizedWaist50Hz-v0}"
DEFAULT_CHECKPOINT="$SPRITE_RL_ROOT/baselines/sprite0825_stage2_g60_model3450_current/model_3450.pt"
CHECKPOINT="${CHECKPOINT:-$DEFAULT_CHECKPOINT}"
NUM_ENVS="${NUM_ENVS:-1}"

require_file "$PLAY_PY"
if [[ "$CHECKPOINT" == "$DEFAULT_CHECKPOINT" ]]; then
  verify_sha256 "$CHECKPOINT" 6a1a80a2a2f7073698c0886133c325a46462ace6bbd3cf7de70246beafb85f15
else
  require_file "$CHECKPOINT"
fi

cd "$ISAACLAB_ROOT"
exec "$ISAACLAB_SH" -p "$PLAY_PY" \
  --task "$TASK" \
  --num_envs "$NUM_ENVS" \
  --checkpoint "$CHECKPOINT" \
  --device cuda:0
