#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

PLAY_PY="$ISAACLAB_ROOT/scripts/reinforcement_learning/rsl_rl/play.py"
case "${BASELINE:-g74}" in
  g74)
    TASK="${TASK:-Isaac-Sprite0825-Stage2-AMP-G74ShoulderRollCenter50Hz-v0}"
    DEFAULT_CHECKPOINT="$SPRITE_RL_ROOT/baselines/sprite0825_stage2_g74_model3000/model_3000.pt"
    CHECKPOINT_SHA=1b186bb10ee3cfa011d464e7c86b4f72b4c90a466c1576e3885cea5cd223f4f5
    ;;
  g60)
    TASK="${TASK:-Isaac-Sprite0825-Stage2-AMP-G60TimeNormalizedWaist50Hz-v0}"
    DEFAULT_CHECKPOINT="$SPRITE_RL_ROOT/baselines/sprite0825_stage2_g60_model3450_current/model_3450.pt"
    CHECKPOINT_SHA=6a1a80a2a2f7073698c0886133c325a46462ace6bbd3cf7de70246beafb85f15
    ;;
  *) echo "Unknown baseline: $BASELINE (use g74 or g60)" >&2; exit 2 ;;
esac
CHECKPOINT="${CHECKPOINT:-$DEFAULT_CHECKPOINT}"
NUM_ENVS="${NUM_ENVS:-1}"

require_file "$PLAY_PY"
if [[ "$CHECKPOINT" == "$DEFAULT_CHECKPOINT" ]]; then
  verify_sha256 "$CHECKPOINT" "$CHECKPOINT_SHA"
else
  require_file "$CHECKPOINT"
fi

cd "$ISAACLAB_ROOT"
exec "$ISAACLAB_SH" -p "$PLAY_PY" \
  --task "$TASK" \
  --num_envs "$NUM_ENVS" \
  --checkpoint "$CHECKPOINT" \
  --device cuda:0
