#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITERATIONS="${MAX_ITERATIONS:-400}"
TASK="Isaac-Sprite0825-Stage2-AMP-G58Preserve925GentleYaw100Hz-Robust-v0"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-$SPRITE_RL_ROOT/baselines/sprite0825_stage2_g58b_model925_speed_candidate/model_925.pt}"
EXPECTED_SHA="e4d74619be6ea0786057ede910785bf58f8f92088f72c431578fdf56644523e9"
EXPERIMENT="$ISAACLAB_ROOT/logs/rsl_rl/sprite0825_stage2_g58_preserve925_gentle_yaw"
SEED_RUN="open_sprite_g58b_model925"

verify_sha256 "$SOURCE_CHECKPOINT" "$EXPECTED_SHA"
mkdir -p "$EXPERIMENT/$SEED_RUN"
cp "$SOURCE_CHECKPOINT" "$EXPERIMENT/$SEED_RUN/model_925.pt"

cd "$ISAACLAB_ROOT"
"$ISAACLAB_SH" -p "$TRAIN_PY" \
  --task "$TASK" --num_envs "$NUM_ENVS" --max_iterations "$MAX_ITERATIONS" \
  --headless --device cuda:0 --resume --load_run "$SEED_RUN" --checkpoint model_925.pt \
  agent.run_name=stage2_G58F_preserve925_gentle_yaw_400_env8192 \
  agent.save_interval=25
