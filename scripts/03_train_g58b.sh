#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITERATIONS="${MAX_ITERATIONS:-150}"
TASK="Isaac-Sprite0825-Stage2-AMP-G58HighSpeedRecovery100Hz-Robust-v0"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-$SPRITE_RL_ROOT/baselines/sprite0825_stage2_g58a_model799_physical_candidate/model_799.pt}"
EXPECTED_SHA="47202164047b044e3e524cb65134b583725204057cae91fe41047583ba08a66b"
EXPERIMENT="$ISAACLAB_ROOT/logs/rsl_rl/sprite0825_stage2_g58_high_speed_recovery"
SEED_RUN="open_sprite_g58a_model799"

verify_sha256 "$SOURCE_CHECKPOINT" "$EXPECTED_SHA"
mkdir -p "$EXPERIMENT/$SEED_RUN"
cp "$SOURCE_CHECKPOINT" "$EXPERIMENT/$SEED_RUN/model_799.pt"

cd "$ISAACLAB_ROOT"
"$ISAACLAB_SH" -p "$TRAIN_PY" \
  --task "$TASK" --num_envs "$NUM_ENVS" --max_iterations "$MAX_ITERATIONS" \
  --headless --device cuda:0 --resume --load_run "$SEED_RUN" --checkpoint model_799.pt \
  agent.run_name=stage2_G58B_high_speed_from_G58Am799_150_env8192 \
  agent.save_interval=25
