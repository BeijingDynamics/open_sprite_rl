#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITERATIONS="${MAX_ITERATIONS:-300}"
TASK="Isaac-Sprite0825-Stage2-AMP-G58FullMotorEnvelope100Hz-Robust-v0"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-$SPRITE_RL_ROOT/baselines/sprite0825_stage2_g57_model500_forward_candidate/model_500.pt}"
EXPECTED_SHA="c4bb13b271e840f6f9c182e5681c5ab7ebc1f48055406fb49bb1d6afda8e419d"
EXPERIMENT="$ISAACLAB_ROOT/logs/rsl_rl/sprite0825_stage2_g58_full_motor_envelope"
SEED_RUN="open_sprite_g57_model500"

verify_sha256 "$SOURCE_CHECKPOINT" "$EXPECTED_SHA"
mkdir -p "$EXPERIMENT/$SEED_RUN"
cp "$SOURCE_CHECKPOINT" "$EXPERIMENT/$SEED_RUN/model_500.pt"

cd "$ISAACLAB_ROOT"
"$ISAACLAB_SH" -p "$TRAIN_PY" \
  --task "$TASK" --num_envs "$NUM_ENVS" --max_iterations "$MAX_ITERATIONS" \
  --headless --device cuda:0 --resume --load_run "$SEED_RUN" --checkpoint model_500.pt \
  agent.run_name=stage2_G58A_full_motor_envelope_from_G57m500_300_env8192 \
  agent.save_interval=25
