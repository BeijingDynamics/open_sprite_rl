#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITERATIONS="${MAX_ITERATIONS:-600}"
TASK="Isaac-Sprite0825-Stage2-AMP-G60TimeNormalizedWaist50Hz-Robust-v0"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-$SPRITE_RL_ROOT/baselines/sprite0825_stage2_g59_model2999_native50_parent/model_2999.pt}"
EXPECTED_SHA="eb84e5aedb8ace647d877788a98c86db6ce9670e4c003890e6e1d2ef21205a98"
EXPERIMENT="$ISAACLAB_ROOT/logs/rsl_rl/sprite0825_stage2_g60_time_normalized_waist_50hz"
SEED_RUN="open_sprite_g59_model2999"

verify_sha256 "$SOURCE_CHECKPOINT" "$EXPECTED_SHA"
mkdir -p "$EXPERIMENT/$SEED_RUN"
cp "$SOURCE_CHECKPOINT" "$EXPERIMENT/$SEED_RUN/model_2999.pt"

cd "$ISAACLAB_ROOT"
"$ISAACLAB_SH" -p "$TRAIN_PY" \
  --task "$TASK" --num_envs "$NUM_ENVS" --max_iterations "$MAX_ITERATIONS" \
  --headless --device cuda:0 --resume --load_run "$SEED_RUN" --checkpoint model_2999.pt \
  agent.run_name=stage2_G60_time_normalized_waist_from_G59m2999 \
  agent.save_interval=50
