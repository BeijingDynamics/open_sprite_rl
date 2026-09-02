#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate isaaclab

ROOT=/root/sprite/sprite_isaaclab/IsaacLab
EXPERIMENT=sprite0825_stage2_g58_preserve925_gentle_yaw
SEED_DIR="$ROOT/logs/rsl_rl/$EXPERIMENT/seed_g58b_model925"
SOURCE_CHECKPOINT="$ROOT/logs/rsl_rl/sprite0825_stage2_g58_high_speed_recovery/2026-08-31_20-12-21_stage2_G58B_high_speed_from_G58Am799_150_env8192/model_925.pt"
EXPECTED_SHA=e4d74619be6ea0786057ede910785bf58f8f92088f72c431578fdf56644523e9
JOB_DIR="$ROOT/logs/sprite0615_stage2_jobs"
LOG="$JOB_DIR/stage2_G58F_preserve925_gentle_yaw_400_env8192.log"
PID_FILE="$JOB_DIR/stage2_G58F_preserve925_gentle_yaw.pid"

mkdir -p "$SEED_DIR" "$JOB_DIR"
actual_sha=$(sha256sum "$SOURCE_CHECKPOINT" | awk '{print $1}')
if [[ "$actual_sha" != "$EXPECTED_SHA" ]]; then
  echo "Refusing to launch: model925 SHA mismatch: $actual_sha" >&2
  exit 1
fi
cp -f "$SOURCE_CHECKPOINT" "$SEED_DIR/model_925.pt"

cd "$ROOT"
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=yes

nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Sprite0825-Stage2-AMP-G58Preserve925GentleYaw100Hz-Robust-v0 \
  --num_envs 8192 \
  --max_iterations 400 \
  --run_name stage2_G58F_preserve925_gentle_yaw_400_env8192 \
  --resume \
  --load_run seed_g58b_model925 \
  --checkpoint model_925.pt \
  --headless \
  >"$LOG" 2>&1 &

pid=$!
echo "$pid" >"$PID_FILE"
echo "PID $pid"
echo "LOG $LOG"
