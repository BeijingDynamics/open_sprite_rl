#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/sprite/sprite_isaaclab/IsaacLab
OUT=/root/gpufree-data/g57_native_pm01_scratch_1000
LOG="$OUT/train.log"
TASK=Isaac-Sprite0825-Stage2-AMP-G57NativePM01Forward100Hz-Robust-v0
RUN=stage2_G57_native_pm01_scratch_1000_env8192

mkdir -p "$OUT"
cd "$ROOT"
source /opt/conda/etc/profile.d/conda.sh
conda activate isaaclab
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=yes

exec ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task "$TASK" \
  --num_envs 8192 \
  --max_iterations 1000 \
  --headless \
  --device cuda:0 \
  agent.run_name="$RUN" \
  agent.save_interval=100 \
  >"$LOG" 2>&1
