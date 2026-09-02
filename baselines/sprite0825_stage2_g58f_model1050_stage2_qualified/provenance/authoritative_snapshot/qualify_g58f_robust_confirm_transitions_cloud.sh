#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate isaaclab

ROOT=/root/sprite/sprite_isaaclab/IsaacLab
CHECKPOINT="$ROOT/logs/rsl_rl/sprite0825_stage2_g58_preserve925_gentle_yaw/2026-09-01_21-22-15_stage2_G58F_preserve925_gentle_yaw_400_env8192/model_1050.pt"
EXPECTED_SHA=deda0cc80023c00e6fd7723830bf3a9b9ceb4b33e72c9e1e446d9b0873f1d912
OUT=/root/gpufree-data/g58f_preserve925_gentle_yaw_400/model1050_supplemental
CLEAN=Isaac-Sprite0825-Stage2-AMP-G58Preserve925GentleYaw100Hz-v0
ROBUST=Isaac-Sprite0825-Stage2-AMP-G58Preserve925GentleYaw100Hz-Robust-v0

mkdir -p "$OUT"
actual_sha=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
[[ "$actual_sha" == "$EXPECTED_SHA" ]]
cd "$ROOT"
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=yes

./isaaclab.sh -p evaluate_sprite0615_stage2.py \
  --task "$ROBUST" --checkpoint "$CHECKPOINT" --protocol yaw --duration 30 \
  --num_envs 320 --seed 44 --output "$OUT/seed_44_robust_yaw30_env320.json" \
  --headless --device cuda:0 >"$OUT/seed_44_robust_yaw30_env320.log" 2>&1

./isaaclab.sh -p evaluate_sprite0615_stage2.py \
  --task "$CLEAN" --checkpoint "$CHECKPOINT" --protocol transitions --cycles 20 \
  --num_envs 64 --seed 42 --output "$OUT/transitions20_seed42.json" \
  --headless --device cuda:0 >"$OUT/transitions20_seed42.log" 2>&1

