#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 SEED..." >&2
  exit 2
fi

source /opt/conda/etc/profile.d/conda.sh
conda activate isaaclab

ROOT=/root/sprite/sprite_isaaclab/IsaacLab
CHECKPOINT="$ROOT/logs/rsl_rl/sprite0825_stage2_g58_preserve925_gentle_yaw/2026-09-01_21-22-15_stage2_G58F_preserve925_gentle_yaw_400_env8192/model_1050.pt"
EXPECTED_SHA=deda0cc80023c00e6fd7723830bf3a9b9ceb4b33e72c9e1e446d9b0873f1d912
OUT=/root/gpufree-data/g58f_preserve925_gentle_yaw_400/model1050_supplemental
TASK=Isaac-Sprite0825-Stage2-AMP-G58Preserve925GentleYaw100Hz-v0

mkdir -p "$OUT"
actual_sha=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
[[ "$actual_sha" == "$EXPECTED_SHA" ]]
cd "$ROOT"
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=yes

for seed in "$@"; do
  ./isaaclab.sh -p evaluate_sprite0615_stage2_yaw.py \
    --task "$TASK" --checkpoint "$CHECKPOINT" --num_envs 160 \
    --target_vx 0.30 --duration 30 --warmup 5 --seed "$seed" \
    --output "$OUT/full_yaw_seed${seed}.json" \
    --headless --device cuda:0 >"$OUT/full_yaw_seed${seed}.log" 2>&1
done
