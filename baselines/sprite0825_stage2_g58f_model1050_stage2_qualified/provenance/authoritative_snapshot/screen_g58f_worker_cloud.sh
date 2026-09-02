#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 ITERATION..." >&2
  exit 2
fi

source /opt/conda/etc/profile.d/conda.sh
conda activate isaaclab

ROOT=/root/sprite/sprite_isaaclab/IsaacLab
RUN="$ROOT/logs/rsl_rl/sprite0825_stage2_g58_preserve925_gentle_yaw/2026-09-01_21-22-15_stage2_G58F_preserve925_gentle_yaw_400_env8192"
OUT=/root/gpufree-data/g58f_preserve925_gentle_yaw_400/screen
TASK=Isaac-Sprite0825-Stage2-AMP-G58Preserve925GentleYaw100Hz-v0
REFERENCE=/root/sprite/sprite_isaaclab/assets/sprite0615_references/stage2_amp/sprite0825_pm01_native_straight_vx045_mapped_100hz_headingcanon_v1.npz

cd "$ROOT"
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=yes

for iteration in "$@"; do
  checkpoint="$RUN/model_${iteration}.pt"
  test -f "$checkpoint"

  yaw="$OUT/model_${iteration}_yaw20.json"
  if [[ ! -s "$yaw" ]]; then
    ./isaaclab.sh -p evaluate_sprite0615_stage2.py \
      --task "$TASK" --checkpoint "$checkpoint" --protocol yaw --duration 20 \
      --num_envs 96 --seed 42 --output "$yaw" \
      --headless --device cuda:0 >"${yaw%.json}.log" 2>&1
  fi

  grid="$OUT/model_${iteration}_grid128.json"
  if [[ ! -s "$grid" ]]; then
    ./isaaclab.sh -p evaluate_sprite0615_stage2.py \
      --task "$TASK" --checkpoint "$checkpoint" --protocol grid \
      --num_envs 128 --seed 42 --output "$grid" \
      --headless --device cuda:0 >"${grid%.json}.log" 2>&1
  fi

  style="$OUT/model_${iteration}_style_vx030.json"
  if [[ ! -s "$style" ]]; then
    ./isaaclab.sh -p evaluate_sprite0615_stage2_style.py \
      --task "$TASK" --checkpoint "$checkpoint" --reference "$REFERENCE" \
      --reference-body-mode archive --mode velocity --target-vx 0.30 \
      --num-envs 64 --steps 2000 --warmup-steps 400 --sample-stride 5 \
      --cycle-frames 2061 --reference-inter-touchdown-s 0.259 --seed 42 \
      --output "$style" \
      --headless --device cuda:0 >"${style%.json}.log" 2>&1
  fi

  /opt/conda/envs/isaaclab/bin/python compare_sprite0615_stage2_style.py \
    --baseline "$OUT/model_925_style_vx030.json" \
    --candidate "$style" \
    --output "$OUT/model_${iteration}_vs_925_style_gates.json" \
    >"$OUT/model_${iteration}_vs_925_style_gates.log" 2>&1
done
