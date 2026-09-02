#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate isaaclab

root=/root/sprite/sprite_isaaclab/IsaacLab
checkpoint="$root/logs/rsl_rl/sprite0825_stage2_g58_preserve925_gentle_yaw/2026-09-01_21-22-15_stage2_G58F_preserve925_gentle_yaw_400_env8192/model_1050.pt"
expected_sha=deda0cc80023c00e6fd7723830bf3a9b9ceb4b33e72c9e1e446d9b0873f1d912
task=Isaac-Sprite0825-Stage2-AMP-G58Preserve925GentleYaw100Hz-v0
reference=/root/sprite/sprite_isaaclab/assets/sprite0615_references/stage2_amp/sprite0825_pm01_native_straight_vx045_mapped_100hz_headingcanon_v1.npz
out=/root/gpufree-data/g58f_preserve925_gentle_yaw_400/model1050_postqual/motor_aligned

mkdir -p "$out"
[[ "$(sha256sum "$checkpoint" | awk '{print $1}')" == "$expected_sha" ]]
cd "$root"
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=yes

run_case() {
  local label="$1"
  local vx="$2"
  local yaw="$3"
  local json="$out/${label}.json"
  ./isaaclab.sh -p evaluate_sprite0615_stage2_style.py \
    --task "$task" --checkpoint "$checkpoint" --reference "$reference" \
    --reference-body-mode archive --mode velocity \
    --target-vx "$vx" --target-yaw-rate "$yaw" \
    --num-envs 64 --steps 3000 --warmup-steps 500 --sample-stride 5 \
    --cycle-frames 2061 --reference-inter-touchdown-s 0.259 --seed 42 \
    --output "$json" --headless --device cuda:0 \
    >"${json%.json}.log" 2>&1
}

(
  run_case vx015_yaw000 0.15 0.00
  run_case vx030_yawp040 0.30 0.40
) &
worker_a=$!

(
  run_case vx045_yaw000 0.45 0.00
  run_case vx030_yawm040 0.30 -0.40
) &
worker_b=$!

wait "$worker_a"
wait "$worker_b"

echo "G58F_MODEL1050_ALIGNED_MOTOR_DONE"
sha256sum "$out"/*.json
