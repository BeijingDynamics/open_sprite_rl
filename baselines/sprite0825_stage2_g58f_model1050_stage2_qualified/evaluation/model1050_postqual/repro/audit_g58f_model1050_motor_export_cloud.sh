#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate isaaclab

root=/root/sprite/sprite_isaaclab/IsaacLab
run="$root/logs/rsl_rl/sprite0825_stage2_g58_preserve925_gentle_yaw/2026-09-01_21-22-15_stage2_G58F_preserve925_gentle_yaw_400_env8192"
checkpoint="$run/model_1050.pt"
expected_sha=deda0cc80023c00e6fd7723830bf3a9b9ceb4b33e72c9e1e446d9b0873f1d912
task=Isaac-Sprite0825-Stage2-AMP-G58Preserve925GentleYaw100Hz-v0
reference=/root/sprite/sprite_isaaclab/assets/sprite0615_references/stage2_amp/sprite0825_pm01_native_straight_vx045_mapped_100hz_headingcanon_v1.npz
asset_urdf=/root/sprite/sprite_isaaclab/assets/sprite0825_sanitized_v4/sprite0825_float.urdf
out=/root/gpufree-data/g58f_preserve925_gentle_yaw_400/model1050_postqual

mkdir -p "$out/motor" "$out/export"
actual_sha="$(sha256sum "$checkpoint" | awk '{print $1}')"
[[ "$actual_sha" == "$expected_sha" ]]
cd "$root"
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=yes

run_motor_case() {
  local label="$1"
  local vx="$2"
  local yaw="$3"
  local json="$out/motor/${label}.json"
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
  run_motor_case vx015_yaw000 0.15 0.00
  run_motor_case vx030_yawp040 0.30 0.40
) &
worker_a=$!

(
  run_motor_case vx045_yaw000 0.45 0.00
  run_motor_case vx030_yawm040 0.30 -0.40
) &
worker_b=$!

./isaaclab.sh -p export_sprite0825_stage2_deploy.py \
  --task "$task" --checkpoint "$checkpoint" \
  --output-dir "$out/export" --asset-urdf "$asset_urdf" \
  --label sprite0825_stage2_g58f_model1050_provisional \
  --expected-observation-dim 1488 --num-envs 1 \
  --headless --device cuda:0 >"$out/export.log" 2>&1

wait "$worker_a"
wait "$worker_b"

sha256sum "$checkpoint" "$out/export/policy.onnx" "$out/export/policy.pt" \
  "$out/export/contract.json" >"$out/SHA256SUMS"
echo "G58F_MODEL1050_MOTOR_EXPORT_DONE"
cat "$out/SHA256SUMS"
