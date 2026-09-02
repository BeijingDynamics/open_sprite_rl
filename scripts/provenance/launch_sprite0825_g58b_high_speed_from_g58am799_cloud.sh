#!/usr/bin/env bash
set -euo pipefail

root=/root/sprite/sprite_isaaclab/IsaacLab
experiment="$root/logs/rsl_rl/sprite0825_stage2_g58_high_speed_recovery"
import_run="$experiment/g58am799_import"
source_model="$root/baselines/sprite0825_stage2_g58a_model799_physical_candidate/model_799.pt"
expected_sha=47202164047b044e3e524cb65134b583725204057cae91fe41047583ba08a66b
out=/root/gpufree-data/g58b_high_speed_from_g58am799_150
log="$out/train.log"
task=Isaac-Sprite0825-Stage2-AMP-G58HighSpeedRecovery100Hz-Robust-v0

actual_sha="$(sha256sum "$source_model" | awk '{print $1}')"
[[ "$actual_sha" == "$expected_sha" ]] || {
  echo "Checkpoint SHA mismatch: $actual_sha" >&2
  exit 4
}

mkdir -p "$import_run" "$out"
cp "$source_model" "$import_run/model_799.pt"
sha256sum "$source_model" "$import_run/model_799.pt" >"$out/source_sha256.txt"

cd "$root"
source /opt/conda/etc/profile.d/conda.sh
conda activate isaaclab
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=yes

exec ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task "$task" \
  --num_envs 8192 \
  --max_iterations 150 \
  --headless \
  --device cuda:0 \
  --resume \
  --load_run g58am799_import \
  --checkpoint model_799.pt \
  agent.run_name=stage2_G58B_high_speed_from_G58Am799_150_env8192 \
  agent.save_interval=25 \
  >"$log" 2>&1
