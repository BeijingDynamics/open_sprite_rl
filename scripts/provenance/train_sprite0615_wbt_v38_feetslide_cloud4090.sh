#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate isaaclab
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=yes

root=/root/sprite/sprite_isaaclab/IsaacLab
experiment=sprite0615_wbt_pm01_v38_push_feetslide010
source_checkpoint="$root/artifacts/sprite0615_wbt_v36_model100_anchor/model_100.pt"
source_sha=e78e32edbb8c1bfdb5188ae2943b507a1b7bb67d5d53c55bedb8bf26d4113c1c
source_run=0000_v36_model100_frozen_source
run_name=v38_push_feetslide010_from_v36m100_50_env8192_lr2e6
job_logs="$root/logs/sprite0615_wbt_jobs"
mkdir -p "$job_logs" "$root/logs/rsl_rl/$experiment/$source_run"
log_file="$job_logs/${run_name}_$(date +%Y%m%d_%H%M%S).log"
cd "$root"
[[ $(sha256sum "$source_checkpoint" | awk '{print $1}') == "$source_sha" ]] || exit 4
cp --reflink=auto "$source_checkpoint" "$root/logs/rsl_rl/$experiment/$source_run/model_100.pt"
chmod 0444 "$root/logs/rsl_rl/$experiment/$source_run/model_100.pt"

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_actor_critic_only.py \
  --task Tracking-Flat-Sprite0615-WBT-Timescale150-PM01Rated-PhaseCycleB-Smooth015-VariableSpeed-FullPeakLegacyScale-AnkleTorque1e3-PushStage1-FeetSlide010-v0 \
  --num_envs 8192 --max_iterations 50 --headless --device cuda:0 \
  --experiment_name "$experiment" --run_name "$run_name" \
  --resume --load_run "$source_run" --checkpoint model_100.pt \
  agent.algorithm.learning_rate=0.000002 agent.algorithm.schedule=fixed agent.save_interval=2 \
  >"$log_file" 2>&1

run_dir=$(find "$root/logs/rsl_rl/$experiment" -maxdepth 1 -type d -name "*_${run_name}" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
checkpoint=$(find "$run_dir" -maxdepth 1 -name 'model_*.pt' -printf '%f\n' | sort -V | tail -1)
{
  echo "RUN_DIR=$run_dir"
  echo "CHECKPOINT=$run_dir/$checkpoint"
  echo "LOG_FILE=$log_file"
  echo "SOURCE_SHA256=$source_sha"
  echo "PUSH_STAGE=half-envelope"
  echo "FEET_SLIDE_WEIGHT=-0.10"
  echo "ANKLE_TORQUE_L2_WEIGHT=-1e-3"
} >"$job_logs/v38_handoff.txt"
echo "V38_TRAIN_COMPLETE run_dir=$run_dir checkpoint=$checkpoint"
