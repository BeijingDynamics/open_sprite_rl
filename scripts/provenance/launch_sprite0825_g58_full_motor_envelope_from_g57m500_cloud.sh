#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/sprite/sprite_isaaclab/IsaacLab
EXPERIMENT="$ROOT/logs/rsl_rl/sprite0825_stage2_g58_full_motor_envelope"
IMPORT_RUN="$EXPERIMENT/g57m500_import"
SOURCE="$ROOT/logs/rsl_rl/sprite0825_stage2_g57_native_pm01_unconditioned/2026-08-31_14-16-07_stage2_G57_native_pm01_scratch_1000_env8192/model_500.pt"
OUT=/root/gpufree-data/g58_full_motor_envelope_from_g57m500_300
LOG="$OUT/train.log"
TASK=Isaac-Sprite0825-Stage2-AMP-G58FullMotorEnvelope100Hz-Robust-v0

mkdir -p "$IMPORT_RUN" "$OUT"
cp -f "$SOURCE" "$IMPORT_RUN/model_500.pt"
sha256sum "$SOURCE" "$IMPORT_RUN/model_500.pt" >"$OUT/source_sha256.txt"

cd "$ROOT"
source /opt/conda/etc/profile.d/conda.sh
conda activate isaaclab
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=yes

exec ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task "$TASK" \
  --num_envs 8192 \
  --max_iterations 300 \
  --headless \
  --device cuda:0 \
  --resume \
  --load_run g57m500_import \
  --checkpoint model_500.pt \
  agent.run_name=stage2_G58A_full_motor_envelope_from_G57m500_300_env8192 \
  agent.save_interval=25 \
  >"$LOG" 2>&1
