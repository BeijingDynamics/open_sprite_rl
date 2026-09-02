#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/tony/sprite/sprite_isaaclab/IsaacLab
PY=/home/tony/sprite/sprite_isaaclab/env_isaaclab/bin/python
RAW=/home/tony/sprite/stage2_pm01_native_audit_2026-08-27/g55_straight_vx045/pm01_g55_straight_vx045_raw.npz
PM100=/home/tony/sprite/sprite_isaaclab/assets/sprite0615_references/stage2_amp/pm01_native_g55_straight_vx045_100hz_v1.npz
MAPPED100=/home/tony/sprite/sprite_isaaclab/assets/sprite0615_references/stage2_amp/sprite0825_pm01_native_straight_vx045_mapped_100hz_v1.npz
CANON100=/home/tony/sprite/sprite_isaaclab/assets/sprite0615_references/stage2_amp/sprite0825_pm01_native_straight_vx045_mapped_100hz_headingcanon_v1.npz
OUT=/home/tony/sprite/sprite_isaaclab/assets/sprite0615_references/stage2_amp/g57_sprite0825_native_pm01_100hz_v1
CONTRACT="$ROOT/baselines/sprite0615_wbt_v38_model2_stage1_qualified/deploy/model_2/contract.json"
TEMPLATE="$ROOT/baselines/sprite0615_wbt_v38_model2_stage1_qualified/deploy/model_2/reference_motion.npz"

cd "$ROOT"
"$PY" convert_pm01_native_telemetry_to_expert_npz.py \
  --input "$RAW" \
  --output "$PM100" \
  --summary audits/pm01_native_g55_straight_vx045_100hz_v1.json \
  --fps 100
"$PY" retarget_pm01_amp_expert_to_sprite0615.py \
  --pm01 "$PM100" \
  --sprite-contract "$CONTRACT" \
  --sprite-template "$TEMPLATE" \
  --output "$MAPPED100" \
  --output-fps 100 \
  --length-scale 0.6884057971014492
"$PY" canonicalize_sprite_wbt_straight_heading.py \
  --input "$MAPPED100" \
  --output "$CANON100"
"$PY" build_sprite0825_g57_native_amp_expert.py \
  --mapped "$CANON100" \
  --phase-source "$PM100" \
  --output "$OUT" \
  --stand-frames 400 \
  --walk-copies 3
sha256sum "$RAW" "$PM100" "$MAPPED100" "$CANON100" "$OUT"/*.npz \
  | tee audits/sprite0825_g57_native_amp_SHA256SUMS
