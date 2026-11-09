#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
BASELINE="${BASELINE:-g74}"
case "$BASELINE" in
  g74)
    PACKAGE="$ROOT/baselines/sprite0825_stage2_g74_model3000"
    V5="$ROOT/candidates/sprite0825_stage2_g74_model5999_4340_shoulders"
    RUNNER="$V5/runtime/run_sprite0825_stage2_mujoco.py"
    MJCF="$V5/assets/mujoco/sprite0825_v5_4340_shoulders/scene_external_pd.xml"
    ONNX_SHA=05f7ea02b6020ec44f4d5fa71e579294a73dc8c87d4b54f11198a7e83cb96d5d
    MJCF_SHA=b4f6dd161f6af1d4fadb44da155d1ff3c319341b541aab99fc4c176c22a40d9d
    ;;
  g60)
    PACKAGE="$ROOT/baselines/sprite0825_stage2_g60_model3450_current"
    RUNNER="$PACKAGE/runtime/run_sprite0825_stage2_mujoco.py"
    MJCF="$ROOT/deploy/sprite0825_v4_stage2/scene_external_pd.xml"
    ONNX_SHA=43f213e4c5b9079e13b7f6f3635f224766227757417a0940ca49d585231016e3
    MJCF_SHA=6ec42ff665fde05db58b507cc2747fb2a877e64efea003ff6b3d9e2f8c72e04f
    ;;
  *) echo "Unknown baseline: $BASELINE (use g74 or g60)" >&2; exit 2 ;;
esac
CONTRACT="$PACKAGE/deploy/contract.json"
PYTHON="${PYTHON:-python3}"
SECONDS="${1:-120}"
VIEWER="${VIEWER:-1}"

check_hash() {
  local file="$1" expected="$2" actual
  actual="$(sha256sum "$file" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || {
    echo "SHA256 mismatch: $file" >&2
    exit 4
  }
}

check_hash "$PACKAGE/deploy/policy.onnx" "$ONNX_SHA"
check_hash "$MJCF" "$MJCF_SHA"

mkdir -p "$ROOT/outputs"
viewer_args=()
if [[ "$VIEWER" == "1" ]]; then
  viewer_args+=(--viewer --real-time)
  echo "Controls: Z walk, X stop, Q left, E right"
else
  echo "Running headless MuJoCo smoke test"
fi

exec "$PYTHON" "$RUNNER" \
  --contract "$CONTRACT" \
  --mjcf "$MJCF" \
  --seconds "$SECONDS" \
  --command-x 0.30 \
  --command-yaw 0.0 \
  --command-ramp 0.8 \
  "${viewer_args[@]}" --log-every 5 \
  --summary-json "$ROOT/outputs/mujoco_last_review.json"
