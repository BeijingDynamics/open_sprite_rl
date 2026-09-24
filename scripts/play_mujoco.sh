#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PACKAGE="$ROOT/baselines/sprite0825_stage2_g60_model3450_current"
CONTRACT="$PACKAGE/deploy/contract.json"
RUNNER="$PACKAGE/runtime/run_sprite0825_stage2_mujoco.py"
MJCF="$ROOT/deploy/sprite0825_v4_stage2/scene_external_pd.xml"
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

check_hash "$PACKAGE/deploy/policy.onnx" 43f213e4c5b9079e13b7f6f3635f224766227757417a0940ca49d585231016e3
check_hash "$MJCF" 6ec42ff665fde05db58b507cc2747fb2a877e64efea003ff6b3d9e2f8c72e04f

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
