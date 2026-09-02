#!/usr/bin/env bash
set -euo pipefail

root=/home/tony/sprite
package="$root/sprite_isaaclab/IsaacLab/baselines/sprite0825_stage2_g58f_model1050_visual_candidate"
contract="$package/deploy/contract.json"
mjcf="$root/sprite_deploy/mujoco/sprite0825_v4_stage2/scene_external_pd.xml"
runner="$package/runtime/run_sprite0825_stage2_mujoco.py"
out="$package/evaluation/mujoco_keyboard_review"
seconds="${1:-600}"

verify() {
  local file="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$file" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || {
    echo "SHA mismatch: $file $actual" >&2
    exit 4
  }
}

select_display() {
  local candidate
  for candidate in "${DISPLAY:-}" :1 :0; do
    [[ -n "$candidate" ]] || continue
    if DISPLAY="$candidate" xdpyinfo >/dev/null 2>&1; then
      export DISPLAY="$candidate"
      echo "Using desktop DISPLAY=$DISPLAY"
      return 0
    fi
  done
  echo "No accessible desktop display found." >&2
  return 1
}

verify "$package/model_1050.pt" deda0cc80023c00e6fd7723830bf3a9b9ceb4b33e72c9e1e446d9b0873f1d912
verify "$package/deploy/policy.onnx" 75a5c89552539da344e21566843f6c1fd1eac52e7601bbf8b5de64aaaae9eb26
verify "$contract" 53a7003b5bc035abdf1c7225e0fef681ffb964cdd6c5060d6fad4d537a11a4de
verify "$runner" cdb02b432c2be03091f895b8610a96972fd62c872fee5e5b6079aadd5c6ecf20
verify "$mjcf" 6ec42ff665fde05db58b507cc2747fb2a877e64efea003ff6b3d9e2f8c72e04f

"$root/.venv/bin/python" - "$contract" <<'PY'
import json
import math
import sys

c = json.load(open(sys.argv[1]))
root_height = float(c["deployment_initial_root_height_m"])
assert math.isclose(root_height, 0.52, rel_tol=0.0, abs_tol=1.0e-6)
assert c["mujoco_base_ang_vel_source"] == "freejoint_local"
assert c["mujoco_contact"]["sliding_friction"] == 1.0
assert c["mujoco_contact"]["robot_self_collision_enabled"] is False
assert c["observation_has_horizontal_base_velocity"] is False
print(
    f"CONTRACT_OK root={root_height:.9f} imu=freejoint_local friction=1.0 "
    "self_collision=off horizontal_velocity=absent"
)
PY

select_display
mkdir -p "$out"
echo "MODEL1050 IS A VISUAL CANDIDATE, NOT THE PROMOTED BASELINE."
echo "Controls: Z walk at 0.30 m/s; X stop; Q add +0.15 rad/s left yaw; E add -0.15 rad/s right yaw."
echo "Suggested review: Z, Q, E, E, Q, then X; repeat at least five walk/turn/stop/restart cycles."
echo "Use at most two consecutive Q or E taps; the automatic matrix already covers +-0.40 rad/s."
echo "Hard visual gate: touchdown must remain as slow and controlled as native model925."
echo "Also check natural two-leg turns, foot-edge contact, symmetry, and restart."
echo "Close the MuJoCo window when finished."

"$root/.venv/bin/python" "$runner" \
  --contract "$contract" --mjcf "$mjcf" \
  --seconds "$seconds" --command-x 0.30 --command-yaw 0.0 --command-ramp 0.8 \
  --viewer --real-time --log-every 5 \
  --summary-json "$out/last_review.json"
