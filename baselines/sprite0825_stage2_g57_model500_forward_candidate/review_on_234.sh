#!/usr/bin/env bash
set -euo pipefail

root=/home/tony/sprite/sprite_isaaclab/IsaacLab
package="$root/baselines/sprite0825_stage2_g57_model500_forward_candidate"
checkpoint="$package/model_500.pt"
expected_sha=c4bb13b271e840f6f9c182e5681c5ab7ebc1f48055406fb49bb1d6afda8e419d

test -f "$checkpoint"
actual_sha="$(sha256sum "$checkpoint" | awk '{print $1}')"
[[ "$actual_sha" == "$expected_sha" ]] || {
  echo "Checkpoint SHA mismatch: $actual_sha" >&2
  exit 4
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

select_display
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=yes
cd "$root"
source ../env_isaaclab/bin/activate

echo "Sprite0825 G57 native PM01 model 500 forward candidate"
echo "Sequence repeats twice: stand -> 0.15 -> 0.30 -> 0.45 -> stop -> restart -> stand"
echo "Focus: native gait quality, speed separation, foot contact, posture, arms, stopping, and restart."

./isaaclab.sh -p play_sprite0825_g57_command_sequence.py \
  --task Isaac-Sprite0825-Stage2-AMP-G57NativePM01Forward100Hz-v0 \
  --checkpoint "$checkpoint" \
  --repeats 2 \
  --device cuda:0
