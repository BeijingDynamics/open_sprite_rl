#!/usr/bin/env bash
set -euo pipefail

package="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

fail() {
  echo "QUALIFIED PACKAGE VERIFICATION FAILED: $*" >&2
  exit 4
}

[[ -f "$package/QUALIFIED" ]] || fail "QUALIFIED marker missing"
[[ -f "$package/SHA256SUMS" ]] || fail "SHA256SUMS missing"

(
  cd "$package"
  sha256sum -c SHA256SUMS
)

if find "$package" -perm /222 -print -quit | grep -q .; then
  fail "package contains writable entries"
fi

grep -q '^APPROVED:' "$package/approvals/isaac_visual.txt" \
  || fail "explicit Isaac visual approval missing"
grep -q '^APPROVED:' "$package/approvals/mujoco_keyboard.txt" \
  || fail "explicit MuJoCo keyboard approval missing"

jq -e '.observation_has_horizontal_base_velocity == false' "$package/deploy/contract.json" >/dev/null \
  || fail "actor contract contains horizontal base velocity"
jq -e '.pass == true and (.failed_gates | length) == 0 and ([.gates[].pass] | all)' \
  "$package/evaluation/g58f_model1050_style_screen_evidence/model_1050_vs_925_style_gates.json" >/dev/null \
  || fail "style gates do not pass"
jq -e '.qualified == true and ([.gates[]] | all)' \
  "$package/evaluation/mujoco_contract_defaults/matrix_summary.json" >/dev/null \
  || fail "MuJoCo automatic matrix does not pass"
jq -e '.qualified == true' \
  "$package/evaluation/model1050_postqual/motor_aligned/summary.json" >/dev/null \
  || fail "physical-motor audit does not pass"
jq -e '.survival_rate == 1 and .complete_cycle_protocol_rate == 1 and .cycles >= 20' \
  "$package/evaluation/model1050_supplemental/transitions20_seed42.json" >/dev/null \
  || fail "Isaac transitions do not pass"
jq -e '.survived == true and .robot_self_contacts.sample_count == 0' \
  "$package/evaluation/mujoco_keyboard_review/last_review.json" >/dev/null \
  || fail "MuJoCo keyboard summary does not pass"

for result in "$package"/evaluation/model1050_supplemental/full_yaw_seed*.json; do
  jq -e '.survival_rate == 1 and ([.grid[].mae_yaw_rate <= 0.12] | all)' "$result" >/dev/null \
    || fail "full-yaw result does not pass: $result"
done

echo "QUALIFIED_PACKAGE_VERIFIED $package"
