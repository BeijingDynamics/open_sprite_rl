# Sprite0825 G60 Time-Normalized Waist Plan

Status: training on the RTX 4090; G59 model2999 remains the frozen fallback.

## Why G60 exists

The controlled MuJoCo trace confirmed Tony's visual observation. At the same
`0.30 m/s` command, the qualified 100 Hz G58F policy used about 247 steps/min
and 0.074 m per step, while native-50-Hz G59 model2999 used about 123 steps/min
and 0.131 m per step. The G59 waist-roll joint's 5th-to-95th percentile range
was 0.351 rad, versus 0.100 rad in G58F.

The G59 expert clips use about 150 steps/min. Their mean speed is 0.214 m/s and
their implied step length is 0.086 m. G59 therefore fixed the old high-frequency
small-step gait, but overshot toward a lower cadence and excessive waist roll.

Reconstructing the global `waist_yaw_link` attitude from the same MuJoCo traces
confirmed that this is visible upper-body sway rather than harmless internal
waist compensation. Torso-roll RMS / 95th-percentile absolute roll changed from
`2.11 / 2.86 deg` in G58F to `7.52 / 10.06 deg` in G59. Pelvis roll remained
small (`0.32 deg` RMS in G59), so most of the extra motion is above the pelvis.
Tony has a recording of the G58F review; no repeat visual A/B is required.

G59 changed EngineAI PM01's policy period from 0.01 s to 0.02 s while retaining
its per-frame action-rate and action-smoothness weights. For equal physical
motion, those finite differences scale with `dt^2` and `dt^4`. The repository
already contains the corresponding 31-action/23-action compensation formulas,
but G59 did not apply them.

## Isolated G60 changes

- Keep the G59 50 Hz policy, 500 Hz physics, 795-D actor, 31 actions, AMP expert,
  asset, action scales, motor limits, command ranges, and perturbations.
- Change action-rate weight from `-0.06` to `-0.011129032258064516`.
- Change action-smoothness weight from `-0.04` to
  `-0.0018548387096774194`.
- Preserve the physical-time curriculum by changing those two intervals from
  4800 100-Hz steps to 2400 50-Hz steps.
- Add a soft waist-roll reward with a free region of `+/-0.10 rad`, scale 4.0,
  and weight 0.2. The waist is not locked.
- Continue from frozen G59 model2999 at learning rate `5e-7` for 600 iterations.

## Acceptance gates

- Cadence: 135-155 steps/min at a `0.30 m/s` command.
- Estimated step length: 0.10-0.13 m.
- Waist-roll 5th-to-95th percentile range: 0.18-0.25 rad; do not collapse it.
- Preserve G59 visual gait quality, arm motion, foot clearance, and control feel.
- Pass straight, left/right `0.20 rad/s`, repeated stop/restart, and MuJoCo
  survival gates.
- No regression in foot-edge contact, ankle differential load, knee
  torque-speed envelope, or the no-horizontal-base-velocity actor contract.
- Quantitative promotion is multi-objective: a checkpoint is rejected if it
  improves cadence but materially degrades survival, command response, foot
  contact, torque limits, or tracking continuity.

## Active run

- Task: `Isaac-Sprite0825-Stage2-AMP-G60TimeNormalizedWaist50Hz-Robust-v0`
- Source SHA256:
  `eb84e5aedb8ace647d877788a98c86db6ce9670e4c003890e6e1d2ef21205a98`
- Cloud run begins at iteration 2999 and ends at iteration 3599.
- Persistent job directory: `/root/gpufree-data/g60_time_normalized_waist`
- Isaac run directory timestamp: `2026-09-03_12-13-50`

## Automatic post-training screen

- Persistent screen PID: `/root/gpufree-data/g60_time_normalized_waist/screen_driver.pid`
- Driver: `/root/gpufree-data/g60_time_normalized_waist/screen_g60_after_training_cloud.sh`
- Results: `/root/gpufree-data/g60_time_normalized_waist/screen`
- Stage 1 evaluates G59 plus every saved G60 milestone for 24 seconds at
  `0.30 m/s`, then ranks survival, cadence, step length, speed, and heading
  drift. Stage 2 runs grid, five stop/restart cycles, 30-second yaw, AMP style,
  contact, and aligned motor-envelope measurements on G59 plus the top three
  G60 checkpoints.
- The screen process waits for the exact live training PID and starts only
  after training exits, so it does not compete with training for GPU memory.

## Post-run design audit

G60 exposed a continuation-specific flaw in EngineAI's curriculum helper. Its
parameter is named `interval_epochs`, but the implementation uses
`env.common_step_counter`. RSL-RL restores the learning iteration from the
checkpoint while a newly created Isaac environment resets that counter to
zero. G59 model2999 had reached 15 curriculum updates and a scale of
`0.1 ** (0.8 ** 15) = 0.9221798488`; G60 restarted energy, action-rate, and
action-smoothness regularization at scale 0.1. Therefore G60 is not the fully
isolated experiment originally intended.

All G60 milestones remain valid measurements and are screened normally. A G61
configuration is staged, but not launched, with restart-invariant direct reward
functions and fixed mature weights:

- energy: `-0.003688719395`
- 50 Hz action rate: `-0.010262969285`
- 50 Hz action smoothness: `-0.001710494881`

G61 will start from frozen G59 model2999 only if the completed G60 screen shows
that no G60 checkpoint satisfies the multi-objective gates. This preserves a
clean experimental lineage rather than continuing from a known confounded run.
