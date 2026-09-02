# Sprite0825 Stage 2 G58F Model 1050 Candidate

Status: **PROVISIONAL VISUAL CANDIDATE - NOT QUALIFIED OR PROMOTED**

## Identity

- Checkpoint: `model_1050.pt`
- Checkpoint SHA-256: `deda0cc80023c00e6fd7723830bf3a9b9ceb4b33e72c9e1e446d9b0873f1d912`
- Source visual baseline: native G58B `model_925.pt`
- Source baseline SHA-256: `e4d74619be6ea0786057ede910785bf58f8f92088f72c431578fdf56644523e9`
- Task: `Isaac-Sprite0825-Stage2-AMP-G58Preserve925GentleYaw100Hz-Robust-v0`
- Training: 8,192 environments, 400 requested continuation updates, G58F checkpoint 1050 selected by the first complete numerical/style screen pass

The frozen V38 Stage 1 package and native model 925 remain unchanged. This
candidate does not replace either baseline until Tony completes the visual and
keyboard gates.

## Method And Contract

G58F is a conservative continuation of the Stage 2 AMP/teacher-anchored
command-conditioned policy. Relative to G58B, it extends command coverage to
`vx=(0.15, 0.45) m/s` and `yaw_rate=(-0.20, 0.20) rad/s` while preserving the
motion source, rewards, randomization, actions, 100 Hz policy rate, 31-joint
control, PM01-style actuators, and 0.04-second deployment handoff.

The deployment actor has 1,488 observations and 31 actions. Its contract
explicitly excludes horizontal base velocity, global root position, and global
yaw. MuJoCo uses the same ONNX policy, observation order, action scale, PD,
motor envelopes, contact settings, and initial-state protocol as Isaac.

## Automatic Evidence

- Isaac five-seed clean/robust matrix complete.
- Independent seed-44 robust-yaw confirmation: 99.375% survival.
- Full `0/+-0.20/+-0.40 rad/s` yaw matrix, seeds 42-46: 100% survival.
- Twenty start/stop/restart cycles: all pass; median start 0.12 s, median stop 0.60 s.
- All ten model-925 numerical style-regression gates pass.
- Four aligned Isaac physical-motor audits pass for J4340P and differential J4310P.
- MuJoCo straight, four yaw cases, and twenty-cycle start/stop matrix all pass.
- MuJoCo matrix summary SHA-256: `abb02c63e2b6c176ae2be54042af69a01575dc46759129f22d147810493a55b2`.

The knees reach the simulated directional torque-speed boundary at high speed
or turning. This passes the current envelope but has little margin and remains
a hardware-telemetry watch item.

## Deployment Artifacts

- ONNX SHA-256: `75a5c89552539da344e21566843f6c1fd1eac52e7601bbf8b5de64aaaae9eb26`
- JIT SHA-256: `e6ad39a857de921e6ac391b23c80e8ec4c31be7d0849298d098c1630abb33c91`
- Localized contract SHA-256: `53a7003b5bc035abdf1c7225e0fef681ffb964cdd6c5060d6fad4d537a11a4de`
- Isaac evidence archive SHA-256: `dd11efb90ff63b1eca5593a1c8478558400cf2e4b3969e31df0278fddfd912b4`
- Motor/export evidence archive SHA-256: `c69e284386c2f67fb06bb731c771913104f122f9bf3d3e36059b90b547992340`
- Training provenance archive SHA-256: `c5f66255eebeb457460ac6c5ec3f98a8475841d4b1eb6f9c9e25b0c120870c72`
- Provenance inventory SHA-256: `88dd9253126bdf29d30d0c2baed320008c33773e6f72fb7368f7bb6474d58e4f`

The provenance snapshot contains the complete Sprite AMP locomotion source
package, exact launcher, resolved Hydra environment and agent YAML files,
screeners, evaluators, motor audits, transition/full-yaw scripts, and exporter.

## Pending Hard Gates

1. Fixed-side A/B/C visual comparison against native model 925. Model 1050
   must preserve model 925's slower swing-foot descent and controlled touchdown.
2. MuJoCo keyboard review with `Z/X/Q/E`, including repeated walk/turn/stop/
   restart cycles and natural two-leg turning.
3. Visual rejection for persistent foot-edge walking, asymmetry, dragging,
   high-frequency swing-leg motion, or degraded gait rhythm overrides numerical
   metrics.
4. Only after these pass may a final manifest be generated and the package be
   renamed, made read-only, and promoted as Stage 2 qualified.

## Review Commands

Isaac fixed-side comparison:

```bash
cd /home/tony/sprite/sprite_isaaclab/IsaacLab
./review_sprite0825_g58b925_g58f1050_visual_ab_on_234.sh
```

MuJoCo keyboard review, only after the Isaac visual gate passes:

```bash
cd /home/tony/sprite
./review_sprite0825_g58f_model1050_mujoco_keyboard_on_234.sh
```
