# Sprite0825 Stage 2 Final Audit

Date: 2026-09-02

Status: **COMPLETE AND QUALIFIED**

## Frozen Result

- Package: `/home/tony/sprite/sprite_isaaclab/IsaacLab/baselines/sprite0825_stage2_g58f_model1050_stage2_qualified`
- Checkpoint: `model_1050.pt`
- Checkpoint SHA-256: `deda0cc80023c00e6fd7723830bf3a9b9ceb4b33e72c9e1e446d9b0873f1d912`
- ONNX SHA-256: `75a5c89552539da344e21566843f6c1fd1eac52e7601bbf8b5de64aaaae9eb26`
- TorchScript SHA-256: `e6ad39a857de921e6ac391b23c80e8ec4c31be7d0849298d098c1630abb33c91`
- Contract SHA-256: `53a7003b5bc035abdf1c7225e0fef681ffb964cdd6c5060d6fad4d537a11a4de`
- Package manifest SHA-256: `99e00356c4861919ef6275b66eb574b07bfc9bbdd8ad94af597e26731d678abf`

The package verifier passes its complete SHA-256 inventory. No path in the
qualified package has owner, group, or other write permission.

The Stage 1 V38 baseline remains content-identical to checkpoint SHA-256
`b8d1851de07e654c367d0ad9d6d2e32a9fbc9588e4f26aabd7e5e4d0d71166ff`.
Its complete delivery manifests pass, and directory write permissions found
during the final audit were removed without changing content. TWIST2 was not
used or modified.

## Isaac Lab Gates

| Requirement | Final evidence |
| --- | --- |
| Five-seed clean termination rate | 0 |
| Push termination rate | 1/480 = 0.208% aggregate; independent seed-44 repeat 2/320 = 0.625% |
| Zero-command stand for 30 s | 100% survival for seeds 42-46 |
| Stand horizontal drift | Worst 0.01444 m, limit 0.10 m |
| Stand yaw drift | Worst 0.2864 deg, limit 5 deg |
| 0.15/0.30/0.45 m/s speed tracking | Worst MAE 0.07665 m/s, limit 0.08 m/s |
| 0/+-0.20/+-0.40 rad/s yaw tracking | 100% supplemental survival; worst MAE 0.03356 rad/s, limit 0.12 rad/s |
| Start/stop/restart | All 20 cycles complete with 100% survival |
| Style preservation | All ten model-925 regression gates pass; Tony approved the fixed-side visual review |
| Perturbation recovery | PM01-style interval pushes every 1-3 s; aggregate termination below 1% |

The original 96-environment seed-44 robust-yaw screen remains unmodified and
shows one termination, or 1.0417%, causing that stricter per-seed summary to
say `pass=false`. The Goal specifies aggregate push termination no higher than
1%; the five-seed aggregate is 0.208%, and the larger independent seed-44
repeat is 0.625%. Both directly satisfy that requirement. This discrepancy is
preserved as evidence rather than hidden or rewritten.

## MuJoCo Gates

| Requirement | Final evidence |
| --- | --- |
| Same deployment contract | Same ONNX, observation order, action scale, PD, limits, contact and startup protocol |
| Horizontal base velocity excluded | Contract and runtime both report false |
| Continuous straight walking | 60 s survived; vx MAE 0.01694 m/s |
| Start/stop/restart | All 20 automatic cycles survived |
| Left/right turning | `+-0.20` and `+-0.40 rad/s` all survived; worst yaw MAE 0.03855 rad/s |
| Keyboard interface | `Z/X/Q/E` reviewed and approved by Tony |
| Human review session | 385.35 s survived; vx MAE 0.02362 m/s; yaw MAE 0.03480 rad/s; zero self-contact samples |
| Foot contact and natural motion | Tony reported very good control feel and approved the final visual gate |
| Physical motor limits | J4340P and differential J4310P RMS, peak, speed and torque-speed gates pass |

## Deployment Contract

- Actor observations: 1,488
- Actions: 31 joint targets
- Physics: 500 Hz
- Policy: 100 Hz
- Commands: `vx`, reserved `vy`, and `yaw_rate`
- History: 15 frames of joint position, joint velocity, previous action, base
  angular velocity and projected gravity, plus the current command
- Horizontal base velocity: absent
- Global root position: absent
- Global yaw: absent
- Deployment handoff: 0.04 s
- Initial velocity injection: none

## Deliverables

The frozen package contains the checkpoint, ONNX and TorchScript policies,
itemized contract, exact training source and resolved Hydra configurations,
Isaac evaluators and matrices, MuJoCo runtime and keyboard review, physical
motor audits, technical decisions and failed-branch history, human approvals,
portable SHA-256 inventory, qualified README and independent verifier.

## Residual Hardware Watch Item

The simulated J4340P knee directional torque-speed p99 reaches the configured
envelope boundary during high-speed walking or turning. It passes the current
simulation gate but has little margin. Knee torque, velocity, current and bus
voltage must be monitored during staged hardware bring-up. This does not block
Stage 2 sim2sim qualification.
