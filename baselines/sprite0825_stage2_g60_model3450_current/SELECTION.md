# Sprite0825 G60 model3450 selection

Status: selected cross-simulator candidate. G59 model2999 remains the frozen
fallback until runtime safety qualification is complete.

## Identity

- Task: `Isaac-Sprite0825-Stage2-AMP-G60TimeNormalizedWaist50Hz-v0`
- Checkpoint: `model_3450.pt`
- Checkpoint SHA256: `6a1a80a2a2f7073698c0886133c325a46462ace6bbd3cf7de70246beafb85f15`
- Policy rate: 50 Hz
- Physics rate: 500 Hz
- Actor observation: 795 dimensions, eight history frames
- Actor excludes horizontal base velocity, global position, and global yaw
- ONNX/PyTorch maximum absolute error: `1.65775418e-07`

## Isaac qualification

G60 model3450 passed the deterministic gait, command-grid, five-cycle
start/stop, yaw, contact, style, and aligned motor-envelope protocols.

- Survival rate: 1.0 in every protocol
- Cadence: 136.59 steps/min (gait), 135.51 steps/min (style)
- Estimated step length: 0.12227 m
- Path speed at 0.30 m/s command: 0.27798 m/s
- Mean absolute heading drift: 0.945 deg
- Command grid actual vx: -0.00082, 0.16033, 0.26961, 0.38735 m/s
- Start/stop complete-cycle rate: 1.0
- Median start/stop latency: 0.12/0.22 s
- Right/left yaw response at +/-0.20 rad/s: -0.19509/+0.20185 rad/s
- Foot alternation fraction: 1.0
- Left/right mean contact tilt: 0.00867/0.00963 rad
- Left/right mean contact slip: 0.01854/0.01867 m/s
- Contact duty gap: 0.00609
- 4340P aligned torque-speed exceedance fraction: 0.0
- Differential 4310P aligned torque-speed exceedance fraction: 0.0

## Robust Isaac qualification

The candidate also passed the G60 robust task with asset randomization and
repeated interval pushes.

- Three seeds, 512 environments each, 60 seconds: 100% survival
- Cadence: 134.86 to 135.18 steps/min
- Estimated step length: 0.12666 to 0.12701 m
- Twenty start/stop cycles: 99.61% complete protocol rate, 100% survival
- Robust +/-0.20 yaw response: -0.1944/+0.2030 rad/s
- Joint tracking RMSE: 0.04042 rad; foot alternation: 1.0
- Mean contact slip: 0.02352/0.02305 m/s left/right
- Mean contact tilt: 0.01547/0.01694 rad left/right
- 4340P and differential 4310P aligned envelope exceedance: 0
- Open-loop absolute heading drift: 8.65 to 9.08 deg under repeated pushes
- PM01-style IMU-yaw outer control reduces the same three seeds to
  0.552/0.602/0.551 deg without changing cadence, step length, or speed

The heading controller is outside the actor and supplies only the existing
yaw-rate command. Global yaw and horizontal velocity remain absent from actor
observations. See `ROBUST_HEADING_AUDIT.md` and the machine-readable reports
under `evaluation/isaac_robust*`.

## Experiment caveat

G60 was continued from G59. The upstream EngineAI curriculum helper uses the
environment common-step counter despite naming its argument `interval_epochs`.
That counter reset on continuation, so regularization returned to its initial
scale and matured during G60. Therefore G60 is behaviorally qualified but is
not evidence for a perfectly isolated time-normalization experiment. A
restart-invariant G61 configuration is retained for future controlled work;
it is not needed merely to erase this caveat if G60 passes MuJoCo.

## MuJoCo qualification

The candidate passed all six automated MuJoCo cases with the same external-PD
scene and runtime used for the G59 baseline: 60 seconds straight, +/-0.20 and
+/-0.40 rad/s turns, and 20 walk/stop cycles.

- All six cases survived with no robot self-contact
- Straight actual vx: 0.27431 m/s; vx MAE: 0.02790 m/s
- Left/right +/-0.20 actual yaw: +0.20480/-0.20904 rad/s
- Left/right +/-0.40 actual yaw: +0.40049/-0.40544 rad/s
- Twenty start/stop cycles completed
- MuJoCo cadence: 131.54 steps/min
- MuJoCo estimated step length: 0.12491 m
- Torso roll RMS/p95: 6.93/9.24 deg
- Torso pitch mean/RMS: -0.28/0.31 deg
- Mean lateral ankle separation: 0.1614 m
- 4340P maximum peak-torque ratio: 0.6653
- 4340P maximum torque-speed-envelope ratio: 1.0 (clipping boundary)
- 4310P maximum rated-RMS ratio: 0.6057
- 4310P maximum peak-torque ratio: 0.4470
- 4310P maximum torque-speed-envelope ratio: 0.4470

Compared with G59, MuJoCo cadence increased from about 123.43 to 131.54
steps/min, estimated step length decreased from about 0.1305 to 0.1249 m, and
torso-roll RMS decreased from about 7.52 to 6.93 deg.

## Remaining gates

- Select and benchmark the target SBC under ONNX, logging, IMU, and USB-CAN load
- Fill the 31-motor CAN/zero/sign/limit/MIT-range contract
- Calibrate both physical differential ankles
- Validate the pelvis IMU mounting transform, timestamps, and gyro bias
- Demonstrate the physical emergency-stop and watchdog chain
- Complete no-transmit shadow mode before any protected actuation
