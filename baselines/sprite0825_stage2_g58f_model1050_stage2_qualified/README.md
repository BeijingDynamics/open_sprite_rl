# Sprite0825 Stage 2 Command-Conditioned Locomotion

Status: **QUALIFIED AND READ-ONLY**

This package is created only after Tony approves both the fixed-side Isaac
visual comparison and the MuJoCo `Z/X/Q/E` keyboard review. The approval
records, prequalification audit, complete evidence, source provenance, and
package-wide SHA-256 manifest are included.

## Policy

- Checkpoint: `model_1050.pt`
- SHA-256: `deda0cc80023c00e6fd7723830bf3a9b9ceb4b33e72c9e1e446d9b0873f1d912`
- Task: `Isaac-Sprite0825-Stage2-AMP-G58Preserve925GentleYaw100Hz-Robust-v0`
- Control rate: 100 Hz
- Physics step: 0.002 s
- Actions: 31 joint targets
- Deployment handoff: 0.04 s
- Initial velocity injection: none

The actor has 1,488 observations: 15-frame histories of joint position, joint
velocity, previous action, base angular velocity and projected gravity, followed
by the current 3-D velocity command. It does not receive horizontal base
velocity, global root position, or global yaw.

## Commands

- `vx`: forward speed, qualified at 0.15, 0.30 and 0.45 m/s
- `vy`: interface retained; Stage 2 training and acceptance keep it at zero
- `yaw_rate`: qualified at `+-0.20` and `+-0.40 rad/s`
- zero velocity: stand/stop command

MuJoCo keyboard controls:

- `Z`: walk at 0.30 m/s
- `X`: stop
- `Q`: add +0.15 rad/s left yaw
- `E`: add -0.15 rad/s right yaw

## Main Artifacts

- `deploy/policy.onnx`: portable actor
- `deploy/policy.pt`: TorchScript actor
- `deploy/contract.json`: portable observation/action/PD/motor/contact contract
- `runtime/run_sprite0825_stage2_mujoco.py`: MuJoCo runner and physical-motor telemetry
- `evaluation/`: Isaac, style, motor, transition, yaw, MuJoCo and keyboard evidence
- `provenance/`: exact AMP source, launcher, resolved Hydra configuration and evaluators
- `approvals/`: Tony's Isaac visual and MuJoCo keyboard acceptance records
- `PREQUALIFICATION_AUDIT.md`: requirement-by-requirement evidence before approval
- `SHA256SUMS`: package-wide immutable inventory

## Important Residual

The J4340P knees reach the simulated directional torque-speed boundary during
high-speed walking or turning. They remain inside the configured envelope but
have little simulated margin. Preserve this as a real-hardware telemetry watch
item; do not infer extra hardware margin from successful simulation.

## Verification

From the package directory:

```bash
./verify_qualified_package.sh
```

The verifier checks the complete SHA-256 inventory, read-only permissions,
checkpoint and deployment hashes, no-horizontal-velocity actor contract, both
human approval records, style/motor/yaw/transition/MuJoCo gates, and the
keyboard-session summary.
