# G60 model3450 robust and heading audit

## Scope

This audit uses the frozen G60 `model3450` checkpoint with the robust Isaac
task. The task applies the same asset randomization and interval pushes used by
the training family. All policy runs use 50 Hz actions and 500 Hz physics.

## Open-loop robust result

- Three seeds, 512 environments per seed, 60 seconds: 100% survival.
- Cadence: 134.86 to 135.18 steps/min.
- Estimated step length: 0.12666 to 0.12701 m.
- Path speed: 0.28422 to 0.28484 m/s for a 0.30 m/s command.
- Mean absolute heading drift: 8.65 to 9.08 degrees under repeated pushes.
- Twenty start/stop cycles: 100% survival and 99.61% complete protocol rate.
- Robust yaw response: -0.1944/+0.2030 rad/s for -0.20/+0.20 commands.
- Alternation: 100%; left/right swing-height gap: 0.000079 m.
- Joint tracking RMSE: 0.04042 rad.
- Mean contact slip: 0.02352/0.02305 m/s left/right.
- Mean contact tilt: 0.01547/0.01694 rad left/right.
- Aligned 4340P and differential 4310P torque-speed exceedance: 0.

The open-loop heading drift is expected because global/integrated yaw is not an
actor observation. The actor can follow a yaw-rate command but cannot infer an
accumulated world-heading error from local angular velocity alone.

## PM01-style outer heading result

The controller remains outside the actor:

```text
wz = clip(0.5 * wrap(target_yaw - integrated_imu_yaw), -0.2, 0.2)
```

Across the same three seeds, mean absolute heading error is 0.552, 0.602, and
0.551 degrees. Every environment survives. Cadence, step length, and path speed
remain inside the open-loop ranges. The actor still receives only the existing
`[vx, vy, yaw_rate]` command and never receives global yaw or horizontal speed.

## Root causes fixed during evaluation

1. The style evaluator passed an oversized tensor directly to CUDA
   `torch.quantile`, so the run completed simulation but failed before writing
   JSON. Full-data mean/RMS/max remain exact; only percentile calculation now
   uses a deterministic sample capped at one million values.
2. `--heading-hold` set `is_heading_env` but G60 inherited
   `heading_command=False`. The private command update therefore ignored the
   heading target and retained reset-time random yaw commands. Those invalid
   results reached about 82 degrees absolute error and are preserved only under
   `evaluation/isaac_robust_heading_hold/invalid_pre_heading_flag_fix`.
3. The corrected evaluator explicitly enables heading mode and records the
   PM01/runtime contract values: stiffness 0.5 and yaw-rate limit 0.2 rad/s.

## Deployment boundary

Integrated IMU yaw belongs to the outer command controller, not to the actor
observation. Standing resets the heading reference. Manual yaw commands are
bounded and passed through; releasing the command holds the new heading. A
stale or invalid IMU must block motor transmission through the safety layer.

This audit qualifies the software behavior in simulation. It does not qualify
an IMU mounting transform, USB-CAN FD timing, motor map, ankle calibration, or
hardware emergency-stop chain.
