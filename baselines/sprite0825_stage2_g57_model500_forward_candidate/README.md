# Sprite0825 Stage 2 G57 model 500 forward candidate

This package freezes the best forward-command checkpoint from the isolated G57
native PM01 scratch run. It is a Stage 2 forward candidate, not a fully
qualified Stage 2 delivery: yaw, push, motor, gait-regression, and MuJoCo gates
remain open.

- Task: `Isaac-Sprite0825-Stage2-AMP-G57NativePM01Forward100Hz-v0`
- Checkpoint: `model_500.pt`
- Checkpoint SHA-256:
  `c4bb13b271e840f6f9c182e5681c5ab7ebc1f48055406fb49bb1d6afda8e419d`
- Asset: `sprite0825_sanitized_v4/sprite0825.usd`
- Actor: 1,488 observations, 31 actions, no horizontal base velocity
- Control: 500 Hz physics, 100 Hz policy
- Expert: 10% quiet native stand and 90% accepted native PM01 steady walk

Passed clean evidence:

- 128-environment speed grid survival: 100%
- Mean vx at commands 0.00/0.15/0.30/0.45 m/s:
  0.0024/0.1814/0.3060/0.4176 m/s
- Speed MAE: 0.0026/0.0314/0.0089/0.0324 m/s
- 64-environment 30-second stand gate: 100%
- Stand p90 displacement: 0.0332 m
- Stand p90 final heading error: 2.37 degrees
- 64-environment 20-cycle sharp start/stop/restart gate: 100%
- Median start/stop latency: 0.36/0.85 seconds

The frozen V38 Stage 1 package remains unchanged. This candidate does not use
V38/G20 motion in its AMP expert and was trained from scratch.
