# G74 model3000: stable pre-reverse baseline

This is the frozen checkpoint used for the supervised ICM45686 800 Hz walking
milestone and forward/turn gamepad trials. It is **not** the G74 model5999
research candidate, the earlier G60 model3450, or a SuperMimic policy.

- [Training checkpoint](model_3000.pt), [ONNX](deploy/policy.onnx),
  [TorchScript](deploy/policy.pt), and [deployment contract](deploy/contract.json).
- [Resolved environment](params/env.yaml) and [PPO/AMP configuration](params/agent.yaml):
  original run snapshots with only host paths and the output directory localized.
  These snapshots are for comparison; use the registered task to launch.
- [G71 V20 source reference](../../assets/references/g71/sprite0825_pm01_g55_ik_contactclean_wbt_v1.npz)
  and [exact AMP dataset](../../assets/references/stage2_amp/g72_g71_4340_shoulders_100hz_v1/manifest.json).
- [Shared v5 simulation assets](../../candidates/sprite0825_stage2_g74_model5999_4340_shoulders/assets):
  the existing model5999 directory hosts the byte-identical simulation assets
  and runner. No model5999 weights are used in G74 model3000 playback.
  This URDF has a 2.07092842025253 kg waist-yaw link, not the later 3.7391 kg revision.
- [Historical seven-case MuJoCo summary](evaluation/mujoco_matrix_summary.json):
  48 passed gates, with a private trace path removed. The [Isaac dual-seed
  screen](evaluation/dualseed_selection.json) passed both 3000 and 5999 on
  seeds 303 and 404; model3000 was selected after the downstream comparison.

From the repository root, use `./scripts/play_mujoco.sh` or
`./scripts/play_isaac.sh`. After installing the pinned Isaac Lab overlay, run
`./scripts/07_train_g74.sh` to launch the scratch recipe (4096 environments,
6000 updates, seed 42). The original run used 500 Hz simulated PD/physics and a
50 Hz actor. The policy is in serial joint space; the historical training
envelope modeled ideal ankle actuator limits, not field calibration or MIT
wire quantization.

See the engineering handoff in [English](../../docs/SPRITE0825_STABLE_WALK_HANDOFF.en.md)
and [中文](../../docs/SPRITE0825_STABLE_WALK_HANDOFF.md). This package contains
no field calibration or ready-to-arm hardware configuration.
