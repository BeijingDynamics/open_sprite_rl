# Sprite0615 WBT V38 Stage 1 Reproduction Manifest

Date: 2026-08-21

## Qualified Policy

- Task:
  `Tracking-Flat-Sprite0615-WBT-Timescale150-PM01Rated-PhaseCycleB-Smooth015-VariableSpeed-FullPeakLegacyScale-AnkleTorque1e3-PushStage1-FeetSlide010-v0`
- Source run:
  `/root/sprite/sprite_isaaclab/IsaacLab/logs/rsl_rl/sprite0615_wbt_pm01_v38_push_feetslide010/2026-08-21_02-30-08_v38_push_feetslide010_from_v36m100_50_env8192_lr2e6`
- Qualified checkpoint: `model_2.pt`
- Checkpoint SHA-256:
  `b8d1851de07e654c367d0ad9d6d2e32a9fbc9588e4f26aabd7e5e4d0d71166ff`
- ONNX SHA-256:
  `a26c5b7f4dd902afb9aaa7eefc5f23567f824eff5b396a4ab6fd51ac6b45c98c`
- Warm start: immutable V36 model 100,
  `e78e32edbb8c1bfdb5188ae2943b507a1b7bb67d5d53c55bedb8bf26d4113c1c`.
- Training: 8192 environments, 50 iterations, learning rate `2e-6`, fresh
  optimizer, save interval 2. The selected model is the early model 2, chosen by
  measured transfer gates rather than training age.

## Isolated Delta

V38 preserves the V36 motion, 164-dimensional observations, 31-dimensional
actions, PM01-style FullPeak implicit actuator model, legacy safe action scale,
ankle torque penalty, and sparse half-envelope push curriculum. Its only reward
addition is Isaac Lab's standard `feet_slide` term on both foot links at
weight `-0.10`.

Configuration hashes:

- `flat_env_cfg.py`:
  `d08d81ce55e17eb9da53cb0b69b31dac621b03934f852298d5643b795d974cd4`
- Sprite task registry `__init__.py`:
  `748164415b668969d98bad6eaa23bb418f81979e7bc40954b92442b1c817d800`

## Isaac Qualification

The full matrix contains 30 valid cells: speed scales 0.8, 1.0, and 1.15;
seeds 7, 17, 29, 43, and 71; clean and full-push evaluation.

- Clean worst terminations per environment: 0
- Push worst terminations per environment: 0
- Clean body-position p90: 0.0436651
- Clean joint-position p90: 1.0232585
- Clean seam action p99: 1.7543049
- Clean ankle motor-ratio p90: 0.9888376
- J4340 rated-RMS maximum: 0.4107965
- Ankle rated-RMS maximum: 0.4848561
- Peak torque-speed p99 maximum: 0.4576236
- Every quality, contact, symmetry, continuity, and hardware gate: true

## Deploy Contract

The actor input is exactly 164 values in this order:

1. Reference joint position and velocity: 62
2. Pelvis-relative orientation 6D: 6
3. Base angular velocity: 3
4. Joint position: 31
5. Joint velocity: 31
6. Previous action: 31

The contract explicitly excludes horizontal base velocity and global
motion-anchor position. Action dimension is 31.

The qualified deployment initialization protocol is:

- PD scale: 1.0
- Handoff: 0.04 s
- Freeze reference clock during handoff
- Smoothstep from pose-equivalent action to learned policy output
- Re-anchor root XY when the policy clock begins
- Initial joint/root velocity injection: zero
- Gait cycle: 139 frames
- Phase starts used for evaluation: 0, 34, 69, 104

The canonical runner reads PD scale and handoff duration from the contract when
the CLI does not override them.

## MuJoCo Qualification

The corrected matrix includes three speeds, four gait phases, and one 60-second
loop. Results:

- Falls: 0
- Minimum root height: 0.438379 m
- Joint-position p90 worst: 1.087400
- Body-position p90 worst: 0.132887
- Root-XY p90 worst: 0.124773
- Action-rate p99 worst: 1.419116
- Joint rated-RMS maximum: 0.422301
- Ankle rated-RMS maximum: 0.529543
- Joint torque-speed envelope maximum: 0.600828
- Ankle torque-speed envelope maximum: 0.424283
- Frozen-baseline comparison: pass

A fresh contract-default smoke test used the archived model XML, applied the
two-policy-step handoff automatically, did not fall, and had minimum root height
0.440574 m and maximum joint torque-speed envelope 0.520095.

## Asset And Audit

The fresh V38 URDF/MJCF equivalence audit passed body/joint naming, axes, limits,
mass, inertia, geom transforms, geom sizes, friction, geom types, and body
mapping. Frozen V14A/V15A checkpoint hashes and read-only modes passed.

The Stage 1 completion audit reports:

- `stage1_sim2sim_pass=true`
- `sim2real_actuator_ready=true`
- frozen baseline hashes: pass
- asset equivalence: pass
- qualified candidate: pass

This means the policy/export/interface and validated motor envelopes are ready
for the later sim2real integration stage. It does not mean a physical robot
deployment has already been performed.

## Locations

Qualified read-only package on 234:

`/home/tony/sprite/sprite_isaaclab/IsaacLab/baselines/sprite0615_wbt_v38_model2_stage1_qualified`

Canonical runner:

`/home/tony/sprite/run_sprite0615_wbt_mujoco_sim2sim.py`

Playback:

```bash
cd /home/tony/sprite
./review_sprite0615_wbt_v38_stage1_qualified_on_234.sh 60 0
```

The first optional argument is duration in seconds; the second is the reference
start frame. The script plays the learned ONNX policy in MuJoCo, not reference
animation.

Use `sha256sum -c SHA256SUMS` for core artifacts and
`sha256sum -c DELIVERY_SHA256SUMS` for the entire package from inside the
qualified directory.

TWIST2 was not used or modified anywhere in this work.

