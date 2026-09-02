# Training Pipeline

## Why this pipeline

The final Sprite0825 policy follows the architecture that produced the cleanest
PM01 behavior in EngineAI's open-source stack: a velocity-command task policy
regularized by an AMP discriminator trained on a selected locomotion expert.
The reference supplies movement style; the command reward supplies speed,
turning, standing, and recovery semantics.

The pipeline intentionally does not combine the historical V38 tracking policy
with the final G57 expert. G57 was trained from scratch using only accepted
native PM01 stand/walk clips.

## Stage table

| Stage | Start | Updates | Selected | Purpose | RTX 4090 time |
| --- | --- | ---: | --- | --- | ---: |
| G57 | scratch | 1000 | model 500 | command actor + native PM01 AMP gait | 3937 s |
| G58A | G57 model 500 | 300 | model 799 | physical motor torque-speed envelopes | 1019 s |
| G58B | G58A model 799 | 150 | model 925 | recover 0.45 m/s speed band | 519 s |
| G58F | G58B model 925 | 400 | model 1050 | gentle yaw without gait regression | 1371 s |

Iteration numbers continue across resumed stages. For example, G58A's 300
updates starting at model 500 end near model 799.

All reported runs used 8192 environments on a 24 GB RTX 4090. Reduce
`NUM_ENVS` if memory is insufficient, but expect different optimization noise
and retune the evaluation budget rather than assuming equal wall-clock time.

## G57: scratch locomotion

```bash
NUM_ENVS=8192 MAX_ITERATIONS=1000 ./scripts/01_train_g57.sh
```

Task:
`Isaac-Sprite0825-Stage2-AMP-G57NativePM01Forward100Hz-Robust-v0`

The expert directory contains one 400-frame stand clip and three identical
1200-frame accepted steady-walk clips. Sampling uses 10% stand and 90% walk.
This deliberate weighting lets zero command mean stand while preserving a
strong locomotion prior.

Do not automatically select the final G57 checkpoint. The qualified lineage
uses model 500 because it passed speed, stand, transition, visual, and contact
gates without later style drift.

## G58A: physical motor envelope

```bash
SOURCE_CHECKPOINT=/path/to/qualified/model_500.pt ./scripts/02_train_g58a.sh
```

Task:
`Isaac-Sprite0825-Stage2-AMP-G58FullMotorEnvelope100Hz-Robust-v0`

This stage keeps the policy, AMP data, command structure, and robot asset while
activating the physical J4340P and differential J4310P torque-speed envelopes.
The lineage checkpoint is model 799.

## G58B: high-speed recovery

```bash
SOURCE_CHECKPOINT=/path/to/qualified/model_799.pt ./scripts/03_train_g58b.sh
```

Task:
`Isaac-Sprite0825-Stage2-AMP-G58HighSpeedRecovery100Hz-Robust-v0`

The selected checkpoint is model 925. It restored 0.45 m/s command tracking
while preserving the controlled touchdown that was visually accepted.

## G58F: gentle yaw

```bash
SOURCE_CHECKPOINT=/path/to/qualified/model_925.pt ./scripts/04_train_g58f.sh
```

Task:
`Isaac-Sprite0825-Stage2-AMP-G58Preserve925GentleYaw100Hz-Robust-v0`

The stage expands training commands to `vx=(0.15, 0.45) m/s` and
`yaw_rate=(-0.20, 0.20) rad/s`. Model 1050 was the first checkpoint to pass the
complete numerical/style screen and was retained instead of later models.

## Acceptance gates

A checkpoint is promotable only after all of these categories pass:

1. Clean multi-seed survival and speed/yaw tracking.
2. PM01-style interval-push recovery.
3. 30-second zero-command stand with drift limits.
4. Twenty start/stop/restart cycles.
5. Style/contact regression against G58B model 925.
6. Physical motor RMS, peak, speed, and torque-speed checks.
7. MuJoCo straight, yaw, transition, and keyboard tests.
8. Human visual review for touchdown, edge contact, asymmetry, dragging,
   high-frequency swing motion, and gait rhythm.

The complete final evidence and evaluators are stored inside
`baselines/sprite0825_stage2_g58f_model1050_stage2_qualified`.

## Historical V38 baseline

V38 is a motion-tracking baseline with 164 observations and a fixed reference
clock. It is useful for studying strict whole-body tracking and sim2sim, but it
does not expose the final Stage 2 command interface. Its package and V36 warm
start are retained under `baselines/` for research comparison.
