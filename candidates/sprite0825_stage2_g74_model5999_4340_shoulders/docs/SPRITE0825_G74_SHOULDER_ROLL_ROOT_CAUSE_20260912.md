# Sprite0825 G74 Shoulder-Roll Root-Cause Note

Date: 2026-09-12

## Decision

G73 is not promoted. G74 is a fresh 6000-iteration run that changes only the
left/right shoulder-roll default, action-offset, and default-pose reward centers
to the qualified G71 walking-reference means:

- left shoulder roll: `+0.082816130 rad`
- right shoulder roll: `-0.076321766 rad`

The G73 action scales, J4340P shoulder actuator model, flat-foot reward, G71 AMP
expert, 50 Hz policy, 500 Hz simulation, observations, commands, randomization,
and every other reward remain unchanged.

## Evidence

Corrected G73 dual-seed evaluation included all twelve J4340P joints and all four
differential-ankle J4310P motors. Model2500 passed every fixed gate except
shoulder tracking P90:

| Metric | Seed 303 | Seed 404 | Gate |
| --- | ---: | ---: | ---: |
| survival | 1.000000 | 1.000000 | >= 0.99 |
| cadence/reference | 0.975517 | 0.981847 | 0.75 to 1.10 |
| joint tracking RMSE, rad | 0.049315 | 0.049215 | <= 0.06 |
| shoulder tracking P90 max, rad | 0.136743 | 0.136798 | <= 0.12 |
| J4340P directional envelope max | 1.0000001 | 1.0000000 | <= 1 + float epsilon |
| ankle J4310P directional envelope max | 0.909043 | 0.839569 | <= 1 + float epsilon |

Shoulder pitch was not the failing joint family. Its left/right P90 errors were
approximately `0.045/0.067 rad`. Both shoulder-roll joints had an approximately
constant error instead:

| Joint | Seed 303 mean abs, rad | Seed 303 P90, rad | Seed 404 P90, rad |
| --- | ---: | ---: | ---: |
| left shoulder roll | 0.104268 | 0.136412 | 0.136798 |
| right shoulder roll | 0.105795 | 0.136743 | 0.136072 |

The inherited robot default pose was `+0.25/-0.25 rad`. The G71 walk is centered
at `+0.082816/-0.076322 rad`, with small temporal standard deviations of
`0.019211/0.014328 rad`. The symmetric approximately `0.105 rad` residual is
therefore a control-center conflict, not evidence that the upgraded J4340P
motors lack torque.

The AMP expert also contains a 400-frame quiet-stand clip centered at
`+0.183163/-0.181501 rad` and three 1199-frame walk clips. Stand is about 10% of
expert frames. G74 deliberately uses the walk center because the failed metric
and primary locomotion behavior use the walk reference; the policy retains
enough action range to command the roughly 0.10 rad stand offset. Stand behavior
must still pass repeated transition tests before promotion.

## Coordinate Contract

- Actor observation: `joint_pos_rel`, eight frames.
- Joint action: position target with `use_default_offset=true`.
- Default-pose shoulder-roll reward: targets the robot default joint position.
- AMP policy frame: absolute joint position from `amp_frame`, scaled by 9.
- AMP expert clips: absolute 31-joint positions plus body-frame root velocity.

Changing the default center therefore aligns the actor action origin and task
reward without rewriting or offsetting the expert data. The actor input origin
also changes, which is why G74 trains from scratch instead of loading G73.

## Preflight And Runtime State

- Existing sim2real static contract: 42/42 passed.
- G74 shoulder-center contract: 4/4 passed.
- Actor dimension: 795; no horizontal base velocity or global pose/yaw.
- Action dimension/order: exact 31-joint contract.
- Formal training: 4096 environments, 6000 iterations, checkpoints at 250-step
  intervals; the mirror set matches all screened candidates exactly:
  500/1000/1500/2000/2500/3000/4000/5000/5999.
- Cloud training PID at launch: 948159.
- Cloud-to-234 mirror PID after the nine-checkpoint-set correction: 191606.
- 234-to-Windows WSL mirror PID after the same correction: 11506.
- Both restarted mirrors verified and skipped the already complete model500 and
  model1000 recovery directories before waiting for model1500.
- First formal recovery point: model500 is verified on cloud, 234, and Windows;
  checkpoint size is 16,430,987 bytes and SHA-256 is
  `40a8d10d4b181555211e752868795c128b040efe80c1ae6e8f33914285c3d0c6`.
  Both mirrored recovery directories pass all four internal hash checks.
- Second formal recovery point: model1000 is verified on cloud, 234, and
  Windows; checkpoint SHA-256 is
  `5b9a8acecf1fd1433ed2f2a3fdf9e203c8c738e8ade903297ba658d8a0f98899`.
  Both mirrored recovery directories pass all four internal hash checks.
- Terminal package/MuJoCo watcher PID at launch: 188258.

## Promotion Gates

Do not relax the existing `0.12 rad` shoulder tracking P90 gate. A candidate must
pass seed303 and seed404 transition, signed-yaw, cadence, torso, foot/contact,
shoulder motion/tracking, observation, J4340P, and differential-ankle gates.
Only then may it be packaged and run through the seven-case MuJoCo matrix.

## Controlled Iteration-1000 Comparison

G73 and G74 were compared at the same iteration and the same total step count
(`98,402,304`). G74 raised `arm_roll_position` from `0.2874` to `0.2997`
against its approximately `0.3` maximum and raised AMP style reward from
`0.0061` to `0.0067`, while timeout survival remained effectively identical
(`0.9952` versus `0.9951`). G74 horizontal velocity error was also slightly
lower (`0.4134` versus `0.4180`), but yaw error was higher (`1.2920` versus
`1.1431`). This is evidence that the intended shoulder-roll center correction
is active without a survival regression; it is not sufficient for promotion,
so the fixed dual-seed shoulder, gait, yaw, contact, and motor gates remain
unchanged.

## Model1500 early diagnostic

This is directional evidence only, not qualification evidence. The gait run used
30 seconds, 256 environments, deterministic seed43. The robust style run used
seed303 but only 2400 steps and 128 environments, versus the formal 4800-step,
256-environment seed303/404 protocol.

- Survival was `1.0` in both diagnostics and actor observations still excluded
  base linear velocity.
- Left/right hip, knee, and ankle gait frequency and range were closely matched;
  deterministic path speed was `0.2959 m/s` for a `0.30 m/s` command.
- Contact-foot roll P99 was `0.00684/0.00529 rad` left/right, with no samples over
  `0.06 rad` in the gait diagnostic.
- Shoulder tracking P90 max was `0.06413 rad`, already below the unchanged
  `0.12 rad` gate. Shoulder-roll P90 was `0.02409/0.01667 rad`, versus G73
  model2500 seed303 `0.13641/0.13674 rad`; the center correction is working.
- Left/right shoulder-pitch motion ratios were `0.347/0.733`. The left side is
  still below the fixed `0.50` lower gate.
- Cadence ratio was `1.466`, above the fixed `1.10` upper gate. Model1500 is too
  fast and is not eligible for promotion.
- Aligned J4340P/J4310P directional peak torque-speed maxima were
  `1.00000012/0.77289`; the J4340P value is at the modeled clipping boundary and
  within the existing float32 epsilon gate, but remains a final-screen watch item.

The source JSON files and SHA256SUMS are mirrored under `early_diagnostic` on
234 and Windows. Training therefore continues without changing rewards or gates;
later checkpoints must show cadence convergence and adequate bilateral shoulder
pitch motion before entering the dual-seed candidate set.

### Model2000 matched gait trend

A second deterministic gait diagnostic used exactly the model1500 gait protocol.
It showed a material convergence trend without changing configuration or gates:

| metric | model1500 | model2000 |
|---|---:|---:|
| survival rate | 1.0 | 1.0 |
| cadence, steps/min | 214.67 | 160.34 |
| estimated step length, m | 0.0827 | 0.1015 |
| path speed for 0.30 m/s command, m/s | 0.2959 | 0.2710 |
| mean absolute heading drift, deg | 15.35 | 9.44 |
| contact-foot roll P99 left/right, rad | 0.00684/0.00529 | 0.00801/0.01004 |

The reference touchdown interval (`0.405 s`) corresponds to about `148.15`
steps/min, so model2000 is much closer to the intended cadence than model1500.
The lower path speed and remaining heading drift still require the formal
transition/yaw/style screens. No additional concurrent diagnostic is scheduled;
the 4090 returns to training-only use until automatic postprocessing.

### Training-log trend through model2500

The on-policy training blocks at iterations 1500, 2000, and 2500 provide a
same-run trend check independent of the early evaluation scripts:

| metric | 1500 | 2000 | 2500 | 3000 |
|---|---:|---:|---:|---:|
| mean reward | 118.83 | 123.10 | 126.73 | 129.12 |
| illegal-contact termination fraction | 0.0131 | 0.0113 | 0.0020 | 0.0014 |
| timeout fraction | 0.9869 | 0.9887 | 0.9980 | 0.9986 |
| velocity-XY error | 0.3536 | 0.3531 | 0.3452 | 0.3299 |
| velocity-yaw error | 0.6585 | 0.6064 | 0.5710 | 0.5477 |
| arm-roll position reward | 0.2990 | 0.2996 | 0.2996 | 0.2991 |
| arm-pitch position reward | 0.0573 | 0.0576 | 0.0581 | 0.0602 |
| action-rate penalty | -0.3607 | -0.3522 | -0.3365 | -0.3156 |

The run is improving survival, yaw tracking, and action rate while preserving
the corrected shoulder-roll behavior. These training rewards do not replace the
fixed gait/style/transition gates, but they provide no evidence for interrupting
or retuning the run before the planned 6000 iterations.

## Training completion and recovery

The scratch run completed all 6000 planned iterations. The exact nine-point
recovery set (`500/1000/1500/2000/2500/3000/4000/5000/5999`) is present on the
cloud host, 234, and the Windows `E:` backup. Both downstream mirrors emitted
`CHECKPOINT_MIRROR_COMPLETE`, and the final four-file recovery manifest verifies.

- Final checkpoint: `model_5999.pt`
- SHA-256: `bc8e84802703926366f6d7348c1da7365d4fbbe678297888443ed8949c3b7825`
- Cloud postprocessing: active coarse screening, followed by unchanged fixed
  seed303/404 qualification for the top three candidates
- 234 package/MuJoCo watcher: active and fail-closed; it cannot promote a model
  until terminal Isaac qualification succeeds

## Seed303 formal screen

The fixed seed303 screen completed for coarse-ranked models 2500, 3000, and
5999. Model2500 failed only the unchanged shoulder-pitch motion-ratio gate: its
right side reached `1.83645x` against the `1.80x` limit. Models3000 and 5999
passed every transition, signed-yaw, survival, cadence, torso, foot/contact,
tracking, observation, J4340P, and differential-ankle gate. The seed303 ranking
selected model5999 first and model3000 second.

Model5999 seed303 highlights are: style survival `1.0`, cadence ratio `0.85008`,
joint tracking RMSE `0.04287 rad`, torso roll mean/P90 `0.08487/0.13341 rad`,
foot spacing `0.12314 m`, shoulder tracking P90 max `0.06983 rad`, and signed
yaw rates `-0.19638/+0.20480 rad/s` for `-0.20/+0.20 rad/s` commands. The
actor observation excludes base linear velocity. Seed404 confirmation is now
running with the same three candidates and unchanged gates.

No part of this work uses or depends on TWIST2.
