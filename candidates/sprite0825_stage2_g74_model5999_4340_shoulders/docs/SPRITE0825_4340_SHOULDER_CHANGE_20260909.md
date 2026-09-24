# Sprite0825 DM-J4340P Shoulder Change

Date: 2026-09-09

## Decision

Upgrade the first two motors of each arm to DM-J4340P-2EC:

- `left_shoulder_pitch_joint`
- `left_shoulder_roll_joint`
- `right_shoulder_pitch_joint`
- `right_shoulder_roll_joint`

Shoulder yaw, elbow, and wrist yaw remain DM-J4310P-2EC. TWIST2 is not used.

## Confirmed limits

| Motor | Rated torque | Peak torque | Rated speed | 38 V no-load envelope |
|---|---:|---:|---:|---:|
| DM-J4340P-2EC | 14 Nm | 40 Nm | 3.77 rad/s | 9.3 rad/s |
| DM-J4310P-2EC | 3.5 Nm | 12.5 Nm | 12.56 rad/s | 36.2 rad/s |

## Asset isolation

The old `sprite0825_sanitized_v4` asset and G59-G70 policies are frozen as
rollback artifacts. The revised asset is:

`sprite0825_sanitized_v5_4340_shoulders`

Hashes:

- URDF: `6c753d563b54103e4278c27a11a22e0f4f1076e4156f0873cb2325f1701a921b`
- USD wrapper: `d21e1ceed89cabcf65ff3d73b7fee9da3640c2b97e0ac704261e52401b97af27`
- G71 reference: `97ec26f45460316e55845137b2adeb83958bd2676967af166085df085b509256`
- MuJoCo floating import URDF: `6614704c0ca0037b181a785092b01a92f4bbcc313c7b2b417946f1746f1c75b6`
- MuJoCo raw MJCF: `5a22c0f0178955e36e3f1e78eea4bf45fc58db8b9131ba5e1ba93d8e1576e32b`
- MuJoCo external-PD MJCF: `641b8199e9e24c059b1b0352e57e29128b44229a03832267eac3407beb174721`
- MuJoCo scene: `b4f6dd161f6af1d4fadb44da155d1ff3c319341b541aab99fc4c176c22a40d9d`

The automated asset audit confirms 31 joints and no changes to axes, origins,
parent/child topology, lower limits, upper limits, or joint order. Only the four
intended URDF effort/velocity fields changed to 14 Nm and 9.3 rad/s.

The isolated v5 MuJoCo model loads as 33 bodies, 32 joints (including the free
root), `nq=38`, and `nv=37`, with total mass `14.651386 kg`. Its array-level
comparison against frozen v4 confirms unchanged joint order, axes, ranges,
body transforms, masses, and inertias. Only shoulder pitch/roll reflected
armature changes from `0.0018` to `0.032`; distal arm joints stay at `0.0018`.
The scene uses 500 Hz physics, gravity `-9.81 m/s^2`, and no duplicate passive
damping.

### Mechanical-inertia boundary

The v5 change models the four new J4340P motors at the actuator level: effort
and speed envelopes, reflected rotor inertia, impedance gains, delay, and the
motor-safety gates are updated. The rigid-body masses, centers of mass, and
inertia tensors still come from the Sprite0825 URDF/CAD data supplied before
the motor replacement and are intentionally unchanged. They must not be
described as measured properties of the completed J4340P shoulder assemblies.

Before hardware commissioning, regenerate or measure the affected shoulder
assembly mass properties after the J4340P motors, brackets, and wiring are in
their final locations. Re-import both Isaac and MuJoCo assets, rerun the
asset/mapping audits, and requalify the frozen policy if the resulting mass,
center of mass, or inertia changes are material.

The public EngineAI PM01 tracking configuration was checked before deciding
whether to add an ad-hoc shoulder-link mass randomization. Its training events
randomize rigid-body material, joint default position, the base-link center of
mass, and external pushes; it does not randomize individual limb masses. G72
keeps that method boundary (with perturbation distances scaled by the
Sprite/PM01 height ratio) instead of introducing an unvalidated per-link mass
distribution mid-run. This does not remove the mechanical-inertia requirement
above: final CAD or measured shoulder-assembly properties remain mandatory,
followed by asset re-import and targeted policy requalification when material.

PM01 evidence inspected on 234:

- `engineai_rl_lab/tasks/tracking/tracking_env_cfg.py`, event definitions near
  `physics_material`, `add_joint_default_pos`, `base_com`, and `push_robot`.
- `engineai_rl_lab/tasks/tracking/config/pm01/flat_env_cfg.py`, where
  `base_com` is attached to `LINK_BASE`.
- Repository-wide search found no PM01 per-link body-mass randomization term.

## Isaac actuator contract

G71 uses separate actuator groups:

- `shoulder_pitch_roll_j4340`: the four upgraded joints, J4340P reflected
  inertia, PM01 10 Hz impedance derivation, 40 Nm peak envelope, 9.3 rad/s.
- `arms_distal_j4310`: shoulder yaw, elbows, and wrist yaw, with the existing
  J4310P model.

The proximal-shoulder action scale changes from about `0.439762082` to
`0.079157175`. Therefore old checkpoints cannot be continued as though the
action semantics were unchanged. G71 starts from scratch.

## 2026-09-11 G72 qualification finding

The completed 15,000-iteration G72 run exposed a static-contract gap. Its
serialized command-policy configuration still resolved the inherited PM01
generic shoulder action scale to `0.20 rad` for shoulder pitch and roll. The
G71 J4340P action scale (`0.07915717472057639 rad`) was imported by the module
but never assigned to G72's action term. The preflight checked motor torque,
speed, armature, timing, observation terms, and joint order, but did not check
the resolved action scale, so it could not catch this mismatch.

The failure is visible in independent dynamic metrics. All three diagnostic
G72 candidates survived and passed transition, yaw-response, overall tracking,
symmetry, spacing, and motor-envelope gates, but shoulder-pitch motion reached
roughly `2.1` to `2.5` times the reference range and shoulder tracking P90 was
`0.123` to `0.132 rad`. G72 remains immutable failure evidence; it is not
silently reinterpreted under corrected action semantics.

G73 is the isolated replacement. It retains the G72/PM01 command-policy method
and expert dataset, changes only the four proximal shoulder pitch/roll action
scales to `0.07915717472057639 rad`, leaves shoulder yaw and distal-arm action
semantics unchanged, and raises the existing PM01 feet-orientation reward from
`0.25` to `0.50`. The latter is evidence-based: the qualified G71 teacher has
about `0.024 rad` contact-foot-tilt P90, whereas G72 produced approximately
`0.065` to `0.076 rad` contact-foot-tilt mean. It is an adaptation of an
existing PM01 term for Sprite's rigid foot, not a new reward family.

The static preflight now rejects a generic shoulder action-scale pattern,
requires the exact proximal and shoulder-yaw scales, and requires the intended
feet-orientation weight before any G73 long run can start.

The independent seed-404 confirmation reproduced the same G72 failure on
`model_12500.pt`: survival remained `1.0` and both commanded yaw directions
completed without failure, but the serialized proximal-shoulder action scale
was still `0.2 rad`. Right shoulder-pitch temporal motion was `2.1193` times
the reference, while left/right contact-foot tilt means were
`0.06568/0.06338 rad`. This separates the action/foot-style contract error from
basic balance or turning ability and supports retraining G73 instead of tuning
around G72.

## Runtime contract validation

Validated independently on 234 and the rented RTX 4090:

- physics: 500 Hz (`dt=0.002`)
- policy: 50 Hz (`decimation=10`)
- motion source: 100 Hz, exactly two source frames per policy step
- actor observation: 164 dimensions
- no actor horizontal base velocity
- no actor global anchor position
- action dimension/order remains 31 joints

## Training

Cloud task:

`Tracking-Flat-Sprite0825-WBT-G71-IKReference-4340Shoulders-Clean-v0`

Experiment:

`sprite0825_wbt_g71_4340_shoulders_50hz`

Run:

`2026-09-09_18-31-16_g71_4340_fromscratch_15000_env4096`

The 4096-environment smoke test passed at about 40k steps/s. The 15,000
iteration from-scratch run was then started, with checkpoints directed to the
persistent `/root/gpufree-data` volume.

Post-training watcher PID `3645295` waits for training PID `3634996` to exit,
requires `model_14999.pt`, then evaluates models 2500, 5000, 7500, 10000,
12500, and 14999 under identical seed-42 conditions. It ranks stability and
motor safety before tracking/style metrics and writes:

`/root/gpufree-data/g71_4340_qualification/ranking.json`

The ranking output is also a hard qualification gate, not merely a
best-effort ordering. Thresholds were frozen before G71 final evaluations and
were anchored to the previously accepted WBT v15 tracking range:

- termination rate <= 0.01 per 1000 environment steps
- body position p90 <= 0.06 m and joint position p90 <= 1.10 rad
- left/right leg joint-error gap p90 <= 0.35 rad
- contact slip p90 <= 0.35 m/s and contact tilt p90 <= 0.30 rad
- all 12 J4340 joints (eight hip/knee plus four shoulder pitch/roll) have
  rated RMS, peak p99, speed p99, and directional torque-speed p99 ratios <= 1.0
- all four physical ankle J4310 motors have rated RMS, peak p99, speed p99,
  and directional torque-speed p99 ratios <= 1.0 after linkage load mapping

The evaluator must report exactly those 12 J4340 joint names and four physical
ankle motor names. Missing or extra names are schema failures, not zero load.

Each candidate is evaluated at fixed seeds 42 and 123. Qualification and
ranking use the worst value across both runs; a missing seed is an explicit
gate failure. The handoff watcher no longer depends on a historical
post-processing PID. It waits for the completed `ranking.json`, validates that
it is parseable and has `qualified=true` plus a non-null best checkpoint, and
only then starts G72.

The gate reports each failed value and limit. The G72 watcher requires
`qualified=true` and a non-null qualified best checkpoint; a parseable ranking
file alone cannot start G72. Synthetic pass/fail regression coverage is in
`test_rank_sprite0825_g71.py`.

## Remaining gates

- qualify intermediate/final G71 checkpoints in Isaac
- treat G71 only as the whole-body tracking teacher and hardware-feasibility
  gate; it is not a joystick/deployment policy
- train G72, the PM01-style 50 Hz command actor, from scratch with the G71
  motion as its unconditioned AMP expert prior
- qualify G72 stand/straight/left/right/stop/restart in Isaac and MuJoCo
- verify shoulder range and symmetry improve without degrading gait
- sync qualified package to 234 and Windows backup
- leave CAN IDs, zeros, encoder directions, linkage signs, MIT ranges, current
  limits, and thermal limits unset until measured on the assembled robot

## G72 command-policy preparation

G72 keeps the proven EngineAI PM01 method boundary: task commands and rewards
train a deployable command actor, while AMP supplies motion style. This is not
direct teacher action distillation. The actor remains free of horizontal base
velocity observations.

The isolated expert directory is:

`assets/sprite0825_references/stage2_amp/g72_g71_4340_shoulders_100hz_v1`

Its 31-joint order exactly matches G57, it runs at 100 Hz, and G59's 50 Hz AMP
loader uses `expert_frame_stride=2`. The reference moves forward 2.352 m in
11.99 s with mean body-frame forward velocity 0.196 m/s; it is not treadmill
motion. The accepted G57 quiet-stand clip supplies 10.0% stand support.

Task:

`Isaac-Sprite0825-Stage2-AMP-G72G71Expert4340Shoulders50Hz-Robust-v0`

Experiment:

`sprite0825_stage2_g72_g71_expert_4340_shoulders_50hz`

The 234 smoke test completed one PPO/AMP iteration with 16 environments. Its
serialized runtime configuration proves that it uses the v5 USD, 500 Hz
physics, 50 Hz policy, four J4340P shoulder pitch/roll joints at 40 Nm and
9.3 rad/s, and the G72 expert dataset. A first smoke exposed and fixed a G57
inheritance hook that had silently rewritten the USD path to frozen v4.

Before the cloud G72 long run, its launcher now performs a fresh one-iteration
16-environment smoke using the cloud source, evaluates `model_0.pt` to recover
the runtime observation/action and motor manifests, and runs
`verify_sprite0825_g72_static_contract.py`. The verifier currently covers 35
fixed checks, including 500/50 Hz timing, v5 asset revision, actuator groups,
31-joint action order, deployable actor observations, Unconditioned AMP data
flow, and differential-ankle torque mapping. The long run cannot start when
any check fails. Regression evidence on both 234 and cloud is `35/35` passing.

## Windows backup

The pre-training package is stored at:

`E:\sprite\rl\sprite0825_g71_4340_shoulders_20260909`

It contains the Isaac and MuJoCo v5 assets, qualified reference, source/config
deltas, validators, training and post-processing launchers, runtime hardware
manifest delta, and a verified `SHA256SUMS.txt`. Qualified checkpoints and
reports will be added after the cloud run completes.

## Runtime template root-cause fix

The initial runtime change correctly updated the checked-in Sprite0825
measurement worksheet, but an audit found that `hardware-template` still
pre-filled J4340P data only for hip/knee joints. Regenerating a worksheet from
a future frozen policy would therefore silently return all four upgraded
shoulder motors to unknown values. The inventory validator also pinned the
leg and ankle profiles but did not reject a legacy motor model on an upgraded
shoulder.

`open_sprite_runtime.hardware` now treats the four shoulder pitch/roll joints
as fixed J4340P profiles in both template generation and inventory validation.
Distal shoulder yaw, elbow, and wrist-yaw motors are deliberately unchanged.
Two regression tests prove that a newly generated template preserves the four
J4340P shoulders and that a J4310P substitution is rejected. The complete
runtime suite passes 48/48 tests on 234.

Tested worktree archive:

`/home/tony/sprite/backups/open_sprite_runtime_worktree_20260909.tar.gz`

Persistent cloud copy:

`/root/gpufree-data/runtime_backups/open_sprite_runtime_worktree_20260909.tar.gz`

SHA-256:

`9bef9f0a51a326b2d54d131adccb908289bf84478772a66ad6a65cfe962f9c1e`

## 2026-09-10 qualification and G72 handoff

G71 completed all 15000 iterations. The dual-seed hard-gate ranker qualified
all six screened checkpoints and selected `model_10000.pt` with the lowest
worst-seed score. It had zero terminations in both qualification seeds. Its
worst reported J4340P torque-speed ratio was 0.3554 and its worst physical
differential-ankle J4310P torque-speed ratio was 0.3625.

The first G72 handoff stopped before training because the detached launcher
had not activated the `isaaclab` conda environment. This was an orchestration
bug, not a task or policy failure. The launcher now sources conda and activates
`isaaclab` explicitly. The rerun passed all 35 static/runtime preflight checks
and started the 4096-environment, 15000-iteration G72 run at:

`/root/gpufree-data/logs/rsl_rl/sprite0825_stage2_g72_g71_expert_4340_shoulders_50hz/2026-09-10_12-30-37_g72_g71_expert_4340_shoulders_scratch_15000_env4096`

G72 completion is now staged. `ISAAC_QUALIFICATION_COMPLETE` means that a
checkpoint passed the automatic Isaac screens; it is deliberately not the
final deployment marker. The same checkpoint must then be exported, packaged
with the v5 asset, transferred to 234, and pass the MuJoCo matrix before final
qualification.

The deployment exporter and validator now enumerate twelve physical J4340P
joints: eight hip/knee joints plus left/right shoulder pitch and roll. The
MuJoCo runtime reads that list from the contract, so all four upgraded shoulder
motors are included in torque, speed, and directional torque-speed evidence.

The frozen v5 MuJoCo asset archive is present on 234, Windows, and cloud
persistent storage with SHA-256:

`144905fa52be096b3054cb5ab01ad9fa707d98d5f02cd439fc9ccdeab338dc51`

The G72 Isaac winner finalizer independently requires the exact twelve-joint
J4340P set and exact four-motor differential-ankle J4310P set. For each family
it gates rated-torque RMS, peak-torque p99, rated-speed RMS, maximum-speed p99,
and directional torque-speed p99 and absolute maximum. Four synthetic
regression tests pass, including missing-shoulder and p99-overload failures.
The finalizer also passed against the real G72 preflight evaluator JSON, which
contained all twelve J4340P joints and all four physical ankle motors.

The non-invasive cloud handoff audit
`verify_sprite0825_g72_cloud_pipeline.sh` checks the running experiment's
serialized configuration plus all launcher, evaluator, summary, finalizer,
reference, v5 asset, and package-builder dependencies. It passed `12/12` on
the live 4090 instance while G72 training continued. The authoritative
postprocessor is `/root/gpufree-data/postprocess_sprite0825_g72_cloud.sh`;
the training launcher invokes that exact persistent path and the postprocessor
then invokes the package builder under `g72_4340_setup`.

The G72 winner screen also has explicit arm-participation gates. The style
evaluator measures per-environment temporal joint motion before averaging, so
different reset phases cannot masquerade as arm swing. Both shoulder-pitch
joints must produce between 50% and 180% of their own G71-reference temporal
standard deviation, and the worst phase-invariant p90 tracking error across
the four upgraded shoulder pitch/roll joints must be at most `0.12 rad`.
These are intentionally reference-relative because the accepted G71 motion is
not perfectly equal-amplitude left/right (`0.055 rad` versus `0.026 rad`
shoulder-pitch standard deviation). A regression test proves that a candidate
with a stalled right shoulder is rejected. The exact evaluator, summarizer,
and regression test are included in the final candidate source snapshot.

The Isaac policy action order and MuJoCo's native joint order are not the same.
This is intentional and safe only because the runtime resolves every contract
joint through `mj_name2id` and builds explicit qpos/dof address arrays before
constructing observations or applying PD torque. The G72 export contract now
declares this mapping mode, and its validator rejects a contract that does not
require explicit name-based mapping.

The cloud-to-234 handoff verifies the cloud archive checksum before extraction,
rejects absolute or parent-traversal paths, rejects unsafe links, requires one
G72 candidate root, verifies package-internal hashes, and only then starts the
headless MuJoCo matrix. Five archive regression tests cover the accepted case,
parent traversal, unsafe symlink, multiple roots, and a non-G72 package name.

The first full G72 recovery point is `model_1000.pt`, SHA-256
`260b0aa19f89aaa243d9dc12e4c4ebf9cfdd722e5c7b8e92e2d4530f7033538a`.
The checkpoint and exact run `env.yaml`/`agent.yaml` are verified on cloud,
234, and Windows. This is a recovery point only, not a qualified candidate;
formal checkpoint screening still starts at iteration 2500.

The first formal screening checkpoint, `model_2500.pt`, was reached while the
same uninterrupted G72 process remained healthy. Its SHA-256 is:

`5ed457af07ee58b0b3ed086c01653565d0e07731662a6f669976a447f4b217b3`

The checkpoint, `params/env.yaml`, `params/agent.yaml`, portable checksum
manifest, and recovery command are verified on cloud persistent storage, 234,
and Windows. This is a reproducible formal candidate input, not yet a qualified
winner; Isaac screening remains deferred until the 15000-iteration training
process finishes so evaluation cannot contend with training for the GPU.

Formal checkpoint retention is now automatic for iterations `2500`, `5000`,
`7500`, `10000`, `12500`, and `14999`. The cloud archiver copies each complete
checkpoint with the exact run parameters and a relative-path SHA-256 manifest.
The independent 234 mirror downloads into a temporary directory, verifies the
manifest, and only then publishes the final directory. Interrupted or partial
copies are never promoted. At installation, the detached cloud archiver was
PID `4094283` and the detached 234 mirror was PID `66796`, both with PPID 1.

A packaging audit found that the live postprocessor correctly used the fixed
coarse gait ranker, but the final source snapshot did not copy that ranker or
its Isaac evaluator. The package builder now includes
`evaluate_sprite0615_stage2.py` and
`summarize_sprite0825_g72_gait_screen.py` alongside the detailed style and
multiobjective evaluators. A pure-Python regression test proves that a valid
gait is selected and an out-of-range cadence is rejected. The updated cloud
pipeline passes all 14 dependency, handoff, packaging, and regression checks
without loading Isaac Sim.

The detailed robust screen now repeats all transition, yaw, style, shoulder,
and motor-envelope gates with independent seeds `303` and `404`. The final
winner must belong to the intersection of the passing sets, preserving the
seed-303 ranking only within that intersection. Regression tests cover a
different winner after intersection, no common passing candidate, and an
invalid same-seed comparison. The candidate package validates the selected
iteration against `dualseed_selection.json` and includes both complete Isaac
evidence directories. The expanded cloud pipeline passes `16/16` checks.

The 234 MuJoCo launcher now performs a semantic validation before simulation:
the packaged checkpoint SHA must match the manifest, both manifest and
selection must name seeds `303` and `404`, the iteration must be the winner in
their passing intersection, both motor-finalized summaries must pass every
gate, and all transition/yaw/style JSON files must be present for both seeds.
Positive and deliberately single-seed package fixtures pass/fail as expected.
With the launcher itself promoted to a required dependency, cloud preflight is
now `19/19`.

The final cloud-to-MuJoCo handoff is also automatic. A detached watcher on 234
polls for a completed candidate archive and checksum, exits explicitly if
cloud qualification fails, downloads into a temporary directory, verifies the
archive, and then invokes the fail-closed installer and seven-case headless
MuJoCo matrix. It was installed as PID `70090` with PPID 1 and a 300-second
poll interval. The final frozen marker is written only after the MuJoCo matrix
passes. The watcher source is included in the candidate snapshot; the complete
pipeline now passes `20/20` checks.

The final visual-review entry point is
`/home/tony/sprite/review_sprite0825_g72_final_mujoco_on_234.sh`. It refuses
to auto-select unless exactly one package has passed the complete automatic
MuJoCo matrix and carries `SIM2REAL_CANDIDATE_FROZEN`. It then verifies the
package SHA-256 manifest and dual-seed Isaac evidence before opening the
interactive MuJoCo review. A pre-freeze negative test returned exit code `2`
as required, proving that the script cannot silently review an older or
unqualified checkpoint. Including this gate, cloud pipeline preflight passes
`21/21` checks.

The packaged MuJoCo XML retains URDF-origin `actuatorfrcrange` metadata, but
the parity runtime deliberately has `model.nu == 0`: there are no internal
MuJoCo actuators. It computes the 500 Hz external PD torque, clips all J4340P
legs and shoulder pitch/roll joints against the deployment contract's
`40 Nm` peak and `9.3 rad/s` no-load torque-speed envelope, and writes the
result through an explicit name-to-DOF map into `qfrc_applied`. The final
package validator now joins these facts instead of validating them in
isolation. It checks the exact 12-joint J4340P inventory, every corresponding
joint limit in the deploy contract, the audited runtime and MJCF hashes, zero
internal actuators, and the external-force address mapping. A negative fixture
that changes one shoulder back to a rated-only `14 Nm` limit is rejected.

The cloud package builder now runs this joined semantic validator before it
writes `PACKAGE_READY_FOR_MUJOCO`, and stores the result as
`evaluation/package_semantic_validation.json`. The 234 handoff independently
runs the same validator again before simulation. This makes a malformed or
stale package fail on the cloud and again at the simulator boundary.

The formal-checkpoint archive originally copied the exact `env.yaml`,
`agent.yaml`, and recovery command but hashed only the model file. This was a
real evidence gap. The cloud archiver now hashes every file in each formal
bundle, and the 234 mirror requires explicit manifest entries for the model,
both parameter files, and `RECOVERY.txt` before publication. `model_2500` was
regenerated and independently reverified on cloud, 234, and Windows with a
four-entry manifest. The corrected detached helpers are cloud PID `4129411`
and 234 PID `73120`; pipeline preflight now includes both helper sources and
passes `23/23` checks.

Formal-bundle verification is now implemented once in
`validate_g72_formal_checkpoint_bundle.py` and called by both the cloud
archiver and 234 mirror. It requires the manifest file set to equal the actual
bundle file set, requires the model, `env.yaml`, `agent.yaml`, and recovery
record, rejects unsafe paths and symlinks, and verifies every SHA-256. Tests
prove that a complete fixture passes while a model-only manifest and a
tampered environment configuration both fail. The corrected detached helper
PIDs are cloud `4134307` and 234 `73980`; expanded preflight passes `25/25`.

A restart audit then exposed that a valid formal bundle could be regenerated,
changing only `RECOVERY.txt` and leaving cloud and 234 with independently valid
but non-identical manifests. Formal publication now builds an invalid or new
bundle in a temporary directory, validates it, and atomically publishes it;
an already valid bundle is immutable across helper restarts. The mirror also
requires its local manifest SHA to equal the current cloud manifest SHA before
reporting `ALREADY_VERIFIED`. It detected and repaired the existing drift.
`model_2500` now has the identical manifest SHA
`2ddb56d99e805484d5e82ac7d34b46f51b7f91608a8a4cf07af44f8ae22ef031`
on cloud, 234, and Windows. The current helper PIDs are cloud `4137840` and 234
`74640`.
