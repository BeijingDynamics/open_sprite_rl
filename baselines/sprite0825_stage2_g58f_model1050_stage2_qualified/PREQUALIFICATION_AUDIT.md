# Sprite0825 Stage 2 Model 1050 Completion Audit

Audit status: **NOT COMPLETE - HUMAN VISUAL AND KEYBOARD GATES PENDING**

Candidate checkpoint SHA-256:
`deda0cc80023c00e6fd7723830bf3a9b9ceb4b33e72c9e1e446d9b0873f1d912`

This audit treats the current files on machine 234 as authoritative. It does
not infer completion from training intent or from a single green summary.

## Isaac Requirements

| Requirement | Status | Direct evidence |
| --- | --- | --- |
| Multi-seed clean termination rate 0 | PASS | Seeds 42-46 clean-grid survival are all 1.0. |
| Push termination rate <=1% | PASS WITH CONFIRMATION | The G58F Robust task inherits PM01's interval `push_by_setting_velocity` event every 1-3 s. The main robust-yaw matrix has 1 death in 480 envs (0.208%). A stricter per-seed screen flagged seed 44 at 1/96 (1.042%); an independent 320-env repeat recorded 2/320 (0.625%), below the Goal limit. Robust-gait has 0 deaths in 480 envs. The original `summary.json` remains unmodified and records its stricter initial failure. |
| Zero-command stand 30 s | PASS | Seeds 42-46 survival are all 1.0 and stand gate pass rate is 1.0. |
| Stand drift <=0.10 m and yaw <=5 deg | PASS | Worst maximum horizontal displacement is 0.01444 m; worst maximum final heading error is 0.2864 deg. |
| `vx=0.15/0.30/0.45 m/s`, MAE <=0.08 | PASS | Worst observed speed MAE is 0.07665 m/s. |
| `yaw_rate=+-0.20/+-0.40 rad/s`, MAE <=0.12 | PASS | Seeds 42-46 full-yaw runs have 100% survival. Worst observed yaw MAE is 0.03356 rad/s. |
| At least 20 start/stop/restart cycles | PASS | 64 envs complete all 20 cycles with 100% survival; median start latency 0.12 s and stop latency 0.60 s. |
| Model-925 style regression <=10% | NUMERICAL PASS, VISUAL PENDING | The preserved raw comparison JSON passes all ten gates. Closest gate is right-foot contact tilt at 1.09905x against the 1.10x limit; raw-action second delta is 1.07234x. Tony's controlled-touchdown gate remains authoritative. |
| Actor excludes horizontal base velocity | PASS | Localized deployment contract says false. The 1,488-D actor terms are 15-frame joint position, joint velocity, previous action, base angular velocity, projected gravity histories, plus the current 3-D velocity command. |

Raw style evidence archive SHA-256:
`aa884a32f41d35f02d966cb6239c2104f47eeb3aa293c9661fe70a1e9fd85510`.
The model-1050 versus model-925 gate JSON SHA-256 is
`5b69f1b563b3dbeb40ff2fe23947fe9fe332c72723df5389aa7a6d9ecf69ca8e`.

## MuJoCo Requirements

| Requirement | Status | Direct evidence |
| --- | --- | --- |
| Same observation/action/PD/limits contract | PASS | Portable contract resolves the same 1,488 observations, 31 actions, 0.01 s policy step, 0.002 s physics step, action scale, PD and physical motor envelopes. ONNX parity max absolute error is 7.153e-7. |
| Actor excludes horizontal base velocity | PASS | Contract assertion and runner observation construction both exclude it. |
| 60 s continuous straight walk | PASS AUTOMATIC | `straight_60s.json` survived; forward MAE is 0.01694 m/s. |
| At least 20 walk/stop cycles | PASS AUTOMATIC; KEYBOARD PENDING | Automated 20-cycle case survived and completed all cycles. Repeated human `Z/X` acceptance remains pending. |
| Left/right turns usable and natural | NUMERICAL PASS; VISUAL PENDING | `+-0.20` and `+-0.40 rad/s` cases all survived with yaw MAE below 0.05 rad/s. Human review must reject pivoting, reverse weight shift, swing-leg chatter, or failed unloading. |
| No persistent foot-edge walking | PENDING HUMAN VISUAL | Contact telemetry alone does not prove acceptable sole use. |
| Motor RMS/peak/speed/torque-speed limits | PASS SIMULATION | MuJoCo worst J4340P ratios: 0.387/0.641/0.932/1.000. Worst differential J4310P ratios: 0.431/0.374/0.202/0.374. Knee torque-speed has little remaining simulated margin and remains a hardware watch item. |
| Keyboard `Z/X/Q/E` interface | IMPLEMENTED AND SMOKE-TESTED; ACCEPTANCE PENDING | Hash-pinned script exists and a 3 s headless load/start/walk smoke survived with zero self-contact. |

MuJoCo matrix summary SHA-256:
`abb02c63e2b6c176ae2be54042af69a01575dc46759129f22d147810493a55b2`.

## Deliverables

| Deliverable | Status |
| --- | --- |
| Best checkpoint and SHA-256 | PRESENT AS CANDIDATE |
| Complete training scripts and configuration | PRESENT: full AMP source package, launcher, and resolved Hydra YAML snapshot |
| Isaac automatic evaluation matrix | PRESENT |
| MuJoCo automatic evaluation | PRESENT |
| MuJoCo keyboard review script | PRESENT; human acceptance pending |
| Itemized observation contract | PRESENT: six terms covering all 1,488 dimensions |
| Portable SHA-256 manifest | PARTIAL: immutable artifacts and evidence have verified hashes; final package-wide manifest waits for acceptance |
| Technical decisions and failed branches | PRESENT in the Stage 2 status history and candidate README |
| Final result document | PENDING visual/keyboard decision |
| Read-only qualified package | PENDING; candidate must not be frozen or renamed early |

A guarded promotion script is prepared at
`/home/tony/sprite/sprite_isaaclab/IsaacLab/promote_sprite0825_g58f_model1050_stage2.sh`
with SHA-256
`ad144400941cbb6f27468b21672ce17d4cf25d575c1555fd0b6d30377f8916a5`.
It requires separate explicit Isaac-visual and MuJoCo-keyboard approval files,
then rechecks checkpoint/artifact hashes and every machine-readable style,
motor, yaw, transition, contract, and keyboard gate. It only creates a new
qualified directory, generates and verifies a package-wide manifest, and then
removes all write permissions. Its current expected-refusal test exits with
code 4 (`Tony Isaac visual approval missing`) and creates no target directory.
The future qualified package uses a separate README template with SHA-256
`2c7c9e3efab9270b1278ae9d907a0445ab91249594eb2de5a50a836d9f31a498`
instead of mislabeling the candidate README. Its embedded final verifier has
SHA-256
`7647e2fbbbc5a05a1c2d2c6ffd04811e513aa1120972eab001831c4ed834cda9`
and rechecks the package-wide manifest, read-only permissions, approvals,
contract and every acceptance gate from inside the frozen package.

## Remaining Completion Path

1. Tony compares model 925 and model 1050 in the fixed-side Isaac A/B/C review.
2. If model 1050 preserves controlled touchdown and natural gait, Tony runs the
   MuJoCo `Z/X/Q/E` review for repeated starts, stops, restarts, and both turns.
3. If either visual gate fails, model 1050 remains evidence only and model 925
   remains the visual baseline; do not weaken the gate.
4. If both pass, record Tony's decision, remove superseded candidate-only files,
   generate and verify a package-wide manifest, rename the package to qualified,
   make it read-only, and run the final requirement audit again.
