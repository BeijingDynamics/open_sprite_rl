# Open Sprite RL

Reproducible whole-body locomotion training and Isaac Lab to MuJoCo transfer
for the 0.95 m, 31-DoF Beijing Dynamics Sprite0825 humanoid.

[中文说明](README.zh-CN.md) · Stable walking handoff:
[English](docs/SPRITE0825_STABLE_WALK_HANDOFF.en.md) /
[中文](docs/SPRITE0825_STABLE_WALK_HANDOFF.md)

## Current result

The current baseline is [G74 model3000](baselines/sprite0825_stage2_g74_model3000/README.md),
the policy that walked stably on the four-J4340P-shoulder Sprite0825 before
reverse-walking research. It passed the Isaac Lab and MuJoCo stand, forward,
turn, and repeated start/stop gates. Supervised physical trials verified starts,
forward walking at a 0.10 m/s command, stopping, and low-speed gamepad turns.
Reverse and lateral walking are not qualified. The actor runs at 50 Hz with
795 observations and 31 joint targets; simulated physics/PD runs at 500 Hz.

The actor deliberately excludes horizontal base velocity, global root position,
and global yaw. It uses eight frames of joint position, joint velocity, previous
action, base angular velocity, and projected gravity, plus the current
`vx/vy/yaw_rate` command. See [Pinned hardware runtime](#pinned-hardware-runtime)
for the matching Jetson deployment source.

Qualified checkpoint:

```text
baselines/sprite0825_stage2_g74_model3000/model_3000.pt
SHA256 1b186bb10ee3cfa011d464e7c86b4f72b4c90a466c1576e3885cea5cd223f4f5
```

## Method

G74 is a fresh velocity-command PPO run with an unconditional AMP prior, not
SuperMimic or a continuation of G60. It uses the G71 contact-cleaned walk and
stand prior, the matching v5 asset, and corrected shoulder-roll centers.
Earlier lineages are retained:

1. **G59** trains an earlier actor from scratch at the deployable 50 Hz rate.
2. **G60** continues G59 model 2999 with physical-time-normalized action
   regularization and a soft waist-roll objective. Model 3450 is the selected
   historical v4 baseline.
3. **G57** is the historical 100 Hz scratch run with a deployable command
   interface and an AMP
   prior made from 10% native stand and 90% accepted PM01 steady walk.
4. **G58A** continues G57 under Sprite's physical J4340P and differential
   J4310P torque-speed envelopes.
5. **G58B** restores the 0.45 m/s upper speed band without changing the expert.
6. **G58F** conservatively adds gentle yaw coverage while preserving the
   visually approved G58B model 925 gait.

G59 is a fresh 50 Hz run, not a continuation of G58F. G60 continues G59.
The G74 package under `candidates/` preserves the later four-J4340P-shoulder
asset and model 5999, but remains a candidate rather than the default release.

The historical V38 whole-body tracking policy is included as a separate
motion-tracking baseline. It is not the parent of G57.

See [Training pipeline](docs/TRAINING_PIPELINE.md) and
[Reproducibility contract](docs/REPRODUCIBILITY.md).

## Tested stack

- Ubuntu 24.04
- NVIDIA RTX 4090 24 GB, 14 CPU cores, 50 GB RAM
- NVIDIA driver 580.126.09
- Isaac Sim 5.1.0
- Isaac Lab commit `b4c321024792976150ca55fddb26fa34480d974e`
- `rsl-rl-lib==5.0.1`
- EngineAI AMP commit `83ba64bbb58a02e14483e52adce5f893f3f31cdf`
- Python 3.11, PyTorch 2.7.0+cu128, NumPy 1.26.0 for training

## Install

Clone Isaac Lab and EngineAI AMP at the pinned commits, install them in the
same Isaac Sim Python environment, then install this repository's overlay:

On Windows hosts, enable long paths before cloning because the frozen audit
package retains its provenance tree: `git config --global core.longpaths true`.

```bash
git clone https://github.com/isaac-sim/IsaacLab.git "$HOME/IsaacLab"
git -C "$HOME/IsaacLab" checkout b4c321024792976150ca55fddb26fa34480d974e

git clone https://github.com/engineai-robotics/engineai_amp.git "$HOME/engineai_amp"
git -C "$HOME/engineai_amp" checkout 83ba64bbb58a02e14483e52adce5f893f3f31cdf

cd "$HOME/IsaacLab"
./isaaclab.sh --install rsl_rl
./isaaclab.sh -p -m pip install -e "$HOME/engineai_amp"

cd /path/to/open_sprite_rl
export ISAACLAB_ROOT="$HOME/IsaacLab"
./scripts/install_overlay.sh
./scripts/verify_release.py
sha256sum -c ARTIFACT_SHA256SUMS
```

Accept the NVIDIA Isaac Sim EULA before unattended training:

```bash
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=yes
```

## Reproduce training

To reproduce the G74 scratch recipe (4096 environments, 6000 updates):

```bash
export ISAACLAB_ROOT="$HOME/IsaacLab"
export SPRITE_RL_ROOT="$PWD"

./scripts/07_train_g74.sh
```

For the earlier 50 Hz lineage, run `05_train_g59.sh` and `06_train_g60.sh`.
The bundled G59 model 2999 is the exact G60 handoff. To reproduce the historical
100 Hz lineage instead, run `01_train_g57.sh` through `04_train_g58f.sh`.
Qualify the selected checkpoint after each stage and set `SOURCE_CHECKPOINT`
before the continuation when replacing a bundled handoff. Reinforcement
learning is stochastic: reproduce the acceptance envelope and gait, not a
bit-identical tensor file.

The historical G57-G58F stages took approximately 65.6, 17.0, 8.6, and 22.8
minutes on the tested RTX 4090, before multi-seed qualification. G59 is a much
longer 3000-update scratch run; plan capacity accordingly.

Review the included qualified checkpoint in Isaac Lab with:

```bash
./scripts/play_isaac.sh
```

Set `CHECKPOINT=/path/to/model_N.pt` to review another G74 checkpoint. Set
`BASELINE=g60` on either playback script to review the older G60 baseline.

## Play the qualified policy in MuJoCo

MuJoCo playback does not require Isaac Sim:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-mujoco.txt
./scripts/play_mujoco.sh 120
```

Keyboard controls are `Z` walk, `X` stop, `Q` turn left, and `E` turn right.
For a CI or server smoke test without a display, run
`VIEWER=0 ./scripts/play_mujoco.sh 10`.

## Repository layout

- `assets/`: Sprite0825 simulation asset and accepted AMP expert clips
- `isaaclab_overlay/`: custom Isaac Lab task and AMP implementation
- `baselines/`: lineage checkpoints, current policy, contracts, and evidence
- `candidates/`: preserved hardware-revision candidates that are not defaults
- `deploy/`: qualified Sprite0825 MuJoCo model
- `scripts/`: portable install, training, playback, and verification commands
- `scripts/provenance/`: untouched historical launch scripts for audit only
- `docs/`: architecture, data provenance, hardware limits, and acceptance gates

## Pinned hardware runtime

**G74 model3000 pairs with
[`open_sprite_runtime@4515206`](https://github.com/BeijingDynamics/open_sprite_runtime/tree/4515206c24404e18a2d5c168eccdb7468aea6ab7).**
This is the pre-reverse Jetson source snapshot for the ICM45686 800 Hz raw
six-axis pipeline and supervised Xbox forward, stop, and low-speed turning.
Use the pinned commit, not the runtime repository's moving `main` branch:

```bash
git clone https://github.com/BeijingDynamics/open_sprite_runtime.git
git -C open_sprite_runtime checkout --detach 4515206c24404e18a2d5c168eccdb7468aea6ab7
```

The RL repository's `runtime/` directories contain simulation tools. Neither
checkout is a ready-to-arm robot installation. Each robot and target Jetson
still needs:

- **Policy and control parameters:** the matching G74 model3000 ONNX and
  observation/action contract; a generated native parameter bundle with
  hardware-validated PD gains, command/feedback limits, and watchdog settings.
  Do not use the simulation contract's PD gains as motor commands.
- **31-motor inventory:** CAN FD interfaces, command and feedback IDs,
  encoder zeros and signs, direct-joint direction and ratios, measured
  position bounds, torque/speed/current/temperature limits, and each drive's
  read-back MIT PMAX/VMAX/TMAX.
- **Differential calibration:** separate measured 2×2 matrices, motor row
  order, offsets, and usable ranges for each ankle and head pitch/roll.
- **IMU and gamepad:** verify the pelvis mounting rotation, SPI and INT1 DRDY
  settings, and projected-gravity and gyro directions. Keep the robot still
  for the 15-second startup gyro-bias estimate. Pair the Xbox controller and
  verify its local axes, buttons, and deadman behavior.
- **Host build and qualification:** install the target ONNX Runtime library,
  build and qualify the native policy, CAN, and ICM executables, and regenerate
  machine-specific qualification seals. Verify real-time permissions, run
  zero-gain preflight, and confirm fall arrest and independent power cutoff
  before any separately approved powered test.

See the pinned [Xbox deployment notes](https://github.com/BeijingDynamics/open_sprite_runtime/blob/4515206c24404e18a2d5c168eccdb7468aea6ab7/docs/XBOX_DEPLOYMENT_20261019.md)
and [hardware contract](https://github.com/BeijingDynamics/open_sprite_runtime/blob/4515206c24404e18a2d5c168eccdb7468aea6ab7/docs/HARDWARE_CONTRACT.md).
The published snapshots do not include site calibration, field binaries, or
machine-specific qualification seals.

TWIST2 was not used or modified.

## License

Original project code and Sprite assets are released under the GNU Affero
General Public License v3.0 only (`AGPL-3.0-only`). Third-party components and
PM01-derived expert data retain their upstream licenses and notices; see
`NOTICE` and `third_party_licenses/`.
