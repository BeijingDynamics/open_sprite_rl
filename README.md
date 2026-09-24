# Open Sprite RL

Reproducible whole-body locomotion training and Isaac Lab to MuJoCo transfer
for the 0.95 m, 31-DoF Beijing Dynamics Sprite0825 humanoid.

[中文说明](README.zh-CN.md)

## Current result

The current deployment baseline is G60 model 3450. It can stand, start, stop,
restart, walk at 0.15/0.30/0.45 m/s, and turn in both directions. The same ONNX
policy passed the Isaac Lab and MuJoCo gates. Its actor has 795 observations and
31 joint-target actions at 50 Hz, with physics and external PD at 500 Hz.

The actor deliberately excludes horizontal base velocity, global root position,
and global yaw. It uses eight frames of joint position, joint velocity, previous
action, base angular velocity, and projected gravity, plus the current
`vx/vy/yaw_rate` command. This is the policy currently used by the separate
`open_sprite_runtime` hardware-admission work; physical locomotion qualification
is still in progress and is not claimed by this repository.

Qualified checkpoint:

```text
baselines/sprite0825_stage2_g60_model3450_current/model_3450.pt
SHA256 6a1a80a2a2f7073698c0886133c325a46462ace6bbd3cf7de70246beafb85f15
```

## Method

This is the EngineAI PM01-style command-policy plus AMP pipeline adapted to
Sprite0825. Two reproducible lineages are retained:

1. **G59** trains the current actor from scratch at the deployable 50 Hz rate.
2. **G60** continues G59 model 2999 with physical-time-normalized action
   regularization and a soft waist-roll objective. Model 3450 is the selected
   current baseline.
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

The exact 4090 lineage is preserved. Each continuation defaults to the bundled,
hash-checked handoff checkpoint so every stage can be reproduced independently:

```bash
export ISAACLAB_ROOT="$HOME/IsaacLab"
export SPRITE_RL_ROOT="$PWD"

./scripts/05_train_g59.sh
./scripts/06_train_g60.sh
```

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

Set `CHECKPOINT=/path/to/model_N.pt` to review another G60 checkpoint.

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

## Safety and scope

This release contains training and sim2sim artifacts, not a ready-to-run
physical robot controller. Hardware execution belongs in `open_sprite_runtime`.
Verify motor directions, zero offsets, CAN IDs, joint limits, differential
kinematics, estimator conventions, emergency stop, and current/temperature
limits before hardware use.

TWIST2 was not used or modified.

## License

Original project code and Sprite assets are released under the GNU Affero
General Public License v3.0 only (`AGPL-3.0-only`). Third-party components and
PM01-derived expert data retain their upstream licenses and notices; see
`NOTICE` and `third_party_licenses/`.
