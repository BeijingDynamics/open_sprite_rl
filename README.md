# Open Sprite RL

Reproducible whole-body locomotion training and Isaac Lab to MuJoCo transfer
for the 0.95 m, 31-DoF Beijing Dynamics Sprite0825 humanoid.

[中文说明](README.zh-CN.md)

## Qualified result

The release policy can stand, start, stop, restart, walk at 0.15/0.30/0.45
m/s, and turn at `+-0.20` and `+-0.40 rad/s`. The same ONNX policy passes the
Isaac Lab and MuJoCo gates. Its actor has 1,488 observations and 31 joint-target
actions at 100 Hz, with physics at 500 Hz.

The actor deliberately excludes horizontal base velocity, global root
position, and global yaw. It uses 15 frames of joint position, joint velocity,
previous action, base angular velocity, and projected gravity, plus the current
`vx/vy/yaw_rate` command.

Qualified checkpoint:

```text
baselines/sprite0825_stage2_g58f_model1050_stage2_qualified/model_1050.pt
SHA256 deda0cc80023c00e6fd7723830bf3a9b9ceb4b33e72c9e1e446d9b0873f1d912
```

## Method

This is the EngineAI PM01-style command-policy plus AMP pipeline adapted to
Sprite0825. It is not a collection of unrelated reward branches:

1. **G57** trains from scratch with a deployable command interface and an AMP
   prior made from 10% native stand and 90% accepted PM01 steady walk.
2. **G58A** continues G57 under Sprite's physical J4340P and differential
   J4310P torque-speed envelopes.
3. **G58B** restores the 0.45 m/s upper speed band without changing the expert.
4. **G58F** conservatively adds gentle yaw coverage while preserving the
   visually approved G58B model 925 gait.

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

./scripts/01_train_g57.sh
./scripts/02_train_g58a.sh
./scripts/03_train_g58b.sh
./scripts/04_train_g58f.sh
```

For a strict end-to-end run, qualify the selected checkpoint after each stage
and set `SOURCE_CHECKPOINT` to that result before starting the next script.
Reinforcement learning is stochastic: reproduce the acceptance envelope and
gait, not a bit-identical tensor file. The bundled handoff checkpoints make
exact continuation experiments possible.

Measured training time on the tested RTX 4090 was approximately 65.6, 17.0,
8.6, and 22.8 minutes respectively, before the multi-seed qualification runs.

Review the included qualified checkpoint in Isaac Lab with:

```bash
./scripts/play_isaac.sh
```

Set `CHECKPOINT=/path/to/model_N.pt` to review your own G58F checkpoint.

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
- `baselines/`: lineage checkpoints, qualified policy, contracts, and evidence
- `deploy/`: qualified Sprite0825 MuJoCo model
- `scripts/`: portable install, training, playback, and verification commands
- `scripts/provenance/`: untouched historical launch scripts for audit only
- `docs/`: architecture, data provenance, hardware limits, and acceptance gates

## Safety and scope

This release is simulation and sim2sim work. It is not a ready-to-run physical
robot controller. Verify motor directions, zero offsets, CAN IDs, joint limits,
estimator conventions, emergency stop, and current/temperature limits before
hardware use. The J4340P knee torque-speed p99 reached the configured simulation
boundary during fast walking/turning and must be monitored on hardware.

TWIST2 was not used or modified.

## License

Project code and Sprite assets are released under BSD-3-Clause. Third-party
components and PM01-derived expert data retain their upstream notices; see
`NOTICE` and `third_party_licenses/`.
