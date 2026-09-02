# Reproducibility Contract

## Version lock

Use these versions before interpreting a behavioral difference as a reward or
robot issue:

```text
Isaac Sim       5.1.0
Isaac Lab       b4c321024792976150ca55fddb26fa34480d974e
EngineAI AMP    83ba64bbb58a02e14483e52adce5f893f3f31cdf
rsl-rl-lib      5.0.1
torch           2.7.0+cu128
numpy training  1.26.0
Python          3.11
```

The tested cloud host used Ubuntu 24.04, driver 580.126.09, a 24 GB RTX 4090,
14 CPU cores, and 50 GB RAM.

## Artifact hashes

```text
G57 model500  c4bb13b271e840f6f9c182e5681c5ab7ebc1f48055406fb49bb1d6afda8e419d
G58A model799 47202164047b044e3e524cb65134b583725204057cae91fe41047583ba08a66b
G58B model925 e4d74619be6ea0786057ede910785bf58f8f92088f72c431578fdf56644523e9
G58F model1050 deda0cc80023c00e6fd7723830bf3a9b9ceb4b33e72c9e1e446d9b0873f1d912
Final ONNX     75a5c89552539da344e21566843f6c1fd1eac52e7601bbf8b5de64aaaae9eb26
MuJoCo scene   6ec42ff665fde05db58b507cc2747fb2a877e64efea003ff6b3d9e2f8c72e04f
```

Run `./scripts/verify_release.py` before training or playback.

## Deployment contract

- Actor observations: 1488
- Actions: 31 position targets
- Policy rate: 100 Hz
- Physics rate: 500 Hz
- History: 15 frames
- Commands: `vx`, reserved `vy`, `yaw_rate`
- Deployment handoff: 0.04 s
- Initial velocity injection: none
- Horizontal base velocity: absent
- Global root position: absent
- Global yaw: absent

The contract is in the final baseline's `deploy/contract.json`.

## Expected variation

Fresh PPO/AMP training is not bit deterministic across GPU kernels, driver
versions, environment counts, and process scheduling. Reproduction means:

- the same observation/action and physical contract,
- the same expert and task definitions,
- comparable command tracking and survival,
- no style/contact regression,
- successful Isaac to MuJoCo transfer.

It does not mean a fresh model must have the same SHA256 as the bundled model.
Exact continuation from a bundled handoff is much more controlled, but still
may not be bit-identical.

## Why checkpoints are bundled

The four handoff checkpoints separate code/data reproducibility from stochastic
checkpoint selection. A researcher can reproduce any stage independently,
compare a new run against the known input/output, and avoid spending hours
before discovering an earlier-stage mismatch.

## Path portability

Set `SPRITE_RL_ROOT` to this repository and `ISAACLAB_ROOT` to the pinned Isaac
Lab checkout. `SPRITE0825_ASSET_USD` and `SPRITE0825_G57_EXPERT` may override
individual paths. Historical files under `scripts/provenance/` intentionally
retain the original `/root/sprite` paths and are not launch entry points.
