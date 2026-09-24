# Sprite0825 G60 Model 3450

This directory is the current 50 Hz deployment baseline.

- Task: `Isaac-Sprite0825-Stage2-AMP-G60TimeNormalizedWaist50Hz-v0`
- Policy / physics rate: 50 / 500 Hz
- Actor: 795 observations, 31 joint-position targets
- Checkpoint SHA-256: `6a1a80a2a2f7073698c0886133c325a46462ace6bbd3cf7de70246beafb85f15`
- ONNX SHA-256: `43f213e4c5b9079e13b7f6f3635f224766227757417a0940ca49d585231016e3`

The actor contains no horizontal base velocity, global root position, or global
yaw. `SELECTION.md` records the deterministic, robust, heading-hold, and MuJoCo
gates. The retained evaluation set excludes repetitive logs and the uncompressed
60-second per-step trace; its compressed form is included.

This package is qualified for Isaac Lab and MuJoCo and is the input to the
separate physical-runtime admission process. It is not, by itself, permission
to operate hardware.
