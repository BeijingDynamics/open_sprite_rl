# Sprite0825 G74 model 5999

This is the preserved Isaac-qualified G74 candidate using the v5 asset with four
J4340P shoulder motors. It is intentionally not the repository default: the
published package does not contain a completed final MuJoCo qualification
matrix. G60 model 3450 remains the current deployment baseline.

Isaac qualification requires the same checkpoint to pass independent seeds 303
and 404. `PACKAGE_READY_FOR_MUJOCO` means the fail-closed package preflight
passed; it does not mean the later MuJoCo promotion gate was completed.

- Policy: 50 Hz
- Simulation/servo integration: 500 Hz
- Motor inner loop target: 1 kHz Damiao MIT mode
- Actor observation dimension: 795, without horizontal base velocity or global pose
- Rollback baseline: frozen G59 model2999
- Checkpoint SHA256: bc8e84802703926366f6d7348c1da7365d4fbbe678297888443ed8949c3b7825

The original cloud-only paths were removed from the deploy contract and asset
metadata. Paths in historical evaluation evidence are provenance strings and
are not required for playback.
