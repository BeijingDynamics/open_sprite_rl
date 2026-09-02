# Data Provenance

## Final G57 expert

The final locomotion prior is not LAFAN1, the earlier treadmill-like V38 clip,
or a mixture of experimental walk data. It is the accepted native PM01 steady
walk mapped to Sprite0825's 31-joint convention and canonically headed at
100 Hz.

The training directory is:

```text
assets/references/stage2_amp/g57_sprite0825_native_pm01_100hz_v1/
```

It contains:

- `00_stand.npz`: 400 frames
- `01_native_walk.npz`: 1200 frames
- `02_native_walk.npz`: 1200 frames
- `03_native_walk.npz`: 1200 frames

The three walk files intentionally repeat the accepted clip to implement the
90% walk / 10% stand sample weighting used by EngineAI-style AMP training.

The mapped source SHA256 is:

```text
bf4bd17fa4b91c8e560d9cc462e36be45c599fdf87a5dcd0181cc91b1d2d11f8
```

The phase source SHA256 is:

```text
26ac87cab3bc93fe6aaf9e4de027cdeb3f7e841672e08c1e5d35474f8f26195a
```

The complete joint order and per-file hashes are stored in the G57 baseline's
`expert_manifest.json`.

## License and attribution

The PM01 source stack is from EngineAI and is available under BSD-3-Clause.
The derived expert clips retain the EngineAI notice in this repository's
`NOTICE` and `third_party_licenses/ENGINEAI_BSD_3_CLAUSE.txt`.

## Why other motion files are absent

Many earlier motion candidates were useful diagnostics but did not define the
qualified final result. Excluding them makes it hard to accidentally train a
different task while believing it is G57. Add new motion sources as separate,
named experiments; do not silently replace the G57 directory.
