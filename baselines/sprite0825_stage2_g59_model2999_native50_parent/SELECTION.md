# Sprite0825 G59 Native 50 Hz Selection

Status: selected Isaac Lab candidate; MuJoCo and hardware qualification pending.

- Task: `Isaac-Sprite0825-Stage2-AMP-G59Native50Hz-v0`
- Checkpoint: `model_2999.pt`
- SHA256: `eb84e5aedb8ace647d877788a98c86db6ce9670e4c003890e6e1d2ef21205a98`
- Policy rate: 50 Hz
- Physics rate: 500 Hz
- Actor observation: 795 dimensions, eight history frames
- AMP observation: 102 dimensions, three frames

## Selection rationale

Model 2800 and model 2999 were both visually acceptable with no important
visible regression. Model 2999 looked slightly faster and had the best final
mean reward and forward-velocity metric. Model 2800 had slightly better AMP
style and yaw metrics, so it remains the fallback candidate.

Final training metrics for model 2999:

- mean reward: 134.6105
- mean episode length: 1000.0
- horizontal velocity error: 0.2865
- yaw velocity error: 0.4069
- illegal-contact termination: 0.000122

Do not call this checkpoint deployment-qualified until it passes the 50 Hz
ONNX export, MuJoCo sim2sim command sequence, torque/speed audit, and runtime
timing gates.
