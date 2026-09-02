# Hardware and Deployment Boundary

## Actuator assumptions

The leg model uses DM-J4340P motors except for the two ankle axes on each leg,
which use DM-J4310P motors through a 1:1 differential linkage.

Configured reference envelopes:

- J4340P: 14 Nm rated, 40 Nm peak; 9-10 rad/s usable velocity range at the
  approximately 38 V system voltage.
- J4310P: 3.5 Nm rated, 12.5 Nm peak; 12.56 rad/s rated speed and 47.1 rad/s
  no-load maximum at 48 V.
- The ankle pitch/roll generalized torques are mapped through the differential;
  both motor loads must be reconstructed together rather than checking either
  generalized axis in isolation.

## Simulation interface

The policy emits 31 joint-position targets. Isaac Lab and MuJoCo both apply
joint-space impedance control at the physics rate. Real hardware is expected to
use the motors' MIT position/velocity/torque mode or an equivalent low-level
impedance loop.

The deployment actor requires one pelvis/torso IMU supplying angular velocity
and gravity direction, joint position/velocity, the previous policy action, and
the command. It does not require measured horizontal base velocity.

## Required real-robot integration work

Before energizing the robot, supply and verify:

1. Exact motor model and reduction for every joint.
2. Joint zero, sign, encoder sign, and CAN ID mapping.
3. Mechanical hard limits and software limits.
4. Rated/peak current and torque limits at the actual bus voltage.
5. MIT-mode gains and unit conversion.
6. IMU mounting transform and timestamp synchronization.
7. Watchdog, emergency stop, fall detection, and staged current limits.

## Known watch item

The simulated J4340P knee directional torque-speed p99 reaches the configured
envelope boundary during fast walking or turning. It passed sim2sim but has
little modeled margin. Log knee torque, velocity, current, winding temperature,
and bus voltage during hardware bring-up.

Foot-edge contact also consumes ankle roll authority that would otherwise be
available for pitch. Monitor both differential ankle motor loads and reject
persistent edge walking during real-robot tuning.

## Recommended bring-up order

1. Suspended joint-direction and limit test.
2. Low-gain stand with current limits.
3. Tethered start/stop at 0.15 m/s.
4. Straight 0.30 m/s walking.
5. Gentle `+-0.15 rad/s` yaw.
6. Only then test 0.45 m/s and larger yaw commands.
