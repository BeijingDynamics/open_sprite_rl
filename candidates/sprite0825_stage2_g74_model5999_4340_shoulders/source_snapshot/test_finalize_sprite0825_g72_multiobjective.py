import unittest

from finalize_sprite0825_g72_multiobjective import (
    EXPECTED_ANKLE,
    EXPECTED_J4340,
    motor_gates,
)


def ratios(names, override=None):
    result = {}
    for metric in (
        "rated_torque",
        "peak_torque",
        "rated_speed",
        "max_speed",
        "peak_torque_speed",
    ):
        result[metric] = {
            name: {"rms": 0.2, "p99": 0.3, "max": 0.4} for name in names
        }
    if override:
        metric, name, field, value = override
        result[metric][name][field] = value
    return result


class MotorContractTest(unittest.TestCase):
    def test_valid_j4340_set_passes(self):
        _, gates = motor_gates(
            {"motors": ratios(EXPECTED_J4340)},
            "motors",
            EXPECTED_J4340,
            "j4340",
        )
        self.assertTrue(all(gates.values()))

    def test_missing_shoulder_fails_exact_names(self):
        names = EXPECTED_J4340 - {"right_shoulder_roll_joint"}
        _, gates = motor_gates(
            {"motors": ratios(names)}, "motors", EXPECTED_J4340, "j4340"
        )
        self.assertFalse(gates["j4340_exact_motor_names"])

    def test_peak_p99_overload_fails(self):
        name = sorted(EXPECTED_J4340)[0]
        _, gates = motor_gates(
            {"motors": ratios(EXPECTED_J4340, ("peak_torque", name, "p99", 1.01))},
            "motors",
            EXPECTED_J4340,
            "j4340",
        )
        self.assertFalse(gates["j4340_peak_torque_p99_le_1"])

    def test_valid_differential_ankle_set_passes(self):
        _, gates = motor_gates(
            {"motors": ratios(EXPECTED_ANKLE)},
            "motors",
            EXPECTED_ANKLE,
            "ankle_j4310",
        )
        self.assertTrue(all(gates.values()))


if __name__ == "__main__":
    unittest.main()
