from __future__ import annotations
import numpy as np


def trot_ctrl(
    stand_ctrl: np.ndarray,
    t: float,
    freq_hz: float,
    hip_amp: float,
    knee_amp: float,
    shoulder_amp: float = 0.0,
    turn_bias: float = 0.0,
) -> np.ndarray:
    phase = 2.0 * np.pi * freq_hz * t

    s_a = np.sin(phase)
    s_b = np.sin(phase + np.pi)

    sh = shoulder_amp * np.array([s_a, -s_a, s_b, -s_b], dtype=np.float32)

    hip = hip_amp * np.array(
        [s_a + turn_bias, s_b - turn_bias,
         s_b + turn_bias, s_a - turn_bias],
        dtype=np.float32
    )

    k_a = np.sin(phase + np.pi / 2)
    k_b = np.sin(phase + np.pi + np.pi / 2)
    knee = knee_amp * np.array([k_a, k_b, k_b, k_a], dtype=np.float32)

    ctrl = stand_ctrl.copy()
    ctrl[0:4] += sh
    ctrl[4:8] += hip
    ctrl[8:12] += knee
    return ctrl