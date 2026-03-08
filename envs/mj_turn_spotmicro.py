from __future__ import annotations
import numpy as np
from gymnasium import spaces
from .mj_base_spotmicro_dr import SpotMicroMujocoBaseEnvDR
from gaits.spotmicro_gaits import trot_ctrl


def wrap_pi(a: float) -> float:
    return (a + np.pi) % (2*np.pi) - np.pi


class SpotMicroTurnMJ(SpotMicroMujocoBaseEnvDR):
    def __init__(self, xml_path: str, target_yaw_range=(0.6, 2.4), **kwargs):
        super().__init__(xml_path, episode_steps=1600, **kwargs)
        self.action_space = spaces.Box(
            low=np.array([-0.02, -0.4], dtype=np.float32),
            high=np.array([ 0.02,  0.4], dtype=np.float32),
        )
        self.target_yaw_range = target_yaw_range
        self.target_yaw = 0.0

        self.base_freq = 1.4
        self.base_hip_amp = 0.18
        self.base_knee_amp = 0.35

        self.alive_bonus = 0.5
        self.z_ref = 0.20

    def _reset_task(self):
        yaw0 = float(self._base_rpy()[2])
        delta = float(self.np_random.uniform(*self.target_yaw_range))
        delta *= (1.0 if self.np_random.random() < 0.5 else -1.0)
        self.target_yaw = wrap_pi(yaw0 + delta)

    def _ctrl_from_action(self, action: np.ndarray) -> np.ndarray:
        turn_bias_cmd = float(action[0])
        stride_scale = float(action[1])

        freq = np.clip(self.base_freq * (1.0 + 0.5 * stride_scale), 0.6, 2.8)
        hip_amp = np.clip(self.base_hip_amp * (1.0 + 0.8 * abs(stride_scale)), 0.03, 0.40)
        knee_amp = np.clip(self.base_knee_amp * (1.0 + 0.8 * abs(stride_scale)), 0.03, 0.70)

        t = self._step_count * self.control_dt
        ctrl = trot_ctrl(
            stand_ctrl=self.stand_ctrl,
            t=t,
            freq_hz=float(freq),
            hip_amp=float(hip_amp),
            knee_amp=float(knee_amp),
            shoulder_amp=0.05,
            turn_bias=float(np.clip(turn_bias_cmd * 0.8, -0.12, 0.12)),
        )
        return ctrl

    def _reward_done_info(self):
        yaw = float(self._base_rpy()[2])
        yaw_err = wrap_pi(self.target_yaw - yaw)
        reached = abs(yaw_err) < 0.10

        pos_pen = - (abs(float(self.data.qpos[0])) + abs(float(self.data.qpos[1])))

        roll, pitch, _ = self._base_rpy()
        stable = - (abs(float(roll)) + abs(float(pitch)))

        z = float(self.data.qpos[2])
        height_pen = -5.0 * max(0.0, (self.z_ref - z))

        reward = -abs(yaw_err) + 0.2 * pos_pen + 0.2 * stable + self.alive_bonus + height_pen

        return reward, reached, {"yaw": yaw, "target_yaw": self.target_yaw, "yaw_err": yaw_err, "z": z}