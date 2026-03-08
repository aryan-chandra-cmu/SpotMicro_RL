from __future__ import annotations
import numpy as np
from gymnasium import spaces
from .mj_base_spotmicro_dr import SpotMicroMujocoBaseEnvDR
from gaits.spotmicro_gaits import trot_ctrl


class SpotMicroWalkMJ(SpotMicroMujocoBaseEnvDR):
    def __init__(self, xml_path: str, target_x_range=(0.8, 2.5), **kwargs):
        super().__init__(xml_path, episode_steps=2000, **kwargs)
        self.action_space = spaces.Box(
            low=np.array([-0.4, -0.4], dtype=np.float32),
            high=np.array([0.4, 0.4], dtype=np.float32),
        )

        self.target_x_range = target_x_range
        self._x0 = 0.0
        self._y0 = 0.0
        self.target_x = 0.0
        self.target_y = 0.0

        self.base_freq = 1.6
        self.base_hip_amp = 0.22
        self.base_knee_amp = 0.40

        self.alive_bonus = 0.0
        self.z_ref = 0.20

    def _reset_task(self):
        self._x0 = float(self.data.qpos[0])
        self._y0 = float(self.data.qpos[1])
        mag = float(self.np_random.uniform(*self.target_x_range))
        self.target_x = mag * (1.0 if self.np_random.random() < 0.8 else -1.0)

    def _ctrl_from_action(self, action: np.ndarray) -> np.ndarray:
        a0 = float(action[0])
        a1 = float(action[1])

        freq = np.clip(self.base_freq * (1.0 + 0.7 * a0), 0.6, 3.0)
        hip_amp = np.clip(self.base_hip_amp * (1.0 + 1.0 * a1), 0.03, 0.45)
        knee_amp = np.clip(self.base_knee_amp * (1.0 + 1.0 * a1), 0.03, 0.70)

        t = self._step_count * self.control_dt
        ctrl = trot_ctrl(
            stand_ctrl=self.stand_ctrl,
            t=t,
            freq_hz=float(freq),
            hip_amp=float(hip_amp),
            knee_amp=float(knee_amp),
            shoulder_amp=0.04,
            turn_bias=0.0,
        )
        return ctrl

    def _reward_done_info(self):
        x = float(self.data.qpos[0] - self._x0)
        forward = x if self.target_x >= 0 else -x
        reached = (abs(x) >= abs(self.target_x) - 0.15)

        roll, pitch, _ = self._base_rpy()
        stable = -(abs(float(roll)) + abs(float(pitch)))

        z = float(self.data.qpos[2])
        height_pen = -5.0 * max(0.0, (self.z_ref - z))

        energy = float(np.mean(np.square(self.data.ctrl - self.stand_ctrl)))

        reward = 1.5 * forward + 0.35 * stable - 0.02 * energy + self.alive_bonus + height_pen

        return reward, reached, {"x": x, "target_x": self.target_x, "forward": forward, "z": z}

    def render(self):
        if self.render_mode != "human":
            return

        if self._mj_viewer is None:
            from envs.mj_viewer import MJViewer
            self._mj_viewer = MJViewer(self.model, self.data)

        target_world_x = self._x0 + self.target_x
        target_world_y = self._y0
        target_world_z = 0.03

        self._mj_viewer.clear_user_geoms()
        self._mj_viewer.add_sphere_marker(
            pos=np.array([target_world_x, target_world_y, target_world_z], dtype=np.float64),
            radius=0.04,
            rgba=(1.0, 0.0, 0.0, 0.9),
        )
        self._mj_viewer.sync()