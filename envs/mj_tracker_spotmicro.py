from __future__ import annotations
import numpy as np
from gymnasium import spaces
from .mj_base_spotmicro_dr import SpotMicroMujocoBaseEnvDR
from gaits.spotmicro_gaits import trot_ctrl


def wrap_to_pi(x: float) -> float:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


class SpotMicroTrackerMJ(SpotMicroMujocoBaseEnvDR):
    def __init__(
        self,
        xml_path: str,
        target_radius_range=(0.8, 2.5),
        target_y_range=(-1.5, 1.5),
        goal_tolerance=0.18,
        success_bonus=150.0,
        **kwargs,
    ):
        super().__init__(xml_path, episode_steps=2200, **kwargs)

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(37,),
            dtype=np.float32,
        )

        self.target_radius_range = target_radius_range
        self.target_y_range = target_y_range
        self.goal_tolerance = float(goal_tolerance)
        self.success_bonus = float(success_bonus)

        self._x0 = 0.0
        self._y0 = 0.0

        self.target_x = 0.0
        self.target_y = 0.0

        self.prev_dist = 0.0
        self.prev_progress_along_ray = 0.0

        self.base_freq = 1.6
        self.base_hip_amp = 0.22
        self.base_knee_amp = 0.40
        self.base_shoulder_amp = 0.04

        self.z_ref = 0.20
        self.alive_bonus = 0.0

    def _world_xy(self) -> np.ndarray:
        return np.array([float(self.data.qpos[0]), float(self.data.qpos[1])], dtype=np.float32)

    def _target_world_xy(self) -> np.ndarray:
        return np.array([self._x0 + self.target_x, self._y0 + self.target_y], dtype=np.float32)

    def _goal_vec_world(self) -> np.ndarray:
        return self._target_world_xy() - self._world_xy()

    def _dist_to_goal(self) -> float:
        g = self._goal_vec_world()
        return float(np.linalg.norm(g))

    def _target_yaw(self) -> float:
        g = self._goal_vec_world()
        return float(np.arctan2(g[1], g[0]))

    def _heading_error(self) -> float:
        yaw = float(self._base_rpy()[2])
        return wrap_to_pi(self._target_yaw() - yaw)

    def _heading_correction(self) -> float:
        return float(np.clip(0.9 * self._heading_error(), -0.35, 0.35))

    def _progress_along_target_ray(self) -> float:
        start = np.array([self._x0, self._y0], dtype=np.float32)
        target = self._target_world_xy()
        robot = self._world_xy()

        ray = target - start
        ray_norm = float(np.linalg.norm(ray))
        if ray_norm < 1e-8:
            return 0.0

        ray_hat = ray / ray_norm
        disp = robot - start
        return float(np.dot(disp, ray_hat))

    def _body_frame_planar_velocity(self) -> tuple[float, float]:
        yaw = float(self._base_rpy()[2])
        vx_world = float(self.data.qvel[0])
        vy_world = float(self.data.qvel[1])

        c = np.cos(yaw)
        s = np.sin(yaw)

        vx_body = c * vx_world + s * vy_world
        vy_body = -s * vx_world + c * vy_world
        return float(vx_body), float(vy_body)

    def _obs(self) -> np.ndarray:
        base_obs = super()._obs()

        roll, pitch, yaw = self._base_rpy()
        goal = self._goal_vec_world()
        dx = float(goal[0])
        dy = float(goal[1])
        dist = float(np.linalg.norm(goal))
        heading_error = float(wrap_to_pi(np.arctan2(dy, dx) - yaw))

        vx_body, vy_body = self._body_frame_planar_velocity()
        yaw_rate = float(self.data.qvel[5])

        extra = np.array(
            [
                dx,
                dy,
                dist,
                np.cos(heading_error),
                np.sin(heading_error),
                vx_body,
                vy_body,
            ],
            dtype=np.float32,
        )

        obs = np.concatenate([base_obs, extra], axis=0)

        if self.enable_obs_noise:
            obs[-7:] += self.np_random.normal(0.0, 0.002, size=(7,)).astype(np.float32)

        return obs.astype(np.float32)

    def _reset_task(self):
        self._x0 = float(self.data.qpos[0])
        self._y0 = float(self.data.qpos[1])

        radius = float(self.np_random.uniform(*self.target_radius_range))
        theta = float(self.np_random.uniform(-np.pi, np.pi))

        tx = radius * np.cos(theta)
        ty = radius * np.sin(theta)

        ty += float(self.np_random.uniform(*self.target_y_range)) * 0.15

        self.target_x = float(tx)
        self.target_y = float(ty)

        self.prev_dist = self._dist_to_goal()
        self.prev_progress_along_ray = self._progress_along_target_ray()

    def _ctrl_from_action(self, action: np.ndarray) -> np.ndarray:
        a_freq = float(np.clip(action[0], -1.0, 1.0))
        a_stride = float(np.clip(action[1], -1.0, 1.0))
        a_turn = float(np.clip(action[2], -1.0, 1.0))

        freq = np.clip(self.base_freq * (1.0 + 0.45 * a_freq), 0.8, 2.8)
        hip_amp = np.clip(self.base_hip_amp * (1.0 + 0.80 * a_stride), 0.06, 0.42)
        knee_amp = np.clip(self.base_knee_amp * (1.0 + 0.80 * a_stride), 0.10, 0.72)

        helper_turn = self._heading_correction()
        learned_turn = 0.18 * a_turn
        turn_bias = np.clip(helper_turn + learned_turn, -0.40, 0.40)

        t = self._step_count * self.control_dt

        ctrl = trot_ctrl(
            stand_ctrl=self.stand_ctrl,
            t=t,
            freq_hz=float(freq),
            hip_amp=float(hip_amp),
            knee_amp=float(knee_amp),
            shoulder_amp=float(self.base_shoulder_amp),
            turn_bias=float(turn_bias),
        )
        return ctrl

    def _reward_done_info(self):
        dist = self._dist_to_goal()
        reached = bool(dist <= self.goal_tolerance)

        progress_along_ray = self._progress_along_target_ray()
        delta_progress_ray = progress_along_ray - self.prev_progress_along_ray

        delta_dist = self.prev_dist - dist

        heading_error = self._heading_error()
        heading_align = float(np.cos(heading_error))

        roll, pitch, _ = self._base_rpy()
        stable_term = -0.25 * (abs(float(roll)) + abs(float(pitch)))

        z = float(self.data.qpos[2])
        height_pen = -4.0 * max(0.0, self.z_ref - z)

        yaw_rate = abs(float(self.data.qvel[5]))
        spin_pen = -0.03 * yaw_rate

        energy = float(np.mean(np.square(self.data.ctrl - self.stand_ctrl)))
        energy_pen = -0.015 * energy

        time_pen = -0.01

        reward = 0.0
        reward += 8.0 * delta_progress_ray
        reward += 5.0 * delta_dist
        reward += 0.35 * heading_align
        reward += stable_term
        reward += height_pen
        reward += spin_pen
        reward += energy_pen
        reward += time_pen
        reward += self.alive_bonus

        if reached:
            reward += self.success_bonus

        self.prev_dist = dist
        self.prev_progress_along_ray = progress_along_ray

        info = {
            "goal_x": float(self._target_world_xy()[0]),
            "goal_y": float(self._target_world_xy()[1]),
            "dist_to_goal": float(dist),
            "heading_error": float(heading_error),
            "progress_along_ray": float(progress_along_ray),
            "delta_progress_ray": float(delta_progress_ray),
            "delta_dist": float(delta_dist),
            "reached": reached,
            "z": z,
        }

        return float(reward), reached, info

    def render(self):
        if self.render_mode != "human":
            return

        if self._mj_viewer is None:
            from envs.mj_viewer import MJViewer
            self._mj_viewer = MJViewer(self.model, self.data)

        target_world = self._target_world_xy()
        target_world_x = float(target_world[0])
        target_world_y = float(target_world[1])
        target_world_z = 0.03

        self._mj_viewer.clear_user_geoms()
        self._mj_viewer.add_sphere_marker(
            pos=np.array([target_world_x, target_world_y, target_world_z], dtype=np.float64),
            radius=0.05,
            rgba=(1.0, 0.0, 0.0, 0.95),
        )
        self._mj_viewer.sync()