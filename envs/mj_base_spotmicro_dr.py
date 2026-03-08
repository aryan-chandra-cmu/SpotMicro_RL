from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


def quat_to_euler_xyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return np.array([roll, pitch, yaw], dtype=np.float32)


class SpotMicroMujocoBaseEnvDR(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        xml_path: str,
        render_mode: str | None = None,
        control_dt: float = 0.02,
        sim_substeps: int = 10,
        episode_steps: int = 1500,
        fall_roll: float = 0.9,
        fall_pitch: float = 0.9,
        min_base_z: float = 0.12,
        settle_steps: int = 80,
        enable_domain_randomization: bool = True,
        enable_obs_noise: bool = True,
        enable_act_noise: bool = True,
        enable_random_pushes: bool = True,
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self._mj_viewer = None

        self.control_dt = float(control_dt)
        self.sim_substeps = int(sim_substeps)
        self.sim_dt = self.control_dt / self.sim_substeps
        self.model.opt.timestep = self.sim_dt

        self.episode_steps = int(episode_steps)
        self._step_count = 0

        self.fall_roll = float(fall_roll)
        self.fall_pitch = float(fall_pitch)
        self.min_base_z = float(min_base_z)
        self.settle_steps = int(settle_steps)

        self.enable_domain_randomization = bool(enable_domain_randomization)
        self.enable_obs_noise = bool(enable_obs_noise)
        self.enable_act_noise = bool(enable_act_noise)
        self.enable_random_pushes = bool(enable_random_pushes)

        # Your requested reset pose
        self.stand_targets = {
            "front_left_shoulder":  0.0,
            "front_right_shoulder": 0.0,
            "rear_left_shoulder":   0.0,
            "rear_right_shoulder":  0.0,
            "front_left_leg":   0.85,
            "front_right_leg":  0.85,
            "rear_left_leg":    0.85,
            "rear_right_leg":   0.85,
            "front_left_foot":  -1.4,
            "front_right_foot": -1.4,
            "rear_left_foot":   -1.4,
            "rear_right_foot":  -1.4,
        }

        # ctrl vector built from actuator->joint mapping (robust to actuator ordering)
        self.stand_ctrl = self._ctrl_from_named_targets(self.stand_targets).astype(np.float32)

        # Expanded observation: [roll, pitch, wx, wy, z, vx] + 12 joint q + 12 joint qd = 30
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32)

        self.action_space = None  # subclasses set

        self._nominal = {
            "dof_damping": self.model.dof_damping.copy(),
            "dof_armature": self.model.dof_armature.copy(),
            "body_mass": self.model.body_mass.copy(),
            "body_inertia": self.model.body_inertia.copy(),
            "geom_friction": self.model.geom_friction.copy(),
            "geom_solref": self.model.geom_solref.copy(),
            "geom_solimp": self.model.geom_solimp.copy(),
            "actuator_gainprm": self.model.actuator_gainprm.copy(),
            "actuator_biasprm": self.model.actuator_biasprm.copy(),
            "actuator_forcerange": self.model.actuator_forcerange.copy(),
            "gravity": self.model.opt.gravity.copy(),
        }

        self._geom_floor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._body_base = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

        self._push_period_steps = max(1, int(1.0 / self.control_dt))
        self._push_prob = 0.25
        self._push_fx = 6.0
        self._push_fy = 6.0

        self.fall_penalty = 10.0

    def _ctrl_from_named_targets(self, targets: dict[str, float]) -> np.ndarray:
        ctrl = np.zeros(self.model.nu, dtype=np.float32)
        for a in range(self.model.nu):
            j_id = int(self.model.actuator_trnid[a, 0])
            j_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
            val = float(targets.get(j_name, 0.0))
            lo = float(self.model.actuator_ctrlrange[a, 0])
            hi = float(self.model.actuator_ctrlrange[a, 1])
            ctrl[a] = np.clip(val, lo, hi)
        return ctrl

    def _set_qpos_from_named_targets(self, targets: dict[str, float]) -> None:
        for j_name, val in targets.items():
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
            if j_id < 0:
                continue
            adr = int(self.model.jnt_qposadr[j_id])
            self.data.qpos[adr] = float(val)

    def _base_rpy(self) -> np.ndarray:
        q = self.data.qpos[3:7].copy()
        return quat_to_euler_xyz(q)

    def _obs(self) -> np.ndarray:
        rpy = self._base_rpy()
        w = self.data.qvel[3:6].copy().astype(np.float32)

        q = np.zeros(self.model.nu, dtype=np.float32)
        qd = np.zeros(self.model.nu, dtype=np.float32)
        for a in range(self.model.nu):
            j_id = int(self.model.actuator_trnid[a, 0])
            qadr = int(self.model.jnt_qposadr[j_id])
            vadr = int(self.model.jnt_dofadr[j_id])
            q[a] = float(self.data.qpos[qadr])
            qd[a] = float(self.data.qvel[vadr])

        z = float(self.data.qpos[2])
        vx = float(self.data.qvel[0])

        obs = np.concatenate(
            [np.array([rpy[0], rpy[1], w[0], w[1], z, vx], dtype=np.float32), q, qd],
            axis=0,
        )

        if self.enable_obs_noise:
            obs += self.np_random.normal(0.0, 0.005, size=obs.shape).astype(np.float32)

        return obs

    def _fallen(self) -> bool:
        rpy = self._base_rpy()
        if abs(float(rpy[0])) > self.fall_roll:
            return True
        if abs(float(rpy[1])) > self.fall_pitch:
            return True
        if float(self.data.qpos[2]) < self.min_base_z:
            return True
        return False

    def _restore_nominal(self):
        n = self._nominal
        self.model.dof_damping[:] = n["dof_damping"]
        self.model.dof_armature[:] = n["dof_armature"]
        self.model.body_mass[:] = n["body_mass"]
        self.model.body_inertia[:] = n["body_inertia"]
        self.model.geom_friction[:] = n["geom_friction"]
        self.model.geom_solref[:] = n["geom_solref"]
        self.model.geom_solimp[:] = n["geom_solimp"]
        self.model.actuator_gainprm[:] = n["actuator_gainprm"]
        self.model.actuator_biasprm[:] = n["actuator_biasprm"]
        self.model.actuator_forcerange[:] = n["actuator_forcerange"]
        self.model.opt.gravity[:] = n["gravity"]

    def _apply_domain_randomization(self):
        rng = self.np_random

        g0 = self._nominal["gravity"]
        g_scale = float(rng.uniform(0.98, 1.02))
        g_tilt_x = float(rng.uniform(-0.10, 0.10))
        g_tilt_y = float(rng.uniform(-0.10, 0.10))
        self.model.opt.gravity[:] = np.array([g_tilt_x, g_tilt_y, g0[2] * g_scale], dtype=np.float64)

        if self._geom_floor >= 0:
            mu = float(rng.uniform(0.8, 1.2))
            torsion = float(rng.uniform(0.002, 0.02))
            rolling = float(rng.uniform(0.0, 0.001))
            self.model.geom_friction[self._geom_floor, :] = np.array([mu, torsion, rolling], dtype=np.float64)

            self.model.geom_solref[self._geom_floor, 0] = float(rng.uniform(0.018, 0.030))
            self.model.geom_solref[self._geom_floor, 1] = float(rng.uniform(0.8, 1.2))

        mass_scale = float(rng.uniform(0.90, 1.10))
        self.model.body_mass[:] = self._nominal["body_mass"] * mass_scale
        self.model.body_inertia[:] = self._nominal["body_inertia"] * mass_scale

        if self._body_base >= 0:
            extra = float(rng.uniform(-0.15, 0.30))
            self.model.body_mass[self._body_base] = max(0.1, float(self.model.body_mass[self._body_base] + extra))

        self.model.dof_damping[:] = self._nominal["dof_damping"] * float(rng.uniform(0.85, 1.25))
        self.model.dof_armature[:] = self._nominal["dof_armature"] * float(rng.uniform(0.85, 1.25))

        kp_scale = float(rng.uniform(0.85, 1.20))
        self.model.actuator_gainprm[:] = self._nominal["actuator_gainprm"] * kp_scale
        self.model.actuator_biasprm[:] = self._nominal["actuator_biasprm"] * kp_scale

        tau_scale = float(rng.uniform(0.90, 1.10))
        self.model.actuator_forcerange[:] = self._nominal["actuator_forcerange"] * tau_scale

    def _maybe_random_push(self):
        if not self.enable_random_pushes:
            return
        if self._body_base < 0:
            return
        if (self._step_count % self._push_period_steps) != 0:
            return
        if float(self.np_random.random()) >= self._push_prob:
            return
        fx = float(self.np_random.uniform(-self._push_fx, self._push_fx))
        fy = float(self.np_random.uniform(-self._push_fy, self._push_fy))
        self.data.xfrc_applied[self._body_base, 0] = fx
        self.data.xfrc_applied[self._body_base, 1] = fy

    def render(self):
        if self.render_mode != "human":
            return
        if self._mj_viewer is None:
            from envs.mj_viewer import MJViewer
            self._mj_viewer = MJViewer(self.model, self.data)
        self._mj_viewer.sync()

    def close(self):
        if self._mj_viewer is not None:
            self._mj_viewer.close()
            self._mj_viewer = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self._restore_nominal()
        if self.enable_domain_randomization:
            self._apply_domain_randomization()

        self._set_qpos_from_named_targets(self.stand_targets)
        mujoco.mj_forward(self.model, self.data)

        self.data.ctrl[:] = self.stand_ctrl
        for _ in range(self.settle_steps):
            mujoco.mj_step(self.model, self.data)

        self._step_count = 0
        self._reset_task()
        return self._obs(), {}

    def _reset_task(self) -> None:
        pass

    def _ctrl_from_action(self, action: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _reward_done_info(self) -> tuple[float, bool, dict]:
        raise NotImplementedError

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        self.data.xfrc_applied[:] = 0.0
        self._maybe_random_push()

        ctrl = self._ctrl_from_action(action)

        if self.enable_act_noise:
            ctrl = ctrl + self.np_random.normal(0.0, 0.01, size=ctrl.shape).astype(np.float32)

        self.data.ctrl[:] = ctrl

        for _ in range(self.sim_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        obs = self._obs()
        reward, task_done, info = self._reward_done_info()

        fallen = self._fallen()
        if fallen:
            reward -= self.fall_penalty

        terminated = bool(task_done or fallen)
        truncated = bool(self._step_count >= self.episode_steps)

        return obs, float(reward), terminated, truncated, info