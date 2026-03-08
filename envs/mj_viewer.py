from __future__ import annotations
import numpy as np
import mujoco
import mujoco.viewer


class MJViewer:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self._viewer = mujoco.viewer.launch_passive(model, data)

    def is_running(self) -> bool:
        return self._viewer.is_running()

    def clear_user_geoms(self):
        try:
            self._viewer.user_scn.ngeom = 0
        except Exception:
            pass

    def add_sphere_marker(
        self,
        pos,
        radius: float = 0.035,
        rgba=(1.0, 0.0, 0.0, 0.9),
    ):
        try:
            scn = self._viewer.user_scn
            if scn.ngeom >= scn.maxgeom:
                return

            geom = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([radius, radius, radius], dtype=np.float64),
                np.asarray(pos, dtype=np.float64),
                np.eye(3, dtype=np.float64).reshape(-1),
                np.asarray(rgba, dtype=np.float32),
            )
            scn.ngeom += 1
        except Exception:
            pass

    def sync(self):
        self._viewer.sync()

    def close(self):
        try:
            self._viewer.close()
        except Exception:
            pass