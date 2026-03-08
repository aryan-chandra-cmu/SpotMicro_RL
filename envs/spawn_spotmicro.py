# scripts/spawn_spotmicro_pose.py
# DROP-IN FILE
#
# Spawns SpotMicro and HOLDS the given stand pose using position actuators.
# Also writes the pose into qpos so it truly "spawns" in that configuration.
#
# Run:
#   (mujoco) python .\scripts\spawn_spotmicro_pose.py

import os
import time
from typing import Dict

import mujoco
import mujoco.viewer


def build_actuator_for_joint_map(model: mujoco.MjModel) -> Dict[str, int]:
    """Map joint-name -> actuator index (works for joint-based actuators)."""
    m: Dict[str, int] = {}
    for a in range(model.nu):
        j_id = int(model.actuator_trnid[a, 0])
        if j_id >= 0:
            j_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
            if j_name:
                m[j_name] = a
    return m


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    xml_path = r"C:\Users\prome\robotics\spotmicro\assets\robot_base.xml"

    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"Could not find robot_base.xml at: {xml_path}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Your correct stand pose
    stand_targets = {
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

    # Reset sim state
    mujoco.mj_resetData(model, data)

    # Set base pose (freejoint)
    if model.nq >= 7:
        data.qpos[0:7] = [0.0, 0.0, 0.35, 1.0, 0.0, 0.0, 0.0]  # x,y,z,qw,qx,qy,qz

    # 1) Write pose into qpos (spawn configuration)
    for jname, q in stand_targets.items():
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if j_id < 0:
            raise RuntimeError(f"Joint not found in model: {jname}")

        jtype = int(model.jnt_type[j_id])
        qadr = int(model.jnt_qposadr[j_id])

        if jtype == mujoco.mjtJoint.mjJNT_HINGE or jtype == mujoco.mjtJoint.mjJNT_SLIDE:
            data.qpos[qadr] = float(q)
        else:
            raise RuntimeError(f"Joint '{jname}' is not a 1-DoF hinge/slide (type={jtype}).")

    # Zero velocities so it doesn't "kick" on spawn
    data.qvel[:] = 0.0

    # Forward once
    mujoco.mj_forward(model, data)

    # 2) ALSO set ctrl so actuators HOLD this pose (otherwise they pull to ctrl=0)
    act_for_joint = build_actuator_for_joint_map(model)

    # If there are actuators, hold pose; if not, still works just as spawn.
    has_actuators = model.nu > 0

    if has_actuators:
        for jname, q in stand_targets.items():
            if jname not in act_for_joint:
                raise RuntimeError(
                    f"No actuator controls joint '{jname}'. "
                    f"Your MJCF has actuators, but not for this joint."
                )
            data.ctrl[act_for_joint[jname]] = float(q)

        # One step to “lock in” actuator state before viewer
        mujoco.mj_step(model, data)

    # Viewer loop
    with mujoco.viewer.launch_passive(model, data) as viewer:
        target_hz = 60.0
        dt = model.opt.timestep
        steps_per_frame = max(1, int((1.0 / target_hz) / dt))

        while viewer.is_running():
            # Keep holding stand pose every frame
            if has_actuators:
                for jname, q in stand_targets.items():
                    data.ctrl[act_for_joint[jname]] = float(q)

            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)

            viewer.sync()
            time.sleep(0.001)


if __name__ == "__main__":
    main()