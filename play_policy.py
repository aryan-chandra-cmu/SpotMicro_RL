import time
from stable_baselines3 import PPO

from envs.mj_tracker_spotmicro import SpotMicroTrackerMJ

XML = r"C:\Users\prome\robotics\spotmicro\assets\robot_base.xml"
MODEL_PATH = "spotmicro_tracker_ppo.zip"


def main():
    env = SpotMicroTrackerMJ(
        xml_path=XML,
        render_mode="human",
        control_dt=0.02,
        sim_substeps=10,
        enable_domain_randomization=False,
        enable_obs_noise=False,
        enable_act_noise=False,
        enable_random_pushes=False,
        settle_steps=120,
    )

    model = PPO.load(MODEL_PATH, device="cpu")

    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(env.control_dt)

        if terminated or truncated:
            obs, _ = env.reset()


if __name__ == "__main__":
    main()