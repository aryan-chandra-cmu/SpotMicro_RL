from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from envs.mj_tracker_spotmicro import SpotMicroTrackerMJ

XML = r"C:\Users\prome\robotics\spotmicro\assets\robot_base.xml"


def make_env(enable_dr: bool):
    def _init():
        env = SpotMicroTrackerMJ(
                xml_path=XML,
                render_mode=None,
                control_dt=0.02,
                sim_substeps=10,
                enable_domain_randomization=False,
                enable_obs_noise=False,
                enable_act_noise=False,
                enable_random_pushes=False,
                settle_steps=120,
            )
        return Monitor(env)
    return _init


if __name__ == "__main__":
    ENABLE_DR = False  # set True after it learns basics
    N_ENVS = 8

    env = SubprocVecEnv([make_env(ENABLE_DR) for _ in range(N_ENVS)])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.03,
        tensorboard_log="./tb_waypoint/",
        verbose=1,
    )

    model.learn(total_timesteps=5_000_000)
    model.save("spotmicro_tracker_ppo")