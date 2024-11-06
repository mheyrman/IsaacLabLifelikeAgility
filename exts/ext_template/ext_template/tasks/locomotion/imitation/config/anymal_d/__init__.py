import gymnasium as gym

from . import agents, imitate_env_cfg

##
# Register Gym environments.
##
gym.register(
    id="Isaac-Imitate-Anymal-D-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": imitate_env_cfg.AnymalDImitateEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDImitatePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Imitate-Anymal-D-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": imitate_env_cfg.AnymalDImitateEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDImitatePPORunnerCfg",
    },
)
