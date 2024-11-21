# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to finetune RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import multiprocessing
from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Finetune an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during finetuning.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--run_num", type=int, default=None, help="Run number for the experiment on the cluster.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# overwrite args for cluster training
args_cli.headless = True
args_cli.task = "Isaac-Imitate-Anymal-D-Finetune-v0"
args_cli.logger = "wandb"
run_num = args_cli.run_num


"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner


from sweep_cfg import sweep_config, update_config_from_sweep
import wandb
import time

SWEEP_ID_FILE = "logs/rsl_rl/anymal_d_imitation/sweep_id.txt"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


class PicklableWandbConfig:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError(f"'PicklableWandbConfig' object has no attribute '{name}'")

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


def wandb_agent_process(sweep_id, config_queue):
    def get_agent_config():
        wandb.init(project="isaaclab", entity=os.environ["WANDB_USERNAME"], sync_tensorboard=True)
        # Convert wandb.config to a PicklableWandbConfig object
        config_dict = dict(wandb.config)
        wandb_config = PicklableWandbConfig(config_dict)
        config_queue.put(wandb_config)
        # Wait for the main process to finish
        config_queue.get()

    wandb.agent(sweep_id, function=get_agent_config, project="isaaclab", count=1)

def run_sweep():
    if run_num == 0:
        sweep_id = wandb.sweep(sweep_config, project="isaaclab")
        # Save the sweep ID to a shared location
        with open(SWEEP_ID_FILE, "w") as f:
            f.write(sweep_id)
    else:
        # Wait for the sweep ID file to be available
        print("[Wandb] Waiting for sweep ID file")
        while not os.path.exists(SWEEP_ID_FILE):
            time.sleep(1)  # Wait until the file exists
        with open(SWEEP_ID_FILE, "r") as f:
            sweep_id = f.read().strip()
        
        config_queue = multiprocessing.Queue()
        wandb_process = multiprocessing.Process(target=wandb_agent_process, args=(sweep_id, config_queue))
        wandb_process.start()
        
        # Wait for the wandb config
        wandb_config = config_queue.get()
        
        # launch omniverse app
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app

        # Import extensions to set up environment tasks
        import ext_template.tasks  # noqa: F401

        from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
        from omni.isaac.lab.utils.dict import print_dict
        from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
        from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
        from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

        
        env_cfg = parse_env_cfg(
            args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
        )
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
        env_cfg, agent_cfg = update_config_from_sweep(env_cfg, agent_cfg, wandb_config)
        run_experiment(env_cfg, agent_cfg)
        
        # Signal the wandb process to finish
        config_queue.put(None)
        wandb_process.join()
        simulation_app.close()


def run():
    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Import extensions to set up environment tasks
    import ext_template.tasks  # noqa: F401

    from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
    from omni.isaac.lab.utils.dict import print_dict
    from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
    from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper


    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    wandb.init(project="isaaclab", entity=os.environ["WANDB_USERNAME"])
    run_experiment(env_cfg, agent_cfg)
    simulation_app.close()

def run_experiment(env_cfg, agent_cfg):
    """Finetune with RSL-RL agent."""
    # Import extensions to set up environment tasks

    import ext_template.tasks  # noqa: F401

    from omni.isaac.lab.utils.dict import print_dict
    from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

    
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during finetuning.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    # if agent_cfg.resume:
    #     # get path to previous checkpoint
    #     resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    #     print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    #     # load previously trained model
    #     runner.load(resume_path, load_optimizer=False, load_system_dynamics=False)

    # set seed of the environment
    env.seed(agent_cfg.seed)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run finetuning
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    if run_num is None:
        run()
    else:
        run_sweep()