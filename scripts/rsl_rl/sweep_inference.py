# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to finetune RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Finetune an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during finetuning.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--run_num", type=int, default=None, help="Run number to run the sweep on.")
parser.add_argument("--num_runs", type=int, default=None, help="Number of runs to run the sweep on.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()


from sweep_inference_cfg import sweep_config, update_config_from_sweep
import wandb
import time

SWEEP_ID_FILE = "logs/rsl_rl/anymal_d_imitation/sweep_id.txt"


# overwrite args for cluster training
args_cli.headless = True
args_cli.task = "Isaac-Imitate-Anymal-D-Finetune-v0"
# args_cli.load_run = "2024-09-23_15-34-46"
args_cli.logger = "wandb"
run_num = args_cli.run_num
num_runs = args_cli.num_runs

if run_num == 0:
    sweep_id = wandb.sweep(sweep_config, project="isaaclab")
    # Save the sweep ID to a shared location
    with open(SWEEP_ID_FILE, "w") as f:
        f.write(sweep_id)
    exit()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from omni.isaac.lab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

import ext_template.tasks  # noqa: F401

from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
from omni.isaac.lab_tasks.utils import get_checkpoint_path
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

import re


def prep_sweep(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Finetune with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    # agent_cfg.max_iterations = (
    #     args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    # )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    
    wandb.init()
    env_cfg, agent_cfg = update_config_from_sweep(env_cfg, agent_cfg, wandb.config)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def run_sweep(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    # Wait for the sweep ID file to be available
    print("[Wandb] Waiting for sweep ID file")
    while not os.path.exists(SWEEP_ID_FILE):
        time.sleep(1)  # Wait until the file exists
    with open(SWEEP_ID_FILE, "r") as f:
        sweep_id = f.read().strip()
    wandb.agent(sweep_id, function=lambda: prep_sweep(env_cfg, agent_cfg), project="isaaclab", count=1)
    play_experiment(env_cfg, agent_cfg)
    simulation_app.close()


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def run(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Finetune with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    # agent_cfg.max_iterations = (
    #     args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    # )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    wandb.init(project="isaaclab", entity=os.environ["WANDB_USERNAME"])
    play_experiment(env_cfg, agent_cfg)
    simulation_app.close()


def play_experiment(env_cfg, agent_cfg):
    """Finetune with RSL-RL agent."""    
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    # find all runs in the directory that math the regex expression
    runs = [
        os.path.join(log_root_path, run) for run in os.scandir(log_root_path) if run.is_dir() and re.match(agent_cfg.load_run, run.name)
    ]
    # sort matched runs by alphabetical order (latest run should be last)
    runs.sort()
    # create last run file path
    num_total = len(runs)
    run_path = runs[num_total - num_runs + run_num - 1]

    # list all model checkpoints in the directory
    checkpoint = agent_cfg.load_checkpoint
    model_checkpoints = [f for f in os.listdir(run_path) if re.match(checkpoint, f)]
    # check if any checkpoints are present
    if len(model_checkpoints) == 0:
        raise ValueError(f"No checkpoints in the directory: '{run_path}' match '{checkpoint}'.")
    # sort alphabetically while ensuring that *_10 comes after *_9
    model_checkpoints.sort(key=lambda m: f"{m:0>15}")
    # get latest matched checkpoint file
    checkpoint_file = model_checkpoints[-1]
    resume_path = os.path.join(run_path, checkpoint_file)
    # resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during finetuning.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx
    # export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # export_policy_as_onnx(ppo_runner.alg.actor_critic, export_model_dir, filename="policy.onnx")

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    iter = 0
    wandb.init(project="isaaclab")
    while simulation_app.is_running() and iter <= 1000:
        iter += 1
        # run everything in inference mode
        with torch.inference_mode():
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

            log_info = env.unwrapped.extras['log']
            wandb.log(log_info)

            # print(env_cfg.scene.robot.data.root_pos_w[0].detach().cpu())
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
    wandb.finish()

    # close the simulator
    env.close()
    

if __name__ == "__main__":
    if run_num is None:
        run()
    else:
        run_sweep()