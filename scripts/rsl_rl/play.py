"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--ghost", type=bool, default=False, help="Have a ghost robot showing the motion data next to robot 0"
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

# Import extensions to set up environment tasks
import ext_template.tasks  # noqa: F401

from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_onnx


def load_motion() -> torch.Tensor():
    """Loads motion data in motion_data path to replay concurrently"""
    motion_data = []
    # only works with 1 motion file
    for file in os.listdir("motion_data"):
        if file.endswith(".pt"):
            motion_data = torch.load(os.path.join("motion_data", file))
    return motion_data


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    if args_cli.ghost:
        env_cfg = parse_env_cfg(
            args_cli.task, device=args_cli.device, num_envs=2, use_fabric=not args_cli.disable_fabric
        )
    else:
        env_cfg = parse_env_cfg(
            args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
        )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

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
    motion_step = 0
    # simulate environment
    motion_data = load_motion()
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # # move camera
            # eye = robot_pos + (5.0, 5.0, 5.0)
            # lookat = robot_pos
            # env.unwrapped.sim.set_camera_view(eye=eye, target=lookat)   # broken
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

            if args_cli.ghost:
                robot_pos = (
                    env.unwrapped.scene["robot"].data.root_pos_w[0, 0].item(),
                    env.unwrapped.scene["robot"].data.root_pos_w[0, 1].item(),
                    env.unwrapped.scene["robot"].data.root_pos_w[0, 2].item(),
                )
                robot_quat = (
                    env.unwrapped.scene["robot"].data.root_quat_w[0, 0].item(),
                    env.unwrapped.scene["robot"].data.root_quat_w[0, 1].item(),
                    env.unwrapped.scene["robot"].data.root_quat_w[0, 2].item(),
                    env.unwrapped.scene["robot"].data.root_quat_w[0, 3].item(),
                )
                # ghost the robot
                base_pos = motion_data[:, :, 0:3]  # base position in global frame
                base_quat = motion_data[:, :, 3:7]  # base orientation quaternion in global frame
                # projected_gravity = motion_data[:, :, 13:16]  # projected gravity onto base
                joint_angles = torch.cat(
                    (
                        motion_data[:, :, 16],
                        motion_data[:, :, 19],
                        motion_data[:, :, 22],
                        motion_data[:, :, 25],
                        motion_data[:, :, 17],
                        motion_data[:, :, 20],
                        motion_data[:, :, 23],
                        motion_data[:, :, 26],
                        motion_data[:, :, 18],
                        motion_data[:, :, 21],
                        motion_data[:, :, 24],
                        motion_data[:, :, 27],
                    )
                ).unsqueeze(
                    0
                )  # joint angles

                joint_vels = torch.zeros_like(joint_angles)  # joint positions don't seem to matter..?

                data_length = joint_angles.shape[2]

                robots = env.unwrapped.scene["robot"]  # ghost is robot 1
                root_state = robots.data.default_root_state.clone()  # [pos, quat, lin_vel, ang_vel]
                root_state[1, 0] = robot_pos[0] + 1.0
                root_state[1, 1] = robot_pos[1] + 1.0
                root_state[1, 2] = base_pos[:, motion_step % data_length, 2] + 0.05
                root_state[1, 3] = base_quat[:, motion_step % data_length, 2]
                root_state[1, 4] = base_quat[:, motion_step % data_length, 1]
                root_state[1, 5] = base_quat[:, motion_step % data_length, 0]
                root_state[1, 6] = base_quat[:, motion_step % data_length, 3]
                # print(root_state.shape)
                robots.write_root_state_to_sim(
                    root_state=root_state[1, ...].unsqueeze(0), env_ids=torch.tensor([1]).to(env.unwrapped.device)
                )

                # joint state
                joint_pos = joint_angles[:, :, motion_step % data_length]
                joint_pos = joint_pos.unsqueeze(0).to(env.unwrapped.device)
                joint_vel = joint_vels[:, :, motion_step % data_length]
                joint_vel = joint_vel.unsqueeze(0).to(env.unwrapped.device)

                # write joint angle and velocity information to simulation
                robots.write_joint_state_to_sim(
                    position=joint_pos, velocity=joint_vel, env_ids=torch.tensor([1]).to(env.unwrapped.device)
                )

                motion_step += 1

            # print(env_cfg.scene.robot.data.root_pos_w[0].detach().cpu())
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
