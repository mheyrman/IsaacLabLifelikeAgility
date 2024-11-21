from __future__ import annotations

import os
import torch
from typing import TYPE_CHECKING

import ext_template.tasks.locomotion.imitation.mdp as mdp

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

# from IsaacLabImitateAnymal.motion_data import motion_loader


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(mdp.generated_imitation_commands(env=env, command_name=command_name)[:, 12:], dim=1) > 0.1
    # reward *= torch.norm(mdp.generated_imitation_commands(env=env, command_name)[:, :2], dim=1) > 0.1
    return reward

def track_next_frame_feet(
        env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """L2 reward for tracking foot position relative to base"""
    asset: Articulation = env.scene[asset_cfg.name]
    # TODO: get foot position in body frame, will probably require transformation from world to body frame
    root_pos_w = asset.data.root_pos_w.unsqueeze(1)     # should return [4096, 1, 3]
    foot_pos_w = asset.data.body_state_w[:, 13:, :3]    # should return [4096, 4, 3]

    # subtract the 2nd dim of root_pos_w from foot_pos_w
    foot_pos_b = foot_pos_w - root_pos_w
    foot_pos_b = foot_pos_b.reshape(-1, 12)

    foot_pos_command = mdp.generated_imitation_commands(env=env, command_name=command_name, custom_motion=True)[..., :12]
    reward = -torch.norm(foot_pos_command - foot_pos_b, dim=1) / 2.0

    return torch.exp(reward)


def track_next_frame_vel(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Simple L2 reward for tracking next frame motion"""
    asset: Articulation = env.scene[asset_cfg.name]
    current_motion = asset.data.root_lin_vel_b

    # get next motion command
    next_vel_command = mdp.generated_imitation_commands(env=env, command_name=command_name)[..., 24:27]
    reward = -torch.norm((next_vel_command - current_motion), dim=1) / 0.75

    return torch.exp(reward)

def track_next_frame_ang_vel(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Simple L2 reward for tracking next frame motion"""
    asset: Articulation = env.scene[asset_cfg.name]
    current_motion = asset.data.root_ang_vel_b

    # get next motion command
    next_ang_vel_command = mdp.generated_imitation_commands(env=env, command_name=command_name)[..., 27:30]
    reward = -torch.norm(next_ang_vel_command - current_motion, dim=1) / 0.25

    return torch.exp(reward)


def track_next_frame_proj_grav(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Simple L2 reward for tracking next frame motion"""
    asset: Articulation = env.scene[asset_cfg.name]
    current_motion = asset.data.projected_gravity_b

    # get next motion command
    next_proj_grav_command = mdp.generated_imitation_commands(env=env, command_name=command_name)[..., 30:33]
    reward = -torch.sum(torch.square(next_proj_grav_command - current_motion), dim=1) / 0.01

    return torch.exp(reward)


def track_next_frame_joint(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Simple L2 reward for tracking next frame motion"""
    asset: Articulation = env.scene[asset_cfg.name]
    current_motion = asset.data.joint_pos[:, asset_cfg.joint_ids]

    # get next motion command
    next_joint_command = mdp.generated_imitation_commands(env=env, command_name=command_name)[..., :12]
    reward = -torch.sum(torch.square(next_joint_command - current_motion), dim=1) / 0.5

    return torch.exp(reward)


def track_next_frame_joint_vel(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Simple L2 reward for tracking next frame motion"""
    asset: Articulation = env.scene[asset_cfg.name]
    current_motion = asset.data.joint_vel[:, asset_cfg.joint_ids]

    # get next motion command
    next_joint_vel_command = mdp.generated_imitation_commands(env=env, command_name=command_name)[..., 12:24]
    # compute the difference between the current and the next frame motion
    reward = -torch.norm(next_joint_vel_command - current_motion, dim=1) / 0.75

    return torch.exp(reward)


def track_base_height(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Simple L2 reward for tracking next frame motion"""
    asset: Articulation = env.scene[asset_cfg.name]
    current_motion = asset.data.root_pos_w[..., 2]
    
    # get next motion command
    next_base_height_command = mdp.generated_imitation_commands(env=env, command_name=command_name)[..., 33]
    reward = -torch.square(next_base_height_command - current_motion) / 0.1

    return torch.exp(reward)