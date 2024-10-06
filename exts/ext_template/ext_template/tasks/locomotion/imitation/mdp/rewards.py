from __future__ import annotations
import os

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.assets import Articulation

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
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, 12:], dim=1) > 0.1
    # reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def track_next_frame_vel(
        env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
        ) -> torch.Tensor:
    """ Simple L2 reward for tracking next frame motion """
    asset: Articulation = env.scene[asset_cfg.name]
    current_motion = asset.data.root_lin_vel_b[:, :2]
    # get next motion command
    next_vel_command = env.command_manager.get_command(command_name)[..., 12:14]
    reward = -torch.sum(torch.square(next_vel_command - current_motion), dim=1) / 0.25

    return torch.exp(reward) - 0.5

def track_next_frame_ang_vel(
        env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
        ) -> torch.Tensor:
    """ Simple L2 reward for tracking next frame motion """
    asset: Articulation = env.scene[asset_cfg.name]
    current_motion = asset.data.root_ang_vel_b[:, 2]
    # get next motion command
    next_ang_vel_command = env.command_manager.get_command(command_name)[..., 14]
    reward = -torch.square(next_ang_vel_command - current_motion) / 0.25

    return torch.exp(reward) - 0.5

def track_next_frame_joint(
        env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
        ) -> torch.Tensor:
    """ Simple L2 reward for tracking next frame motion """
    asset: Articulation = env.scene[asset_cfg.name]
    current_motion = asset.data.joint_pos[:, asset_cfg.joint_ids]
    # get next motion command
    next_joint_command = env.command_manager.get_command(command_name)[..., :12]
    # compute the difference between the current and the next frame motion
    reward = -torch.sum(torch.square(next_joint_command - current_motion), dim=1) / 0.25

    return torch.exp(reward) - 0.5
