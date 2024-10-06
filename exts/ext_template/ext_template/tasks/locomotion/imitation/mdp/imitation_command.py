# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

from omni.isaac.lab.envs.manager_based_rl_env import ManagerBasedRLEnv
from omni.isaac.lab.managers.manager_term_cfg import CommandTermCfg
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import CommandTerm
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique

import os

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

    from .commands_cfg import ImitationCommandCfg

class ImitationCommand(CommandTerm):
    """Command generator for generating imitation commands.
    
    The command generator generates joint positions and velocities to be
    imitated by the robot.
    """

    cfg: ImitationCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: CommandTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # create buffers to store the command
        # -- command: joint positions from motion data, x vel, y vel, z anguler vel
        self.imitation_command = torch.zeros(self.num_envs, 15, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # -- metrics
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_base_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_base_ang_vel"] = torch.zeros(self.num_envs, device=self.device)

        # -- data for visualization
        motion_path = os.path.join("motion_data", "motion_data_walk03.pt")
        data_path = os.path.abspath(motion_path)

        motion_data = torch.load(data_path)

        joint_angles = torch.cat((
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
                ))
        
        # vel x, vel y, ang vel z
        base_vel = torch.cat((motion_data[..., 7], motion_data[..., 8], torch.zeros_like(motion_data[..., 12], device=self.device)))
        # base_vel = torch.cat((motion_data[..., 7], motion_data[..., 8], motion_data[..., 12]))

        # self.motion_data (14, n)
        self.motion_data = torch.cat((joint_angles, base_vel), dim=0)
        self.data_index = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def __str__(self) -> str:
        msg = "ImitationCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg
    
    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """
        The desired robot information to imitate.
        
        Shape: (num_envs, 15) 12 joint commands, 2 velocity commands, 1 ang vel command
        """

        return self.imitation_command
    
    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # time when executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt

        # log data
        self.metrics["error_joint_pos"] += (
            torch.norm(self.imitation_command[..., :12] - self.robot.data.joint_pos, dim=1) / max_command_step
        )
        self.metrics["error_base_vel"] += (
            torch.norm(self.imitation_command[..., 12:14] - self.robot.data.root_lin_vel_b[:, :2], dim=1) / max_command_step
        )
        self.metrics["error_base_ang_vel"] += (
            (torch.square(self.imitation_command[..., 14] - self.robot.data.root_ang_vel_b[:, 2])) / max_command_step
        )


    def _resample_command(self, env_ids: Sequence[int]):
        """
        Resample the imitation command.

        This function is called when the command needs to be resampled.
        """
        self.data_index[env_ids] += 0.25

        for env_id in env_ids:
            self.imitation_command[env_id, :] = self.motion_data[..., int(self.data_index[env_id].item()) % 200]

        # update standing envs
        r = torch.empty(len(env_ids), device=self.device)
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        """
        Post-process the imitation command.

        This enforces 0 velocity and default joint positions for the joints of standing envs
        """
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()

        cmd_tmp = self.imitation_command.clone().detach()

        # enforce 0 velocity
        cmd_tmp[..., 12:] = 0.0
        # enforce default joint positions
        cmd_tmp[..., :12] = self.robot.data.default_joint_pos

        # update the command
        self.imitation_command[standing_env_ids, :] = cmd_tmp[standing_env_ids, :]

        self.data_index[standing_env_ids] = 0