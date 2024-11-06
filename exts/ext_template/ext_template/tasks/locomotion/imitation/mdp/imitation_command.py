# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

import os
import torch
from collections.abc import Sequence
from tensordict.tensordict import TensorDict
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs.manager_based_rl_env import ManagerBasedRLEnv
from omni.isaac.lab.managers import CommandTerm
from omni.isaac.lab.managers.manager_term_cfg import CommandTermCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique

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

        # motion data
        motion_dict = {}
        motion_indices = []
        num_motions = 0
        for file in os.listdir("motion_data"):
            if file.endswith(".pt"):
                motion_data = torch.load(os.path.join("motion_data", file))

                joint_angles = torch.cat(
                    (
                        motion_data[..., 16],
                        motion_data[..., 19],
                        motion_data[..., 22],
                        motion_data[..., 25],
                        motion_data[..., 17],
                        motion_data[..., 20],
                        motion_data[..., 23],
                        motion_data[..., 26],
                        motion_data[..., 18],
                        motion_data[..., 21],
                        motion_data[..., 24],
                        motion_data[..., 27],
                    )
                )
                joint_velocities = torch.cat(
                    (
                        motion_data[..., 28],
                        motion_data[..., 31],
                        motion_data[..., 34],
                        motion_data[..., 37],
                        motion_data[..., 29],
                        motion_data[..., 32],
                        motion_data[..., 35],
                        motion_data[..., 38],
                        motion_data[..., 30],
                        motion_data[..., 33],
                        motion_data[..., 36],
                        motion_data[..., 39],
                    )
                )

                # vel x, vel y, ang vel z
                base_vel = torch.cat((motion_data[..., 7], motion_data[..., 8], motion_data[..., 9]))
                # base_vel_next is the next frame's base velocity
                base_vel_next = torch.cat([base_vel[:, 1:], base_vel[:, -1:]], dim=1)
                base_ang_vel = torch.cat((motion_data[..., 10], motion_data[..., 11], motion_data[..., 12]))
                base_ang_vel_next = torch.cat([base_ang_vel[:, 1:], base_ang_vel[:, -1:]], dim=1)
                base_proj_grav = torch.cat((motion_data[..., 13], motion_data[..., 14], motion_data[..., 15]))
                base_height = motion_data[..., 2] + 0.1

                # motion_data (34, n)
                motion_data = torch.cat(
                    (
                        joint_angles,
                        joint_velocities,
                        base_vel,
                        base_ang_vel,
                        base_proj_grav,
                        base_height,
                        base_vel_next,
                        base_ang_vel_next,
                    ),
                    dim=0,
                )

                motion_dict[file] = motion_data
                if motion_indices == []:
                    motion_indices.append(0)
                else:
                    motion_indices.append(motion_indices[num_motions - 1] + motion_data.shape[1])
                num_motions += 1
        motion_indices.append(motion_indices[num_motions - 1] + motion_data.shape[1])  # add terminal index

        # create buffers to store the command
        # -- command: all local frame data
        self.motion = torch.cat([m.to(self.device) for m in motion_dict.values()], dim=1)
        self.start_indices = torch.tensor(motion_indices, device=self.device)
        self.motion_index = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.motion_number = torch.randint(0, num_motions, (self.num_envs,), device=self.device)

        self.imitation_command = torch.zeros(self.num_envs, self.motion.shape[0], device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # -- metrics
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_base_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_base_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_base_proj_grav"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_base_height"] = torch.zeros(self.num_envs, device=self.device)

        self.indexing_dict = {
            "joint_angles_start": 0,
            "joint_angles_len": 12,
            "joint_velocities_start": 12,
            "joint_velocities_len": 12,
            "base_vel_start": 24,
            "base_vel_len": 3,
            "base_ang_vel_start": 27,
            "base_ang_vel_len": 3,
            "base_proj_grav_start": 30,
            "base_proj_grav_len": 3,
            "base_height_start": 33,
            "base_height_len": 1,
            "base_vel_next_start": 34,
            "base_vel_next_len": 3,
            "base_ang_vel_next_start": 37,
            "base_ang_vel_next_len": 3,
        }

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

        Shape: (num_envs, 33) 24 joint commands, 3 velocity commands, 3 ang vel command, 3 proj grav command
        """
        # print(self.imitation_command.shape)

        if not isinstance(self.cfg.terms, list):
            return self.imitation_command

        return_command = []
        for term in self.cfg.terms:
            return_command.append(
                self.imitation_command[
                    :,
                    self.indexing_dict[term + "_start"] : self.indexing_dict[term + "_start"]
                    + self.indexing_dict[term + "_len"],
                ]
            )
        return torch.cat(return_command, dim=1)

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
        self.metrics["error_joint_vel"] += (
            torch.norm(self.imitation_command[..., 12:24] - self.robot.data.joint_vel, dim=1) / max_command_step
        )
        self.metrics["error_base_vel"] += (
            torch.norm(self.imitation_command[..., 24:27] - self.robot.data.root_lin_vel_b, dim=1) / max_command_step
        )
        self.metrics["error_base_ang_vel"] += (
            torch.norm(self.imitation_command[..., 27:30] - self.robot.data.root_ang_vel_b, dim=1) / max_command_step
        )
        self.metrics["error_base_proj_grav"] += (
            torch.norm(self.imitation_command[..., 30:33] - self.robot.data.projected_gravity_b, dim=1)
            / max_command_step
        )
        self.metrics["error_base_height"] += (
            torch.square(self.imitation_command[..., 33] - self.robot.data.root_pos_w[..., 2]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        """
        Resample the imitation command.

        This function is called when the command needs to be resampled.
        """

        # change the motion sometimes
        # if torch.rand(1) < 5e-3:
        # print("Resampled")
        self.motion_number[env_ids] = torch.randint(0, len(self.start_indices) - 1, (len(env_ids),), device=self.device)
        # print(self.motion_number[env_ids])
        # make motion_index the index in self.start_indices where the motion starts
        self.motion_index[env_ids] = self.start_indices[self.motion_number[env_ids]].float()
        self.imitation_command[env_ids] = torch.transpose(
            torch.index_select(self.motion, 1, (self.motion_index[env_ids] // 1).type(torch.int32)), 0, 1
        )
        self.motion_index[env_ids] += 1.0

        # update standing envs
        r = torch.empty(len(env_ids), device=self.device)
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        """
        Post-process the imitation command.
        """
        self.imitation_command[...] = torch.transpose(
            torch.index_select(self.motion, 1, (self.motion_index // 1).type(torch.int32)), 0, 1
        )
        self.motion_index += 1.0
        # if self.motion_index[env] >= self.start_indices[self.motion_number[env] + 1] then return to the start of the motion
        self.motion_index = torch.where(
            self.motion_index >= self.start_indices[self.motion_number + 1],
            self.start_indices[self.motion_number],
            self.motion_index,
        )

        # enforce 0 velocity
        cmd_tmp = self.imitation_command.clone().detach()
        cmd_tmp[..., 12:30] = 0.0
        cmd_tmp[..., 33] = 0.6
        # enforce default joint positions
        cmd_tmp[..., :12] = self.robot.data.default_joint_pos

        # update the command
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.imitation_command[standing_env_ids, :] = cmd_tmp[standing_env_ids, :]
        # print(f"Execution time: {end_time - start_time} seconds")
