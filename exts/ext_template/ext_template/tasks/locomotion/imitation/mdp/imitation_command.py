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

        self.cfg = cfg
        self.env = env

        self.robot: Articulation = env.scene[cfg.asset_name]

        # motion data
        motion_dict = {}
        motion_indices = []
        num_motions = 0
        motion_dir = self.cfg.motion_dir if isinstance(self.cfg.motion_dir, str) else "motion_data_train"
        motion_dir = os.path.join("motion_data", motion_dir)
        for file in os.listdir(motion_dir):
            if file.endswith(".pt") and not file.endswith("end_points.pt"):
                motion_data = torch.load(os.path.join(motion_dir, file))

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
                base_ang_vel = torch.cat((motion_data[..., 10], motion_data[..., 11], motion_data[..., 12]))
                base_proj_grav = torch.cat((motion_data[..., 13], motion_data[..., 14], motion_data[..., 15]))
                base_height = motion_data[..., 2] + 0.1

                # end points
                end_point_data = torch.load(os.path.join(motion_dir, file[:-3] + "_end_points.pt")).to(motion_data.device)
                end_points = torch.cat(
                    (
                        end_point_data[..., 0],
                        end_point_data[..., 1],
                        end_point_data[..., 2],
                        end_point_data[..., 3],
                        end_point_data[..., 4],
                        end_point_data[..., 5],
                        end_point_data[..., 6],
                        end_point_data[..., 7],
                        end_point_data[..., 8],
                        end_point_data[..., 9],
                        end_point_data[..., 10],
                        end_point_data[..., 11],
                    )
                )
                # end_points_next = torch.cat([end_points[:, 1:], end_points[:, -1:]], dim=1)
                # end_points_nnext = torch.cat([end_points_next[:, 1:], end_points_next[:, -1:]], dim=1)

                motion_data = torch.cat(
                    (
                        joint_angles,
                        joint_velocities,
                        base_vel,
                        base_ang_vel,
                        base_proj_grav,
                        base_height,
                        end_points,
                        # end_points_next,
                        # end_points_nnext,
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
        self.custom_imitation_command = torch.zeros(self.num_envs, self.motion.shape[0], device=self.device)
        self.custom_len = 0
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # -- history buffer
        self.hist_len = self.cfg.history_horizon if isinstance(self.cfg.history_horizon, int) else 0
        # self.num_prop_obs = 45
        self.history_buffer = torch.zeros(
            # self.num_envs, self.hist_len, self.num_prop_obs + self.custom_len, device=self.device
            self.num_envs, self.hist_len, self.custom_len, device=self.device
        )

        # -- metrics
        self.metrics["error_foot_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_base_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_base_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_base_proj_grav"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_base_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_imitation_metric"] = torch.zeros(self.num_envs, device=self.device)

        # abuse metrics for storing plot-relevant information for testing
        self.compare = False
        self.data_dict = {}
        self.ext_iter = 0
        if self.compare:
            self.data_dict = {
                "motion_foot1_x": [],
                "motion_foot1_y": [],
                "motion_foot1_z": [],
                "motion_foot2_x": [],
                "motion_foot2_y": [],
                "motion_foot2_z": [],
                "motion_foot3_x": [],
                "motion_foot3_y": [],
                "motion_foot3_z": [],
                "motion_foot4_x": [],
                "motion_foot4_y": [],
                "motion_foot4_z": [],
                "motion_proj_grav_x": [],
                "motion_proj_grav_y": [],
                "motion_proj_grav_z": [],
                "motion_height": [],
                "robot_foot1_x": [],
                "robot_foot1_y": [],
                "robot_foot1_z": [],
                "robot_foot2_x": [],
                "robot_foot2_y": [],
                "robot_foot2_z": [],
                "robot_foot3_x": [],
                "robot_foot3_y": [],
                "robot_foot3_z": [],
                "robot_foot4_x": [],
                "robot_foot4_y": [],
                "robot_foot4_z": [],
                "robot_proj_grav_x": [],
                "robot_proj_grav_y": [],
                "robot_proj_grav_z": [],
                "robot_height": []
            }
            

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
            "base_height_len": 1,               # do not touch this or above
            # "base_vel_next_start": 34,
            # "base_vel_next_len": 3,
            # "base_ang_vel_next_start": 37,
            # "base_ang_vel_next_len": 3,
            # "joint_angles_next_start": 40,
            # "joint_angles_next_len": 12,
            # "joint_angles_nnext_start": 52,
            # "joint_angles_nnext_len": 12,
            "end_points_start": 34,
            "end_points_len": 12,
            "end_points_next_start": 46,
            "end_points_next_len": 12,
            "end_points_nnext_start": 58,
            "end_points_nnext_len": 12,
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
        return_cmd = None
        if not isinstance(self.cfg.terms, list):
            return_cmd = self.imitation_command
        else:
            self.update_custom_imitation_command()
            # return_cmd = self.custom_imitation_command
            imitation_terms = self.custom_imitation_command[:, :self.imitation_command.shape[1]]
            flattened_buffer = self.history_buffer.reshape(self.num_envs, -1)
            return_cmd = torch.cat((imitation_terms, flattened_buffer), dim=1)

        return return_cmd

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # time when executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt

        root_angle = self.robot.data.root_quat_w.unsqueeze(1)  # (4096, 4), w, x, y, z
        root_pos = self.robot.data.root_pos_w.unsqueeze(1)
        foot_pos = self.robot.data.body_state_w[:, 13:, :3] - root_pos

        # Rotate foot_pos by root_angle
        foot_pos = math_utils.quat_rotate_inverse(root_angle, foot_pos)

        # log data
        self.metrics["error_foot_pos"] += (
            torch.norm(self.imitation_command[..., 34:46] - foot_pos.reshape(-1, 12), dim=1) / max_command_step
        )
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
        self.metrics["error_imitation_metric"] = (
            self.metrics["error_foot_pos"] / 7 +
            2 * self.metrics["error_joint_pos"] / 4.5 +
            2 * self.metrics["error_base_vel"] / 3 +
            0.5 * self.metrics["error_base_height"] / 0.03 +
            self.metrics["error_base_proj_grav"] / 0.4 +
            0.5 * self.metrics["error_base_ang_vel"] / 4.5
        )

        # print("error_imitation_metric: ", self.metrics["error_imitation_metric"].mean().item())
        # for metric in self.metrics:
        #     print(f"{metric}: {self.metrics[metric].mean().item()}")
    
    def get_metric(self):
        return self.metrics["error_imitation_metric"].mean().item()

    """
    Updating the command
    """

    def _resample_command(self, env_ids: Sequence[int]):
        """
        Resample the imitation command.

        This function is called when the command needs to be resampled/reset.
        """

        self.motion_number[env_ids] = torch.randint(0, len(self.start_indices) - 1, (len(env_ids),), device=self.device)
        self.motion_index[env_ids] = self.start_indices[self.motion_number[env_ids]].float()
        self.imitation_command[env_ids] = torch.transpose(
            torch.index_select(self.motion, 1, (self.motion_index[env_ids] // 1).type(torch.int32)), 0, 1
        )

        self.motion_index[env_ids] += 1.0

        # reset history buffer
        # self.history_buffer[env_ids] = torch.zeros(self.hist_len, self.num_prop_obs + self.custom_len, device=self.device)
        self.history_buffer[env_ids] = torch.zeros(self.hist_len, self.custom_len, device=self.device)
        # update standing envs
        r = torch.empty(len(env_ids), device=self.device)
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

        if isinstance(self.cfg.terms, list):
            self.update_custom_imitation_command()

    def _update_command(self):
        """
        Post-process the imitation command.
        """
        if self.compare:
            self.set_comp_metrics()
            if self.ext_iter == 999:
                self.plot_metrics()

        self.imitation_command[...] = torch.transpose(
            torch.index_select(self.motion, 1, (self.motion_index // 1).type(torch.int32)), 0, 1
        )
        self.motion_index += 1.0
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

        if isinstance(self.cfg.terms, list):
            self.update_custom_imitation_command()
    
    def update_custom_imitation_command(self):
        """
        For custom imitation command indexing.
        """
        if self.custom_len == 0:
            for term in self.cfg.terms:
                self.custom_len = self.custom_len + self.indexing_dict[term + "_len"]

        if self.custom_imitation_command.shape[1] != self.imitation_command.shape[1] + self.custom_len:
            print("Reshaping custom command to fit inputs...")
            self.custom_imitation_command = torch.zeros(self.num_envs, self.imitation_command.shape[1] + self.custom_len, device=self.device)

        # if self.history_buffer.shape[2] != self.num_prop_obs + self.custom_len:
        #     print("Reshaping history buffer to fit inputs...")
        #     self.history_buffer = torch.zeros(
        #         self.num_envs, self.hist_len, self.num_prop_obs + self.custom_len, device=self.device
        #     )
        if self.history_buffer.shape[2] != self.custom_len:
            print("Reshaping history buffer to fit inputs...")
            self.history_buffer = torch.zeros(
                self.num_envs, self.hist_len, self.custom_len, device=self.device
            )

        current_idx = self.imitation_command.shape[1]
        self.custom_imitation_command[:, :current_idx] = self.imitation_command[...]
        for term in self.cfg.terms:
            term_start = self.indexing_dict[term + "_start"]
            term_len = self.indexing_dict[term + "_len"]
            
            # Directly assign each slice from self.imitation_command into the correct location in self.custom_imitation_command
            self.custom_imitation_command[:, current_idx : current_idx + term_len] = \
                self.imitation_command[:, term_start : term_start + term_len]
    
            current_idx += term_len

        # n_min = -0.1
        # n_max = 0.1
        # noise = torch.rand_like(self.custom_imitation_command[:, self.imitation_command.shape[1]:]) * (n_max - n_min) + n_min
        # self.custom_imitation_command[:, self.imitation_command.shape[1]:] = self.custom_imitation_command[:, self.imitation_command.shape[1]:] + noise

        self.update_history()

    def update_history(self):
        """
        Update the history buffer.

        Only contains observation terms, need to append other obs terms at another phase.
        """
        # cur_prop_cmds = self.get_prop_commands()
        cur_im_cmds = self.custom_imitation_command[:, self.imitation_command.shape[1]:]

        # cur_cmd = torch.cat((cur_prop_cmds, cur_im_cmds), dim=1)
        cur_cmd = cur_im_cmds

        prev_cmds = self.history_buffer[:, 1:, :]
        # print(cur_cmd.shape)
        # print(prev_cmds.shape)

        self.history_buffer = torch.cat((prev_cmds, cur_cmd.unsqueeze(1)), dim=1)

    # def get_prop_commands(self) -> torch.Tensor:
    #     """
    #     Get proprioceptive commands.
    #     """
    #     base_lin_vel = self.robot.data.root_pos_w
    #     base_lin_vel_n = torch.rand_like(base_lin_vel) * (2 * 0.1) - 0.1
    #     base_lin_vel = base_lin_vel + base_lin_vel_n
    #     base_ang_vel = self.robot.data.root_ang_vel_b
    #     base_ang_vel_n = torch.rand_like(base_ang_vel) * (2 * 0.2) - 0.2
    #     base_ang_vel = base_ang_vel + base_ang_vel_n
    #     base_proj_grav = self.robot.data.projected_gravity_b
    #     base_proj_grav_n = torch.rand_like(base_proj_grav) * (2 * 0.05) - 0.05
    #     base_proj_grav = base_proj_grav + base_proj_grav_n
    #     joint_pos = self.robot.data.joint_pos - self.robot.data.default_joint_pos
    #     joint_pos_n = torch.rand_like(joint_pos) * (2 * 0.01) - 0.01
    #     joint_pos = joint_pos + joint_pos_n
    #     joint_vel = self.robot.data.joint_vel - self.robot.data.default_joint_vel
    #     joint_vel_n = torch.rand_like(joint_vel) * (2 * 1.5) - 1.5
    #     joint_vel = joint_vel + joint_vel_n
    #     actions = self.env.action_manager.action

    #     # concat into a single tensor with shape (num_envs, 45)
    #     prop_commands = torch.cat(
    #         (joint_pos, joint_vel, base_lin_vel, base_ang_vel, base_proj_grav, actions), dim=1
    #     ).to(self.device)

    #     return prop_commands

    def set_comp_metrics(self):
        self.ext_iter += 1
        self.data_dict["motion_foot1_x"].append(self.imitation_command[0, 34].item())
        self.data_dict["motion_foot1_y"].append(self.imitation_command[0, 35].item())
        self.data_dict["motion_foot1_z"].append(self.imitation_command[0, 36].item())
        self.data_dict["motion_foot2_x"].append(self.imitation_command[0, 37].item())
        self.data_dict["motion_foot2_y"].append(self.imitation_command[0, 38].item())
        self.data_dict["motion_foot2_z"].append(self.imitation_command[0, 39].item())
        self.data_dict["motion_foot3_x"].append(self.imitation_command[0, 40].item())
        self.data_dict["motion_foot3_y"].append(self.imitation_command[0, 41].item())
        self.data_dict["motion_foot3_z"].append(self.imitation_command[0, 42].item())
        self.data_dict["motion_foot4_x"].append(self.imitation_command[0, 43].item())
        self.data_dict["motion_foot4_y"].append(self.imitation_command[0, 44].item())
        self.data_dict["motion_foot4_z"].append(self.imitation_command[0, 45].item())
        self.data_dict["motion_proj_grav_x"].append(self.imitation_command[0, 30].item())
        self.data_dict["motion_proj_grav_y"].append(self.imitation_command[0, 31].item())
        self.data_dict["motion_proj_grav_z"].append(self.imitation_command[0, 32].item())
        self.data_dict["motion_height"].append(self.imitation_command[0, 33].item())

        root_angle = self.robot.data.root_quat_w.unsqueeze(1)  # (4096, 4), w, x, y, z
        root_pos = self.robot.data.root_pos_w.unsqueeze(1)
        foot_pos = self.robot.data.body_state_w[:, 13:, :3] - root_pos

        # Rotate foot_pos by root_angle
        foot_pos = math_utils.quat_rotate_inverse(root_angle, foot_pos)

        self.data_dict["robot_foot1_x"].append(foot_pos[0, 0, 0].item())
        self.data_dict["robot_foot1_y"].append(foot_pos[0, 0, 1].item())
        self.data_dict["robot_foot1_z"].append(foot_pos[0, 0, 2].item())
        self.data_dict["robot_foot2_x"].append(foot_pos[0, 1, 0].item())
        self.data_dict["robot_foot2_y"].append(foot_pos[0, 1, 1].item())
        self.data_dict["robot_foot2_z"].append(foot_pos[0, 1, 2].item())
        self.data_dict["robot_foot3_x"].append(foot_pos[0, 2, 0].item())
        self.data_dict["robot_foot3_y"].append(foot_pos[0, 2, 1].item())
        self.data_dict["robot_foot3_z"].append(foot_pos[0, 2, 2].item())
        self.data_dict["robot_foot4_x"].append(foot_pos[0, 3, 0].item())
        self.data_dict["robot_foot4_y"].append(foot_pos[0, 3, 1].item())
        self.data_dict["robot_foot4_z"].append(foot_pos[0, 3, 2].item())
        self.data_dict["robot_proj_grav_x"].append(self.robot.data.projected_gravity_b[0, 0].item())
        self.data_dict["robot_proj_grav_y"].append(self.robot.data.projected_gravity_b[0, 1].item())
        self.data_dict["robot_proj_grav_z"].append(self.robot.data.projected_gravity_b[0, 2].item())
        self.data_dict["robot_height"].append(self.robot.data.root_pos_w[0, 2].item())


    def plot_metrics(self):
        import matplotlib.pyplot as plt
        from datetime import datetime

        fig, axs = plt.subplots(3, 2, figsize=(15, 10))
        fig.suptitle('Detailed Metrics Comparison', fontsize=16)

        axs[0, 0].plot(self.data_dict["motion_proj_grav_x"], 'b--', label='motion_x')
        axs[0, 0].plot(self.data_dict["robot_proj_grav_x"], 'b', label='robot_x')
        axs[0, 0].plot(self.data_dict["motion_proj_grav_y"], 'y--', label='motion_y')
        axs[0, 0].plot(self.data_dict["robot_proj_grav_y"], 'y', label='robot_y')
        axs[0, 0].plot(self.data_dict["motion_proj_grav_z"], 'r--', label='motion_z')
        axs[0, 0].plot(self.data_dict["robot_proj_grav_z"], 'r', label='robot_z')
        axs[0, 0].set_title('Projected Gravity')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        axs[0, 1].plot(self.data_dict["motion_height"], 'r--', label='motion')
        axs[0, 1].plot(self.data_dict["robot_height"], 'b', label='robot')
        axs[0, 1].set_title('Base Height')
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        axs[1, 0].plot(self.data_dict["motion_foot1_x"], 'b--', label='motion_x')
        axs[1, 0].plot(self.data_dict["robot_foot1_x"], 'b', label='robot_x')
        axs[1, 0].plot(self.data_dict["motion_foot1_y"], 'y--', label='motion_y')
        axs[1, 0].plot(self.data_dict["robot_foot1_y"], 'y', label='robot_y')
        axs[1, 0].plot(self.data_dict["motion_foot1_z"], 'r--', label='motion_z')
        axs[1, 0].plot(self.data_dict["robot_foot1_z"], 'r', label='robot_z')
        axs[1, 0].set_title('Foot1 Position')
        # axs[1, 0].legend()
        axs[1, 0].grid(True)

        axs[1, 1].plot(self.data_dict["motion_foot2_x"], 'b--', label='motion_x')
        axs[1, 1].plot(self.data_dict["robot_foot2_x"], 'b', label='robot_x')
        axs[1, 1].plot(self.data_dict["motion_foot2_y"], 'y--', label='motion_y')
        axs[1, 1].plot(self.data_dict["robot_foot2_y"], 'y', label='robot_y')
        axs[1, 1].plot(self.data_dict["motion_foot2_z"], 'r--', label='motion_z')
        axs[1, 1].plot(self.data_dict["robot_foot2_z"], 'r', label='robot_z')
        axs[1, 1].set_title('Foot2 Position')
        # axs[1, 1].legend()
        axs[1, 1].grid(True)

        axs[2, 0].plot(self.data_dict["motion_foot3_x"], 'b--', label='motion_x')
        axs[2, 0].plot(self.data_dict["robot_foot3_x"], 'b', label='robot_x')
        axs[2, 0].plot(self.data_dict["motion_foot3_y"], 'y--', label='motion_y')
        axs[2, 0].plot(self.data_dict["robot_foot3_y"], 'y', label='robot_y')
        axs[2, 0].plot(self.data_dict["motion_foot3_z"], 'r--', label='motion_z')
        axs[2, 0].plot(self.data_dict["robot_foot3_z"], 'r', label='robot_z')
        axs[2, 0].set_title('Foot3 Position')
        # axs[2, 0].legend()
        axs[2, 0].grid(True)

        axs[2, 1].plot(self.data_dict["motion_foot4_x"], 'b--', label='motion_x')
        axs[2, 1].plot(self.data_dict["robot_foot4_x"], 'b', label='robot_x')
        axs[2, 1].plot(self.data_dict["motion_foot4_y"], 'y--', label='motion_y')
        axs[2, 1].plot(self.data_dict["robot_foot4_y"], 'y', label='robot_y')
        axs[2, 1].plot(self.data_dict["motion_foot4_z"], 'r--', label='motion_z')
        axs[2, 1].plot(self.data_dict["robot_foot4_z"], 'r', label='robot_z')
        axs[2, 1].set_title('Foot4 Position')
        # axs[2, 1].legend()
        axs[2, 1].grid(True)

        # save plot as stats_{current_time}.png
        plt.savefig(f"stats_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png", dpi=300)