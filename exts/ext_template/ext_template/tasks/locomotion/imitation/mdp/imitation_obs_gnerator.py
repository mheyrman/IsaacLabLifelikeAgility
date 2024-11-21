from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def generated_imitation_commands(
        env: ManagerBasedRLEnv,
        command_name: str,
        num_ref_motion: int = 46,
        custom_motion: bool = False,
        noise: bool = False) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    # num_ref_motion is the total number of reference motions
    commands = env.command_manager.get_command(command_name)
    if noise:
        n_min = -0.2
        n_max = 0.2
        noise = torch.rand_like(commands[:, num_ref_motion:]) * (n_max - n_min) + n_min
        commands[:, num_ref_motion:] = commands[:, num_ref_motion:] + noise

    if not custom_motion:
        return commands[..., :num_ref_motion]   # return full retargeted reference motions
    
    # otherwise return the custom motion
    return commands[..., num_ref_motion:]
