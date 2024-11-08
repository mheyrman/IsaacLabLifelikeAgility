from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def generated_imitation_commands(env: ManagerBasedRLEnv, command_name: str, num_ref_motion: int = 40, custom_motion: bool = False) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    commands = env.command_manager.get_command(command_name)
    if not custom_motion:
        return commands[..., :num_ref_motion]   # return full retargeted reference motions
    
    # otherwise return the custom motion
    return commands[..., num_ref_motion:]
