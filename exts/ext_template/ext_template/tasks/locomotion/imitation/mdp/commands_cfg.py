# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from omni.isaac.lab.managers import CommandTermCfg
from omni.isaac.lab.utils import configclass

from .imitation_command import ImitationCommand


@configclass
class ImitationCommandCfg(CommandTermCfg):
    """Configuration for the imitation command generator."""

    class_type: type = ImitationCommand

    asset_name: str = MISSING

    rel_standing_envs: float = MISSING

    terms: list = MISSING
