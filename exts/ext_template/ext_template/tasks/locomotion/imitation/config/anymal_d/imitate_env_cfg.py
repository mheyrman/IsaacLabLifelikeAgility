from ext_template.tasks.locomotion.imitation.imitation_env_cfg import LocomotionImitationEnvCfg

from omni.isaac.lab.utils import configclass
from omni.isaac.lab_assets.anymal import ANYMAL_D_CFG


@configclass
class AnymalDImitateEnvCfg(LocomotionImitationEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


@configclass
class AnymalDImitateEnvCfg_PLAY(AnymalDImitateEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None

@configclass
class AnymalDImitateEnvCfg_FINETUNE(AnymalDImitateEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        # reduce the number of environments for finetuning
        self.scene.num_envs = 1024
        self.scene.env_spacing = 2.5

        self.observations.policy.enable_corruption = False