"""
This script borrows from quadrupeds.py to load and visualize motions

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p motion_data/motion_loader.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script loads motions.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import os
import torch

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation

##
# Pre-defined configs
##
from omni.isaac.lab_assets.anymal import ANYMAL_D_CFG  # isort:skip


def load_motion_data(data_path: str) -> torch.Tensor:
    """Loads the motion data from the specified relative path."""
    # get the absolute path
    data_path = os.path.abspath(data_path)
    # load the motion data
    motion_data = torch.load(data_path)
    # return the motion data
    return motion_data


def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    """Defines the origins of the the scene."""
    # create tensor based on number of environments
    env_origins = torch.zeros(num_origins, 3)
    # create a grid of origins
    num_cols = np.floor(np.sqrt(num_origins))
    num_rows = np.ceil(num_origins / num_cols)
    xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="xy")
    env_origins[:, 0] = spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
    env_origins[:, 1] = spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
    env_origins[:, 2] = 1.0
    # return the origins
    return env_origins.tolist()


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2", "Origin3"
    # Each group will have a mount and a robot on top of it
    origins = define_origins(num_origins=1, spacing=1.25)

    # Origin 1 with Anymal D
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    # -- Robot
    anymal_d = Articulation(ANYMAL_D_CFG.replace(prim_path="/World/Origin1/Robot"))

    # return the scene information
    scene_entities = {
        "anymal_d": anymal_d,
    }
    return scene_entities, origins


def visualize_motion(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Visualizes the motion data."""
    motion_dict = {}
    motion_keys = []

    # for every file ended in .pt in the motion_data folder
    for file in os.listdir("motion_data"):
        if file.endswith(".pt"):
            # load the motion data
            loaded_data = torch.load(os.path.join("motion_data", file))
            # append to the dict and record key
            motion_dict[file] = loaded_data
            motion_keys.append(file)

    data_num = 0
    while True:
        # get the motion data
        motion_data = motion_dict[motion_keys[data_num]]
        print("motion_key: " + str(motion_keys[data_num]))
        # split data into components
        # base_pos = motion_data[:, :, 0:3]  # base position in global frame
        # base_quat = motion_data[:, :, 3:7]  # base orientation quaternion in global frame
        base_v = motion_data[:, :, 7:10]  # base velocity in local frame
        print(base_v)
        # base_w = motion_data[:, :, 10:13]  # base angular velocity in local frame
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

        # joint_vels = torch.cat((motion_data[:, :, 28],
        #                         motion_data[:, :, 31],
        #                         motion_data[:, :, 34],
        #                         motion_data[:, :, 37],
        #                         motion_data[:, :, 29],
        #                         motion_data[:, :, 32],
        #                         motion_data[:, :, 35],
        #                         motion_data[:, :, 38],
        #                         motion_data[:, :, 30],
        #                         motion_data[:, :, 33],
        #                         motion_data[:, :, 36],
        #                         motion_data[:, :, 39])).unsqueeze(0)  # joint velocities

        joint_vels = torch.zeros_like(joint_angles)  # joint positions don't seem to matter..?

        """Runs the simulation loop."""
        # Define simulation stepping
        sim_dt = sim.get_physics_dt()  # 0.01s
        sim_time = 0.0
        count = 0
        # Simulate physics
        while simulation_app.is_running() and count < 400:
            # reset
            robot = entities["anymal_d"]

            # reset counters
            sim_time = 0.0

            # root state
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] = origins[0]
            root_state[:, 4:7] = 0.0
            root_state[:, 3] = 1.0
            root_state[:, 7:] = 0.0
            robot.write_root_state_to_sim(root_state)

            # joint state
            joint_pos = joint_angles[:, :, count // 2]
            joint_pos = joint_pos.unsqueeze(0)
            joint_vel = joint_vels[:, :, count // 2]
            joint_vel = joint_vel.unsqueeze(0)

            # write joint angle and velocity information to simulation
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            # reset the internal state
            robot.reset()

            # perform step
            sim.step()
            # update sim-time
            sim_time += sim_dt
            count += 1
            # update buffers
            robot.update(sim_dt)

            # Uncomment to replay same imitation data
            # if count == 400:
            #     count = 0
            #     print("RESETTING IMITATION")

        data_num = (data_num + 1) if data_num < 2 else 0


def main():
    """Main function."""

    # Initialize the simulation context
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, gravity=(0.0, 0.0, 0.0)))
    # Set main camera
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])
    # design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    visualize_motion(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
