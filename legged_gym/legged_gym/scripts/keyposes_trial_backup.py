#!/usr/bin/env python
from isaacgym import gymapi, gymutil, gymtorch  # Import gymtorch directly
from isaacgym.torch_utils import *

import torch
import time

# Import your environment and configuration classes
from legged_gym.envs.g1.g1_mimic_view_motion import G1MimicViewMotion
from legged_gym.envs.g1.g1_mimic_config_generate import G1MimicCfg
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

def global_to_local(quat, rigid_body_pos, root_pos):
    num_key_bodies = rigid_body_pos.shape[1]
    num_envs = rigid_body_pos.shape[0]
    total_bodies = num_key_bodies * num_envs
    heading_rot_expand = quat.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, num_key_bodies, 1))
    flat_heading_rot = heading_rot_expand.view(total_bodies, heading_rot_expand.shape[-1])

    flat_end_pos = (rigid_body_pos - root_pos[:, None, :3]).view(total_bodies, 3)
    local_end_pos = quat_rotate_inverse(flat_heading_rot, flat_end_pos).view(num_envs, num_key_bodies, 3)
    return local_end_pos

def push_dof_state(env):
    """Push changes in env.dof_pos to the simulation's DOF state tensor."""
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int32)
    env.gym.set_dof_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.dof_state),
        gymtorch.unwrap_tensor(env_ids),
        env.num_envs
    )

def generate_realtime_keyposes(cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless=False):
    # Create the environment using your robot configuration.
    env = G1MimicViewMotion(cfg, sim_params, physics_engine, sim_device, headless)
    
    # Optionally set the simulation timestep (dt is typically set in your config)
    dt = env.dt

    start = time.time()
    # Define an initial desired keypose as a tensor of joint angles.
    desired_joint_angles = torch.tensor([[-1.7518e-01,  4.8021e-02, -4.7398e-04,  2.7878e-02, -2.7684e-01,
                    1.9065e-01,  9.7109e-02, -4.2435e-02,  4.1422e-02,  1.5924e-01,
                    1.1136e-01,  7.7074e-02,  5.4796e-02, -5.4327e-02,  1.0135e-01,
                    -1.7970e-01,  3.0671e-01,  1.2346e-01,  1.1751e+00, -2.0033e-01,
                    -1.7999e-01,  2.1175e-01,  1.4363e+00]], device='cuda:0')

    print("desired_joint_angles ", desired_joint_angles)

    # Set the robot's joint positions (DOF positions) to your keypose.
    env.dof_pos[:] = desired_joint_angles
    push_dof_state(env)

    # Step the simulation once to update the robot state.
    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)

    # Sync internal simulation state if needed.
    env._motion_sync()

    # Extract global keyposes from the simulation.
    keyposes_global = env.rigid_body_states[:, env._key_body_ids_sim, :3]

    # local_keyposes = global_to_local(env.root_states[:, 3:7], keyposes_global, env.root_states[:, :3])
    # print(env.base_quat, keyposes_global, env.root_states[:, :3])
    local_keyposes = global_to_local(env.base_quat, keyposes_global, env.root_states[:, :3])

    print("Local Keyposes:\n", local_keyposes.cpu().numpy())

    # print("Global Keyposes (after first step):\n", keyposes_global.cpu().numpy())
    print("Time taken to generate keyposes: ", time.time() - start)

    return env

if __name__ == "__main__":
    # Load your robot configuration.
    cfg = G1MimicCfg()  # or LeggedRobotCfg() if preferred

    # Set up simulation parameters.
    sim_params = gymapi.SimParams()
    sim_params.dt = 0.002  # Example timestep
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

    # Choose your physics engine and device.
    physics_engine = gymapi.SIM_PHYSX
    sim_device = "cuda:0"  # Use "cpu" if GPU is not available

    # Create the environment and generate keyposes with visualization (headless=False)
    # (Set headless=False to see the viewer; here you set it to headless=True for initial timing, but you can switch.)
    env = generate_realtime_keyposes(cfg, sim_params, physics_engine, sim_device, headless=False)

    # Retrieve the viewer created by the environment.
    viewer = env.viewer

    i = 1
    # if viewer is None:
    #     print("Viewer was not created. Check your configuration.")
    # else:
    #     print("Starting simulation loop. Close the viewer window to exit.")
    while True:
        # i += 1
        # Alternate between two keypose configurations.
        if i % 2:
            desired_joint_angles = torch.tensor([[ 0.0730, -0.0264, -0.0404,  0.1604, -0.5224,  0.1943,
                                                    -0.4813,  0.0660,  0.1682,  0.1074,  0.0598, -0.1400,
                                                        0.0654, -0.0342, -0.0416, -0.2312,  0.3719,  0.0072,
                                                        0.9095, -0.1198, -0.1329,  0.0955,  1.4931]],
                device=sim_device)
        else:
            desired_joint_angles = torch.tensor([[-1.7518e-01,  4.8021e-02, -4.7398e-04,  2.7878e-02, -2.7684e-01,
                        1.9065e-01,  9.7109e-02, -4.2435e-02,  4.1422e-02,  1.5924e-01,
                        1.1136e-01,  7.7074e-02,  5.4796e-02, -5.4327e-02,  1.0135e-01,
                        -1.7970e-01,  3.0671e-01,  1.2346e-01,  1.1751e+00, -2.0033e-01,
                        -1.7999e-01,  2.1175e-01,  1.4363e+00]], device='cuda:0')
        print("desired_joint_angles ", desired_joint_angles)


        start = time.time()
        # Set the new keypose.
        env.dof_pos[:] = desired_joint_angles
        push_dof_state(env)

        # Step the simulation.
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)
        env.gym.step_graphics(env.sim)
        env.gym.draw_viewer(viewer, env.sim, True)

        # Sync internal state if needed.
        env._motion_sync()

        # Extract global keyposes.
        keyposes_global = env.rigid_body_states[:, env._key_body_ids_sim, :3]
        # Now compute local keyposes.
        # We assume the base orientation is stored in env.base_quat.
        # If not, you might use env.root_states[:, 3:7] instead.
        print(env.base_quat, keyposes_global, env.root_states[:, :3])

        local_keyposes = global_to_local(env.base_quat, keyposes_global, env.root_states[:, :3])
        # local_keyposes = global_to_local(env.root_states[:, 3:7], keyposes_global, env.root_states[:, :3])
        print("Local Keyposes:\n", local_keyposes.cpu().numpy())
        # print("Time taken for step: ", time.time() - start)

        time.sleep(sim_params.dt)

    env.gym.destroy_viewer(viewer)
    env.gym.destroy_sim(env.sim)
