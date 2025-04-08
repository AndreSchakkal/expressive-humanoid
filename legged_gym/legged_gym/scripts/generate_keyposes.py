#!/usr/bin/env python
from isaacgym import gymapi, gymutil, gymtorch  # Import gymtorch directly
from isaacgym.torch_utils import *

import torch
import time

# Import your environment and configuration classes
from legged_gym.envs.g1.g1_mimic_view_motion import G1MimicViewMotion
from legged_gym.envs.g1.g1_mimic_config_generate import G1MimicCfg
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg


import threading
import lcm
from legged_gym.lcm_types.joint_angles import joint_angles
from legged_gym.lcm_types.joint_keyposes import joint_keyposes


# lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
lc = lcm.LCM()
# lc = lcm.LCM("tcp://192.168.123.164:7667")

lc_internal = lcm.LCM("udpm://239.255.76.68:7667?ttl=1")


lc_external = lcm.LCM("udpm://239.255.76.67:7667?ttl=255") # for real robot only
# lc_external = lc_internal   # for simulation only


sim_device = "cuda:0"  # Use "cpu" if GPU is not available

# Global variable to store the latest received joint angles.
# latest_joint_angles = torch.tensor([[-1.7518e-01,  4.8021e-02, -4.7398e-04,  2.7878e-02, -2.7684e-01,
#                     1.9065e-01,  9.7109e-02, -4.2435e-02,  4.1422e-02,  1.5924e-01,
#                     1.1136e-01,  7.7074e-02,  5.4796e-02, -5.4327e-02,  1.0135e-01,
#                     -1.7970e-01,  3.0671e-01,  1.2346e-01,  1.1751e+00, -2.0033e-01,
#                     -1.7999e-01,  2.1175e-01,  1.4363e+00]], device='cuda:0')
latest_joint_angles = torch.zeros(1,23, device=sim_device)


# LCM callback: called whenever a "joint_angles" message is received.
def joint_angles_handler(channel, data):
    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    global latest_joint_angles
    # Decode the incoming message using the generated LCM class.
    msg = joint_angles.decode(data)
    # Convert the list from the message to a torch tensor of shape [1, 23].
    latest_joint_angles = torch.tensor([msg.dof_pos], device=sim_device)
    print("Received joint angles:", latest_joint_angles)

subscription = lc_internal.subscribe("joint_angles", joint_angles_handler)


# Run the LCM event loop in a separate thread so it doesn't block the simulation loop.
def lcm_handle_loop():
    while True:
        lc_internal.handle()  # This blocks until a message is received.

lcm_thread = threading.Thread(target=lcm_handle_loop, daemon=True)
lcm_thread.start()


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

def generate_realtime_keyposes(cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless=True):
    # Create the environment using your robot configuration.
    print("sim_device sim_device sim_device", sim_device)
    env = G1MimicViewMotion(cfg, sim_params, physics_engine, sim_device, headless)
    print("env env envenv ", env.device)
    # Optionally set the simulation timestep (dt is typically set in your config)
    dt = env.dt

    start = time.time()
    # # Define an initial desired keypose as a tensor of joint angles.
    # desired_joint_angles = torch.tensor([[-1.7518e-01,  4.8021e-02, -4.7398e-04,  2.7878e-02, -2.7684e-01,
    #                 1.9065e-01,  9.7109e-02, -4.2435e-02,  4.1422e-02,  1.5924e-01,
    #                 1.1136e-01,  7.7074e-02,  5.4796e-02, -5.4327e-02,  1.0135e-01,
    #                 -1.7970e-01,  3.0671e-01,  1.2346e-01,  1.1751e+00, -2.0033e-01,
    #                 -1.7999e-01,  2.1175e-01,  1.4363e+00]], device=env.device)

    desired_joint_angles = latest_joint_angles

    print("desired_joint_angles ", desired_joint_angles)

    # Set the robot's joint positions (DOF positions) to your keypose.
    env.dof_pos[:] = desired_joint_angles







    

    root_pos = torch.zeros((env.num_envs, 3), device=env.device)
    # root_pos[:, 0] = 2.4134
    # root_pos[:, 1] = 0.0211
    # root_pos[:, 2] = 0.7940
    root_rot = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1) 
    dof_pos = desired_joint_angles

    root_vel = torch.zeros((env.num_envs, 3), device=env.device)
    root_ang_vel = torch.zeros((env.num_envs, 3), device=env.device)
    dof_vel = torch.zeros((env.num_envs, 23), device=env.device)

    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)

    # dof_pos, dof_vel = self.reindex_dof_pos_vel(dof_pos, dof_vel)

    # if not self.save:
    #     root_pos[:, 2] = 30

    env._set_env_state(env_ids=env_ids, 
                        root_pos=root_pos, 
                        root_rot=root_rot, 
                        dof_pos=dof_pos, 
                        root_vel=root_vel, 
                        root_ang_vel=root_ang_vel, 
                        dof_vel=dof_vel)

    env_ids_int32 = env_ids.to(dtype=torch.int32)
    # self.root_states[:,2] = self.root_states[:,2] + 0.07
    env.gym.set_actor_root_state_tensor_indexed(env.sim,
                                                gymtorch.unwrap_tensor(env.root_states),
                                                gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    env.gym.set_dof_state_tensor_indexed(env.sim,
                                        gymtorch.unwrap_tensor(env.dof_state),
                                        gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))







    local_end_pos = global_to_local(env.base_quat, env.rigid_body_states[:, env._key_body_ids_sim, :3], env.root_states[:, :3])
































    # push_dof_state(env)

    # # Step the simulation once to update the robot state.
    # env.gym.simulate(env.sim)
    # env.gym.fetch_results(env.sim, True)
    # env.gym.refresh_dof_state_tensor(env.sim)
    # env.gym.refresh_rigid_body_state_tensor(env.sim)

    # # Sync internal simulation state if needed.
    # env._motion_sync()

    # # Extract global keyposes from the simulation.
    # keyposes_global = env.rigid_body_states[:, env._key_body_ids_sim, :3]

    # # local_keyposes = global_to_local(env.root_states[:, 3:7], keyposes_global, env.root_states[:, :3])
    # # print(env.base_quat, keyposes_global, env.root_states[:, :3])
    # local_keyposes = global_to_local(env.base_quat, keyposes_global, env.root_states[:, :3])

    print("Local Keyposes:\n", local_end_pos.cpu().numpy())

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
    sim_params.use_gpu_pipeline = True

    # Choose your physics engine and device.
    physics_engine = gymapi.SIM_PHYSX
    sim_device = "cuda:0"  # Use "cpu" if GPU is not available


    # Create the environment and generate keyposes with visualization (headless=False)
    # (Set headless=False to see the viewer; here you set it to headless=True for initial timing, but you can switch.)
    env = generate_realtime_keyposes(cfg, sim_params, physics_engine, sim_device, headless=True)

    # Retrieve the viewer created by the environment.
    viewer = env.viewer

    i = 1
    # if viewer is None:
    #     print("Viewer was not created. Check your configuration.")
    # else:
    #     print("Starting simulation loop. Close the viewer window to exit.")
    while True:
        i += 1
        # desired_joint_angles = torch.tensor([[-1.7333e-01,  3.9675e-02, -3.6102e-03, -3.1891e-04, -2.6014e-01,
        #             2.0221e-01,  1.0174e-01, -4.8278e-02,  6.3833e-02,  1.6817e-01,
        #             1.2306e-01,  5.4153e-02,  1.6971e-03, -5.8503e-02,  1.0887e-01,
        #             -1.7139e-01,  3.0066e-01,  1.8666e-01,  1.2035e+00, -1.6245e-01,
        #             -1.7519e-01,  2.5339e-01,  1.4735e+00]], device=sim_device)
        desired_joint_angles = latest_joint_angles
        print("desired_joint_angles ", desired_joint_angles)

        
        start = time.time()
        # Set the new keypose.
        env.dof_pos[:] = desired_joint_angles



        root_pos = torch.zeros((env.num_envs, 3), device=sim_device)
        # root_pos[:, 0] = 2.4134
        # root_pos[:, 1] = 0.0211
        root_pos[:, 2] = 0.7940
        root_rot = torch.tensor([0.0, 0.0, 0.0, 1.0], device=sim_device).repeat(env.num_envs, 1) 
        dof_pos = desired_joint_angles

        root_vel = torch.zeros((env.num_envs, 3),device=sim_device)
        root_ang_vel = torch.zeros((env.num_envs, 3), device=sim_device)
        dof_vel = torch.zeros((env.num_envs, 23), device=sim_device)

        env_ids = torch.arange(env.num_envs, dtype=torch.long, device=sim_device)

        # dof_pos, dof_vel = self.reindex_dof_pos_vel(dof_pos, dof_vel)

        # if not self.save:
        #     root_pos[:, 2] = 30

        env._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        # self.root_states[:,2] = self.root_states[:,2] + 0.07
        env.gym.set_actor_root_state_tensor_indexed(env.sim,
                                                    gymtorch.unwrap_tensor(env.root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        env.gym.set_dof_state_tensor_indexed(env.sim,
                                            gymtorch.unwrap_tensor(env.dof_state),
                                            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))





        local_end_pos = global_to_local(env.base_quat, env.rigid_body_states[:, env._key_body_ids_sim, :3], env.root_states[:, :3])






        # push_dof_state(env)

        # Step the simulation.
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)
        env.gym.step_graphics(env.sim)
        env.gym.draw_viewer(viewer, env.sim, True)

        # # Sync internal state if needed.
        # env._motion_sync()



        # # Extract global keyposes.
        # keyposes_global = env.rigid_body_states[:, env._key_body_ids_sim, :3]
        # # Now compute local keyposes.
        # # We assume the base orientation is stored in env.base_quat.
        # # If not, you might use env.root_states[:, 3:7] instead.
        # print(env.base_quat, keyposes_global, env.root_states[:, :3])

        # local_keyposes = global_to_local(env.base_quat, keyposes_global, env.root_states[:, :3])
        # local_keyposes = global_to_local(env.root_states[:, 3:7], keyposes_global, env.root_states[:, :3])
        
        

        joint_keyposes_msg = joint_keyposes()
        joint_keyposes_msg.dof_pos = dof_pos.reshape(-1)
        joint_keyposes_msg.keyposes = local_end_pos.reshape(local_end_pos.shape[0], -1)[0]
        lc_external.publish("joint_keyposes", joint_keyposes_msg.encode())

        print("Local Keyposes:\n", local_end_pos.cpu().numpy())
        # print("Time taken for step: ", time.time() - start)

        time.sleep(sim_params.dt)

    env.gym.destroy_viewer(viewer)
    env.gym.destroy_sim(env.sim)
