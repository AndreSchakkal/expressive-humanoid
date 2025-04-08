#!/usr/bin/env python
from isaacgym import gymapi, gymutil
import torch
import time

# Import your environment and configuration classes
from legged_gym.envs.g1.g1_mimic import G1Mimic
from legged_gym.envs.g1.g1_mimic_generate import G1MimicViewMotion
from legged_gym.envs.g1.g1_mimic_config import G1MimicCfg
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

def generate_realtime_keyposes(cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless=False):
    # Create the environment using your robot configuration.
    env = G1MimicViewMotion(cfg, sim_params, physics_engine, sim_device, headless)
    
    # Optionally set the simulation timestep (dt is typically set in your config)
    dt = env.dt

    # Define a desired keypose as a tensor of joint angles.
    # Make sure the number of values matches the DOFs expected by your robot.
    desired_joint_angles = torch.tensor([[-0.0720, -0.0435, -0.0501, -0.0984, -0.3172,  0.2371,
                                           -0.3695,  0.0472,  0.0960,  0.8209,  0.0199, -0.0711,
                                            0.1211, -0.0822,  0.0014, -0.1886,  0.2162,  0.0666,
                                            0.9309, -0.0909, -0.2643,  0.1720,  1.2910]],
       device=sim_device)


    # Set the robot's joint positions (DOF positions) to your keypose.
    env.dof_pos[:] = desired_joint_angles

    # Step the simulation once to update the robot state.
    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)

    # (Optional) If your environment uses internal syncing, call _motion_sync()
    env._motion_sync()

    # Extract keyposes: positions of key bodies given by the indices in env._key_body_ids_sim.
    keyposes = env.rigid_body_states[:, env._key_body_ids_sim, :3]
    print("Realtime Keyposes (after first step):\n", keyposes.cpu().numpy())

    return env

if __name__ == "__main__":
    # Load your robot configuration.
    # Here we use G1MimicCfg; adjust if you use a different config.
    cfg = G1MimicCfg()  # or LeggedRobotCfg() if preferred

    # Set up simulation parameters.
    sim_params = gymapi.SimParams()
    sim_params.dt = 0.002 #1 / 60.0  # Example timestep
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

    # Choose your physics engine and device.
    physics_engine = gymapi.SIM_PHYSX
    sim_device = "cuda:0"  # Use "cpu" if GPU is not available

    # Create the environment and generate keyposes with visualization (headless=False)
    env = generate_realtime_keyposes(cfg, sim_params, physics_engine, sim_device, headless=False)

    # Retrieve the viewer created by the environment.
    viewer = env.viewer

    i = 0
    if viewer is None:
        print("Viewer was not created. Check your configuration.")
    else:
        print("Starting simulation loop. Close the viewer window to exit.")
        while not env.gym.query_viewer_has_closed(viewer):
            i += 1
            if i%2 :
                desired_joint_angles = torch.tensor([[ 0.0730, -0.0264, -0.0404,  0.1604, -0.5224,  0.1943, -0.4813,  0.0660,
                    0.1682,  0.1074,  0.0598, -0.1400,  0.0654, -0.0342, -0.0416, -0.2312,
                    0.3719,  0.0072,  0.9095, -0.1198, -0.1329,  0.0955,  1.4931]],
                    device='cuda:0')
            else:
                desired_joint_angles = torch.tensor([[-0.0720, -0.0435, -0.0501, -0.0984, -0.3172,  0.2371,
                                                    -0.3695,  0.0472,  0.0960,  0.8209,  0.0199, -0.0711,
                                                        0.1211, -0.0822,  0.0014, -0.1886,  0.2162,  0.0666,
                                                        0.9309, -0.0909, -0.2643,  0.1720,  1.2910]],
                device=sim_device)                

            # Set the robot's joint positions (DOF positions) to your keypose.
            env.dof_pos[:] = desired_joint_angles



            # Step the simulation
            env.gym.simulate(env.sim)
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.draw_viewer(viewer, env.sim, True)
            

            
            # Optionally, sync internal state
            env._motion_sync()

            # Extract and print keyposes periodically (for example, every frame)
            keyposes = env.rigid_body_states[:, env._key_body_ids_sim, :3]
            print("Realtime Keyposes:\n", keyposes.cpu().numpy())

            # time.sleep(sim_params.dt)
            time.sleep(sim_params.dt)

        env.gym.destroy_viewer(viewer)
        env.gym.destroy_sim(env.sim)
