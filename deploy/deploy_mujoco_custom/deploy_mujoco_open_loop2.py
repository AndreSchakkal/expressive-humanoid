from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil



import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import yaml
from helpers import  get_args#, export_policy_as_jit, task_registry, Logger


import torch

from motions import Motions
from motions_config import MotionsCfg



def euler_from_quaternion_new(quat_angle):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        w = quat_angle[0]; x = quat_angle[1]; y = quat_angle[2]; z = quat_angle[3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1, 1)
        pitch_y = np.arcsin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)
     
        return torch.tensor(roll_x), torch.tensor(pitch_y), torch.tensor(yaw_z) # in radians
        # return roll_x, pitch_y, yaw_z # in radians


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    print()
    print("---------")
    print("actions_scaled + self.default_dof_pos_all ", target_q )                
    print("self.dof_pos ", q)                
    print("self.dof_vel ", dq)
    torques = motor_strength[0]*(target_q - q) * kp + motor_strength[1]*(target_dq - dq) * kd 
    print("torques ", torques)  
    # return motor_strength[0]*(target_q - q) * kp + motor_strength[1]*(target_dq - dq) * kd
    return (target_q - q) * kp + (target_dq - dq) * kd

def open_loop_control(elapsed_time, default_angles, amplitude=0.1, frequency=1.0):
    """
    Generates open-loop target joint positions based on a sinusoidal signal.
    """
    return default_angles + amplitude * np.sin(2 * np.pi * frequency * elapsed_time) * np.ones_like(default_angles)


LEGGED_GYM_ROOT_DIR = "/home/schakkal/expressive-humanoid"


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    motor_strength = torch.tensor([[[1.0688, 0.8927, 1.0870, 0.9373, 1.1623, 1.0584, 0.8330, 1.0028,
          1.0276, 1.1598, 1.1258, 0.8610, 0.8619, 1.1247, 0.9642, 0.8568,
          1.1187, 0.8822, 0.9607, 1.0180, 0.8635, 1.1777, 1.1946]],

        [[1.1676, 0.9501, 1.0591, 1.0001, 1.0866, 1.0795, 1.1757, 1.1455,
          1.1020, 1.1673, 1.1573, 1.0690, 0.9739, 1.1915, 1.0278, 0.9787,
          0.8291, 0.9493, 0.8249, 0.8127, 0.9246, 0.9928, 1.0654]]])


    init_dof = torch.tensor([ 0.0920, -0.0810, -0.0457, -0.1178, -0.0038,  0.0267,  0.0672,  0.0325,
         -0.0416, -0.0553,  0.1020, -0.0047,  0.0476,  0.0053,  0.0438, -0.0663,
          0.1388, -0.3541,  1.2524, -0.0783, -0.1184,  0.2495,  1.1654])



    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco_custom/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    prop_hist_len = 4
    history_len = 10
    n_proprio = 3 + 2 + 2 + 23*3 + 2 # one hot
    n_demo = 9 + 3 + 3 + 3 +6*3 + 9 + 23 - 4 #without ankle



    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    obs_buf = np.zeros(n_proprio, dtype=np.float32)
    obs_history_buf = np.zeros_like(np.stack([obs_buf] * history_len, axis=0))


    raw_action = np.zeros_like(action)
    smoothed_action = np.zeros_like(action)

    # init_motions(cfg)

    # init_motion_buffers(cfg)


    # args = get_args()
    # args.task="g1_mimic"
    # # prepare environment
    # env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # # obs = env.get_observations()
    motions_cfg = MotionsCfg()
    motions_env = Motions(motions_cfg)



    print("obs_buf", obs_buf.shape)
    print("obs_history_buf: ", obs_history_buf.shape)

    counter = 0
    j = 0
    tor = 250


    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    initial_base_pose = d.qpos[:7].copy()
    # load policy
    policy = torch.jit.load(policy_path)

    # path = "/home/schakkal/expressive-humanoid/deploy/pre_train/g1/traced/improved_config-33000-base_jit.pt"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # policy_jit = torch.jit.load(path, map_location=device)



    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()

        # dof_pos_motion = torch.tensor([[ 0.0920, -0.0810, -0.0457, -0.1178, -0.0038,  0.0267,  0.0672,  0.0325,
        #         -0.0416, -0.0553,  0.1020, -0.0047,  0.0476,  0.0053,  0.0438, -0.0663,
        #         0.1388, -0.3541,  1.2524, -0.0783, -0.1184,  0.2495,  1.1654]])
        # dof_vel = torch.tensor([[ 0.0226, -0.0379, -0.0191, -0.0661, -0.0033,  0.0533,  0.0492,  0.0061,
        #         0.0170, -0.1241,  0.0085, -0.0182, -0.1043, -0.1010,  0.0189,  0.0849,
        #         0.0715,  0.0156,  0.0667,  0.0193,  0.0602, -0.0820,  0.1017]])
        # root_vel = torch.tensor([[-0.0001,  0.0102,  0.0015]])
        # # root_rot = torch.tensor([[ 0.0116, -0.0081,  0.6860,  0.7275]])
        # root_rot = torch.tensor([[0.7275, 0.0116, -0.0081,  0.6860]])
        # root_pos_2 = torch.tensor([0.7271])

        # # Convert your torch tensors to NumPy arrays (flattening to 1D)
        # dof_pos_np = dof_pos_motion.cpu().numpy().flatten()    # shape (23,)
        # dof_vel_np = dof_vel.cpu().numpy().flatten()            # shape (23,)
        # root_vel_np = root_vel.cpu().numpy().flatten()          # shape (3,)
        # root_rot_np = root_rot.cpu().numpy().flatten()          # shape (4,)

        # # For the base position, we only have the z-coordinate from root_pos.
        # # Set x and y to zero (or any desired values):
        # root_z = root_pos_2.cpu().numpy().item()  # extract scalar
        # root_pos_np = np.array([root_z+0.01])      # shape (3,)

        # # Build the full state:
        # # qpos: [root_pos (3), root_rot (4), joint positions (23)]
        # qpos_new = np.concatenate([root_pos_np, root_rot_np, dof_pos_np]) #[7:]
        # # qpos_new = dof_pos_np
        # # qvel: [root linear velocity (3), root angular velocity (set to zeros, 3), joint velocities (23)]
        # qvel_new = np.concatenate([root_vel_np, np.zeros(3), dof_vel_np])  #[7:]
        # # qvel_new = dof_vel_np

        # # Set these in the MuJoCo state:        
        # d.qpos[2:] = qpos_new  #[7:]
        # # d.qpos[:] = qpos_new
        # d.qvel[:] = qvel_new

        # # After modifying state, call forward to update the simulation:
        # mujoco.mj_forward(m, d)
        # mujoco.mj_step(m, d)

        # initial_base_pose = np.concatenate([d.qpos[:3],root_rot[0]])


        while viewer.is_running() and time.time() - start < simulation_duration:


            counter += 1
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            # tau = pd_control(target_dof_pos, qj, kps, np.zeros_like(kds), dqj, kds)
            # tau = pd_control(target_dof_pos, d.qpos, kps, np.zeros_like(kds), d.qvel, kds)
            # tau = np.zeros_like(tau)
            # tor = -tor
            # tau[0,0:12] = 0.
            # tau[0,0] = 0
            # tau[0,1] = 0
            # tau[0,2] = 0
            # tau[0,3] = 0
            # tau[0,4] = 0
            # tau[0,5] = 0
            print("tau ", tau)
            # print("d.ctrl[:] ", d.ctrl[:].shape)
            # print("d.ctrl[:] ", d.ctrl[:])
            # print("d.ctrl ", d.ctrl.shape)
            # print("d.ctrl ", d.ctrl)
            d.ctrl[:] = tau


            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            # time.sleep(1)
            step_start = time.time()
            # print("target_dof_pos ", target_dof_pos)
            elapsed_time = time.time() - start

            motions_env._motion_times += motions_env._motion_dt
            motions_env._motion_times[motions_env._motion_times >= motions_env._motion_lengths] = 0.
            motions_env.update_demo_obs()
            # if counter == 10:
            obs_demo1 = motions_env.compute_obs_demo()            
            # counter += 1
            if counter % control_decimation == 0:
                print()
                print()
                print("!!!!!!!!!")
                print("!!!!!!!!!")
                j+=1
                episode_length_buf = j
                # Apply control signal here.

                # print("obs_demo1", obs_demo1)
                # print("motions_env._motion_times ", motions_env._motion_times)


                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                # qj = d.qpos
                # dqj = d.qvel
                quat = d.qpos[3:7]
                # quat = [0.7164, -0.0439, -0.0218,  0.6960]
                print("quat ",quat)

                omega = d.qvel[3:6]
                
                # print(quat)
                # print(d.qvel[3:6])

                # #maybe also switch quat
                # omega = quat_rotate_inverse(torch.from_numpy(quat).float() , torch.from_numpy(d.qvel[3:6]).float().unsqueeze(0) )
                # # omega = quat_rotate_inverse(torch.from_numpy(quat).float() , torch.from_numpy(d.qvel[3:6]).float().unsqueeze(0) )

                q_tensor = torch.from_numpy(np.array([quat[1],quat[2],quat[3],quat[0]])).float().unsqueeze(0)    # shape becomes (1, 4)
                v_tensor = torch.from_numpy(d.qvel[3:6]).float().unsqueeze(0)  # shape becomes (1, 3)

                omega = quat_rotate_inverse(q_tensor, v_tensor)
                omega = omega.squeeze(0)  # if you want a 1D result


                
                # print("o omega o o omega ", omega)

                # wrapped_omega = np.sign(omega) * (np.abs(omega) % (2 * np.pi))
                # wrapped_omega = (omega + np.pi) % (2 * np.pi) - np.pi
                # omega = wrapped_omega
                # print("o wrapped_omega o o wrapped_omega ", wrapped_omega)

                # print("d.qvel.shape ",d.qvel.shape)

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                # period = 0.8
                # count = counter * simulation_dt
                # phase = count % period / period
                # sin_phase = np.sin(2 * np.pi * phase)
                # cos_phase = np.cos(2 * np.pi * phase)





                roll_x, pitch_y, yaw_z = euler_from_quaternion_new(quat)
                # print("roll_x, pitch_y, yaw_z ", roll_x, pitch_y, yaw_z)

                target_yaw =0
                # target_yaw = motions_env.target_yaw
                obs_demo = obs_demo1.squeeze(0).numpy()  # Convert to NumPy shape [64]
                # obs_demo = torch.tensor([-2.4745e-02,  1.2176e-01,  8.1200e-02, -6.5168e-02, -6.0663e-02,
                # -1.1331e-01, -9.6272e-02,  1.2682e-02, -3.0531e-02, -6.9623e-02,
                # 9.4022e-02, -1.0719e-02,  2.0356e-01, -2.7878e-01,  1.0762e+00,
                # -9.1577e-01, -1.3040e+00,  2.5616e-01,  2.0081e-01,  3.9499e-03,
                # -4.9789e-03,  1.5506e-03,  2.9615e-02,  2.1360e-02, -1.5168e-01,
                # -9.8864e-03,  7.7811e-02,  7.3771e-01,  8.3883e-04,  6.2207e-02,
                # -8.2667e-02,  2.9788e-03,  1.4755e-01, -4.1746e-01,  3.1189e-02,
                # 1.8327e-01, -7.1399e-01,  1.5378e-03, -6.6695e-02, -8.2658e-02,
                # 1.1219e-02, -1.4511e-01, -4.1800e-01,  3.4049e-02, -1.7265e-01,
                # -7.1587e-01,  2.8161e-02,  1.2105e-01,  2.9693e-01,  3.1996e-02,
                # 1.8049e-01,  1.0672e-01,  9.9245e-02,  1.8431e-01, -1.0877e-01,
                # 2.4633e-02, -7.8363e-02,  3.1682e-01,  1.0982e-01, -2.8419e-01,
                # 2.9852e-01,  3.0924e-01, -2.7392e-01,  4.0386e-01])
                # obs_demo = np.zeros(64)  # Target motion from pre-recorded motion or from teleoperation  # NEEDS TO BE ADDED


                # Compute NumPy equivalents of torch operations
                yaw_diff = yaw_z - target_yaw
                sin_yaw = np.sin(yaw_diff)
                cos_yaw = np.cos(yaw_diff)

                # Concatenate using NumPy
                obs_buf = np.concatenate((
                    omega,
                    [roll_x, pitch_y],  # imu_obs
                    [sin_yaw, cos_yaw],  # NEEDS TO BE ADDED
                    qj,
                    dqj,
                    # action,  # Last action
                    raw_action,  # Last action
                    [-0.5, -0.5],  # self.contact_filt.float()*0-0.5
                ))

                # print("omega", omega)
                # print("[roll_x, pitch_y]", [roll_x, pitch_y])  # imu_obs
                # print("[sin_yaw, cos_yaw]", [sin_yaw, cos_yaw])  # NEEDS TO BE ADDED
                # print("qj", qj)
                # print("dqj",dqj)
                # print("action",action)  # Last action

                priv_explicit = np.zeros(3)

                # priv_latent handling
                num_actions = len(action)  # Ensure num_actions is correctly defined
                
                
                priv_latent = np.zeros(4 + 1 + num_actions + num_actions)  # for now  # NEEDS TO BE CHANGED OR UNDERSTOOD
                # priv_latent = np.array([ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  9.7470e-01,                                                                                
                #                         6.8799e-02, -1.0730e-01,  8.7041e-02, -6.2713e-02,  1.6226e-01,          
                #                         5.8429e-02, -1.6696e-01,  2.8117e-03,  2.7620e-02,  1.5985e-01,          
                #                         1.2579e-01, -1.3900e-01, -1.3811e-01,  1.2472e-01, -3.5774e-02,          
                #                         -1.4318e-01,  1.1866e-01, -1.1778e-01, -3.9310e-02,  1.7975e-02,          
                #                         -1.3652e-01,  1.7772e-01,  1.9462e-01,  4.4546e-02, -6.2652e-03,          
                #                         -7.5512e-02, -1.8965e-01, -1.4859e-01,  1.6633e-01,  7.5209e-03,          
                #                         4.0745e-02, -3.0808e-02,  6.3970e-02, -1.4891e-01, -1.0964e-01,          
                #                         -1.6342e-01, -2.3097e-02,  4.1836e-02,  3.4393e-02, -1.0548e-01,
                #                         -2.3853e-02,  8.8947e-02,  8.6490e-02, -3.9195e-02,  8.9053e-02,
                #                         -1.5109e-01])




                # Handle history buffer
                if episode_length_buf <= 1:
                    # For a new episode, initialize the history by repeating the current observation.
                    obs_history_buf = np.stack([obs_buf] * history_len, axis=0)
                else:
                    # For an ongoing episode, shift the history by dropping the oldest observation and appending the new one.
                    obs_history_buf = np.concatenate([
                        obs_history_buf[1:],  # Drop the first (oldest) observation.
                        np.expand_dims(obs_buf, axis=0)  # Add the new observation at the end.
                    ], axis=0)

                # print("obs_history_buf ", obs_history_buf.shape, obs_history_buf)

                motion_features = obs_history_buf[-prop_hist_len:].flatten()



                # Final concatenation (everything in NumPy)
                obs_buf1 = np.concatenate([motion_features, obs_buf, obs_demo, priv_explicit, priv_latent, obs_history_buf.reshape(1, -1).flatten()])

                # obs_tensor = obs_buf  # Now a NumPy array
                obs_tensor = torch.from_numpy(obs_buf1).unsqueeze(0)
                obs_tensor = torch.tensor(obs_tensor, dtype=torch.float32)




                obs_tensor = obs_tensor[:, motions_cfg.env.n_feature:]


                # print("obs.shape",obs_tensor.shape)
                # print(obs_tensor.dtype)






                # roll_x, pitch_y, yaw_z =  euler_from_quaternion(quat)


                # target_yaw = 0
                # obs_demo = torch.zeros(64) # Target motion from pre-recorded motion or from teleoperation  # NEEDS TO BE ADDED


                # obs_buf = torch.cat((omega,
                #                     roll_x, pitch_y, #imu_obs
                #                     torch.sin(yaw_z - target_yaw),  # NEEDS TO BE ADDED
                #                     torch.cos(yaw_z - target_yaw),  # NEEDS TO BE ADDED
                #                     qj,
                #                     dqj,
                #                     action,   # Last action
                #                     -0.5, -0.5, # self.contact_filt.float()*0-0.5         # should be a scalar if I'm not mistaken 
                #                     ),dim=-1)


                # # obs_demo = # Target motion from pre-recorded motion or from teleoperation  # NEEDS TO BE ADDED

                # priv_explicit = torch.zeros(3)


                # # motor_strength_range = [0.8, 1.2]
                # # str_rng = motor_strength_range
                # # self.motor_strength = (str_rng[1] - str_rng[0]) * torch.rand(2, self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) + str_rng[0]

                # priv_latent = torch.zeros(4+1+num_actions+num_actions)  # for now   # NEEDS TO BE CHANGED OR UNDERSTOOD


                # if episode_length_buf <= 1:
                #     # For a new episode, initialize the history by repeating the urrent observation.
                #     history_len = 10
                #     obs_history_buf = torch.stack([obs_buf] * history_len, dim=0)
                # else:
                #     # For an ongoing episode, shift the history by dropping the oldest observation and appending the new one.
                #     obs_history_buf = torch.cat([
                #         obs_history_buf[1:],      # Drop the first (oldest) observation.
                #         obs_buf.unsqueeze(0)           # Add the new observation at the end.
                #     ], dim=0)


                # prop_hist_len = 4
                # motion_features = obs_history_buf[-prop_hist_len:].flatten()



                # obs_buf = torch.cat([motion_features, obs_buf, obs_demo, priv_explicit, priv_latent, obs_history_buf.view(1, -1)], dim=-1)
                # obs_tensor =  obs_buf #.unsqueeze(0)?
                # obs_tensor = torch.from_numpy(obs).unsqueeze(0)







                # obs[:3] = omega
                # obs[3:6] = gravity_orientation
                # obs[6:9] = cmd * cmd_scale
                # obs[9 : 9 + num_actions] = qj
                # obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                # obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                # obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
                # obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                
                
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()

                # raw_action = action.copy()
                # smoothing_factor = 0.3  # adjust between 0 (more smoothing) and 1 (no smoothing)
                # smoothed_action[12:] = (
                #     smoothing_factor * action[12:] + (1 - smoothing_factor) * smoothed_action[12:]
                # )
                # action[12:] = smoothed_action[12:].copy()




                # action = policy_jit(obs_tensor.detach()).detach().numpy() 



                # print("Raw actions 1:", action)
                # action = action * np.pi / 180  # Convert degrees to radians

                clip_actions = motions_cfg.normalization.clip_actions / motions_cfg.control.action_scale
                action = np.clip(action, -clip_actions, clip_actions)
                print("actions:", action)



                # transform action to target_dof_pos
                # print("default_angles ", default_angles)
                target_dof_pos = action * action_scale + default_angles

            
            # target_dof_pos = open_loop_control(elapsed_time, default_angles, amplitude=0.1, frequency=1)
            # target_dof_pos = default_angles.copy()



            # --- Fix the floating base: ---
            # Override the base pose and velocity to prevent the robot from falling.
            # Floating base is typically stored in d.qpos[0:7] (3 for position and 4 for quaternion)
            # and its velocity in d.qvel[0:6] (linear and angular velocities).

            # d.qpos[:7] = initial_base_pose  #[7:]
            # d.qvel[:6] = 0.0                #[7:]

            # alpha = 0.9  # Blending factor (0: no reset, 1: full reset)
            # d.qpos[:7] = alpha * initial_base_pose + (1 - alpha) * d.qpos[:7]
            # d.qvel[:6] = alpha * 0.0 + (1 - alpha) * d.qvel[:6]

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            print("m.opt.timestep ", m.opt.timestep)
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            print("time_until_next_step ",time_until_next_step)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
