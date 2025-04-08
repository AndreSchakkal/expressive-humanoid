from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil



import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import yaml
from helpers import  get_args#, export_policy_as_jit, task_registry, Logger
from isaacgym.torch_utils import *
from legged_gym.envs.base.legged_robot import LeggedRobot, euler_from_quaternion

import math
import torch

from motions import Motions
from motions_config import MotionsCfg


import lcm
from legged_gym.lcm_types.joint_keyposes import joint_keyposes
import threading

def build_demo_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, local_key_body_pos, dof_offsets):
    local_root_ang_vel = quat_rotate_inverse(root_rot, root_ang_vel)
    local_root_vel = quat_rotate_inverse(root_rot, root_vel)
    # print(local_root_vel[0])

    # heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    # local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)
    # local_root_vel = quat_rotate(heading_rot, root_vel)
    # print(local_root_vel[0], "\n")

    # root_pos_expand = root_pos.unsqueeze(-2)  # [num_envs, 1, 3]
    # local_key_body_pos = key_body_pos - root_pos_expand
    
    # heading_rot_expand = heading_rot.unsqueeze(-2)
    # heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    # flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    # flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], heading_rot_expand.shape[2])
    # local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    # flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])
    roll, pitch, yaw = euler_from_quaternion(root_rot)
    print()
    # print("dof_pos in build demo: ", dof_pos.shape, dof_pos)
    print()
    print("dof_pos",dof_pos.shape) 
    print("local_root_vel",local_root_vel.shape) 
    print("local_root_ang_vel",local_root_ang_vel.shape) 
    print("roll",roll[:, None].shape) 
    print("pitch",pitch[:, None].shape) 
    print("root_pos[:, 2:3]", root_pos[:, 2:3].shape) 
    print("local_key_body_pos",local_key_body_pos.view(local_key_body_pos.shape[0], -1).shape)

    # return torch.cat((dof_pos, local_root_vel, local_root_ang_vel, roll[:, None], pitch[:, None], root_pos[:, 2:3], torch.zeros_like(local_key_body_pos.view(local_key_body_pos.shape[0], -1))), dim=-1)
    return torch.cat((dof_pos, local_root_vel, local_root_ang_vel, roll[:, None], pitch[:, None], root_pos[:, 2:3], local_key_body_pos.view(local_key_body_pos.shape[0], -1)), dim=-1)


class LCMAgent:
    def __init__(self, url: str = "udpm://239.255.76.68:7667?ttl=1"):
        """
        Initializes the LCMAgent.
        
        Args:
            url (str): The LCM URL to use. Default is "udpm://239.255.76.68:7667?ttl=1".
        """
        self.lc = lcm.LCM(url)
        self.demo_dof_pos = None
        self.local_key_body_pos = None

        # Subscribe to the "joint_keyposes" channel
        self.lc.subscribe("joint_keyposes", self.joint_keyposes_handler)

        # Event to control the LCM thread loop
        self._stop_event = threading.Event()

        # Start the LCM handling loop in a daemon thread
        self.thread = threading.Thread(target=self.lcm_loop, daemon=True)
        self.thread.start()

    def joint_keyposes_handler(self, channel, data):
        """
        Callback function for handling joint_keyposes messages.
        Decodes the message and updates demo_dof_pos and local_key_body_pos.
        """
        msg = joint_keyposes.decode(data)
        # Convert received keyposes (list of doubles) into a NumPy array.
        self.demo_dof_pos = np.array(msg.dof_pos).reshape(1, -1)
        self.local_key_body_pos = np.array(msg.keyposes).reshape(1, -1)
        # print("LCMAgent: Received joint_keyposes:", self.demo_dof_pos)

    def lcm_loop(self):
        """
        The loop that handles incoming LCM messages.
        Runs until the stop event is set.
        """
        while not self._stop_event.is_set():
            self.lc.handle()

    def stop(self):
        """
        Stops the LCM loop and waits for the thread to finish.
        """
        self._stop_event.set()
        self.thread.join()

lcm_agent = LCMAgent()

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
    return (target_q - q) * kp + (target_dq - dq) * kd

LEGGED_GYM_ROOT_DIR = "/home/schakkal/expressive-humanoid"


if __name__ == "__main__":
    # get config file name from command line
    import argparse

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

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt



    # load policy
    policy = torch.jit.load(policy_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            # print("target_dof_pos ", target_dof_pos)
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            # print("tau ", tau)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # print()
                # print()
                # print("!!!!!!!!!")
                # print("!!!!!!!!!")
                j+=1
                episode_length_buf = j
                # Apply control signal here.
                motions_env._motion_times += motions_env._motion_dt
                motions_env._motion_times[motions_env._motion_times >= motions_env._motion_lengths] = 0.
                motions_env.update_demo_obs()
                # if counter == 10:
                obs_demo1 = motions_env.compute_obs_demo()
                # print("obs_demo1", obs_demo1)
                # print("motions_env._motion_times ", motions_env._motion_times)

                # 7. Prepare a demonstration observation (from teleop or prerecorded)
                root_pos = torch.zeros(1,3)
                root_pos[:, 2] = 0.7940
                # root_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
                root_rot = torch.tensor([[0.0, 0., 0.7071, 0.7071]])
                # 0.7071 0 0 0.7071
                root_vel = torch.zeros(1,3)
                root_ang_vel = torch.zeros(1,3)

                if lcm_agent.demo_dof_pos is None:
                    demo_dof_pos = torch.zeros(1,23)
                else:
                    demo_dof_pos = lcm_agent.demo_dof_pos
                
                dof_vel = None
                key_body_pos = None

                if lcm_agent.local_key_body_pos is None:
                    local_key_body_pos = torch.zeros(1,12*3)
                else:
                    local_key_body_pos = lcm_agent.local_key_body_pos
    
    
                demo_dof_pos = torch.as_tensor(demo_dof_pos)
                local_key_body_pos = torch.as_tensor(local_key_body_pos)

                mask = torch.ones(demo_dof_pos.size(1), dtype=torch.bool)
                mask[[4, 5, 10, 11]] = False
                demo_dof_pos = demo_dof_pos[:,mask]

                
                dof_offsets = None
                # Replace obs_demo1 with your actual demo observation data.
                
                obs_demo1 = build_demo_observations(root_pos, root_rot, root_vel, root_ang_vel, demo_dof_pos, dof_vel, key_body_pos, local_key_body_pos, dof_offsets).reshape(1, -1)
                
                # obs_demo1[:,19]=0.4   #x velocity
                # obs_demo1[:,20]=0.4    #y velocity
        
                print("obs_demo1 ", obs_demo1.shape)
                # # Open the file in append mode and write the values.
                # with open("demo_log_teleop.txt", "a") as f:
                #     f.write("dof_pos:\n")
                #     f.write(str(obs_demo1[:23]) + "\n")
                #     f.write("local_key_body_pos:\n")
                #     f.write(str(obs_demo1[-36:]) + "\n")
                #     f.write("-" * 40 + "\n")

                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]
                # print("quat ",quat)


                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)





                roll_x, pitch_y, yaw_z = euler_from_quaternion_new(quat)
                # print("roll_x, pitch_y, yaw_z ", roll_x, pitch_y, yaw_z)

                target_yaw =0
                # target_yaw = motions_env.target_yaw
                obs_demo = obs_demo1.squeeze(0).numpy()  # Convert to NumPy shape [64]
                # obs_demo = np.array([ 9.2327e-02, -8.0047e-02, -4.5305e-02, -1.2066e-01,  6.8453e-02,
                #                         3.3536e-02, -4.2229e-02, -6.1153e-02,  4.7212e-02,  5.0201e-03,
                #                         4.3613e-02, -6.6849e-02,  1.3842e-01, -3.5182e-01,  1.2539e+00,
                #                         -7.9236e-02, -1.1733e-01,  2.4667e-01,  1.1676e+00,  7.5185e-03,
                #                         -2.5311e-03,  1.5537e-03,  2.6594e-04,  2.5125e-02, -7.1149e-04,
                #                         4.9025e-03, -2.5057e-02,  7.2739e-01, -6.4723e-03,  6.4681e-02,
                #                         -8.1457e-02, -1.6231e-02,  9.8279e-02, -4.1653e-01, -2.1423e-02,
                #                         7.5010e-02, -7.1559e-01,  3.4317e-03, -6.3820e-02, -8.3817e-02,
                #                         -2.3783e-02, -1.0309e-01, -4.1971e-01, -2.5463e-02, -9.3612e-02,
                #                         -7.1956e-01,  8.0917e-03,  9.9057e-02,  3.1015e-01,  2.4128e-02,
                #                         1.6404e-01,  1.2528e-01,  7.8441e-02,  1.8207e-01, -9.3122e-02,
                #                         1.8080e-02, -1.0112e-01,  3.1067e-01,  4.2900e-02, -1.5911e-01,
                #                         1.2563e-01,  1.1620e-01, -1.6577e-01, -8.7809e-02])
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
                    action,  # Last action
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
                priv_latent = np.array([ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  9.7470e-01,                                                                                
                                        6.8799e-02, -1.0730e-01,  8.7041e-02, -6.2713e-02,  1.6226e-01,          
                                        5.8429e-02, -1.6696e-01,  2.8117e-03,  2.7620e-02,  1.5985e-01,          
                                        1.2579e-01, -1.3900e-01, -1.3811e-01,  1.2472e-01, -3.5774e-02,          
                                        -1.4318e-01,  1.1866e-01, -1.1778e-01, -3.9310e-02,  1.7975e-02,          
                                        -1.3652e-01,  1.7772e-01,  1.9462e-01,  4.4546e-02, -6.2652e-03,          
                                        -7.5512e-02, -1.8965e-01, -1.4859e-01,  1.6633e-01,  7.5209e-03,          
                                        4.0745e-02, -3.0808e-02,  6.3970e-02, -1.4891e-01, -1.0964e-01,          
                                        -1.6342e-01, -2.3097e-02,  4.1836e-02,  3.4393e-02, -1.0548e-01,
                                        -2.3853e-02,  8.8947e-02,  8.6490e-02, -3.9195e-02,  8.9053e-02,
                                        -1.5109e-01])




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


                motion_features = obs_history_buf[-prop_hist_len:].flatten()



                # Final concatenation (everything in NumPy)
                obs_buf1 = np.concatenate([motion_features, obs_buf, obs_demo, priv_explicit, priv_latent, obs_history_buf.reshape(1, -1).flatten()])

                # obs_tensor = obs_buf  # Now a NumPy array
                obs_tensor = torch.from_numpy(obs_buf1).unsqueeze(0)
                obs_tensor = torch.tensor(obs_tensor, dtype=torch.float32)

                obs_tensor = obs_tensor[:, motions_cfg.env.n_feature:]






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
                
                # action[5] = 0

                # print("Raw actions 1:", action)
                # action = action * np.pi / 180  # Convert degrees to radians

                clip_actions = motions_cfg.normalization.clip_actions / motions_cfg.control.action_scale
                action = np.clip(action, -clip_actions, clip_actions)
                # print("Raw actions:", action)



                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
