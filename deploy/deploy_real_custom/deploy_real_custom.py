from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil


from legged_gym import LEGGED_GYM_ROOT_DIR
from typing import Union
import numpy as np
import time
import torch

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data, euler_from_quaternion_new
from common.remote_controller import RemoteController, KeyMap
from config import Config


from motions import Motions
from motions_config import MotionsCfg

class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()

        # Initialize the policy network
        self.policy = torch.jit.load(config.policy_path)
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)

        # Initialize smoothed_action (e.g., in __init__)
        self.smoothed_action = np.zeros_like(self.action)
        self.raw_action = np.zeros_like(self.action)

        self.default_angles = np.concatenate([self.config.leg_joint_target, self.config.arm_waist_target])
        self.target_dof_pos = self.default_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0
        self.episode_length_buf = 0
        self.motions_cfg = MotionsCfg()
        self.motions_env = Motions(self.motions_cfg)
        history_len = 10
        self.obs_history_buf = np.zeros_like(np.stack([np.zeros(78)] * history_len, axis=0))

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            # h1 uses the go msg type
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        else:
            raise ValueError("Invalid msg_type")

        # wait for the subscriber to receive data
        self.wait_for_low_state()

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx + self.config.wrist_joint2motor_idx
        kps = self.config.kps + self.config.arm_waist_kps + self.config.wrist_kps
        kds = self.config.kds + self.config.arm_waist_kds + self.config.wrist_kds
        default_pos = np.concatenate((self.config.leg_joint_target, self.config.arm_waist_target, self.config.wrist_target), axis=0)
        dof_size = len(dof_idx)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.leg_joint_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            for i in range(len(self.config.wrist_joint2motor_idx)):
                motor_idx = self.config.wrist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.wrist_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.wrist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.wrist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def run(self):
        # self.counter += 1
        # self.episode_length_buf = self.counter
        control_decimation = 4


        # if self.counter % control_decimation == 0:

        self.motions_env._motion_times += self.motions_env._motion_dt
        self.motions_env._motion_times[self.motions_env._motion_times >= self.motions_env._motion_lengths] = 0.
        self.motions_env.update_demo_obs()
        obs_demo1 = self.motions_env.compute_obs_demo()


        # Get the current joint position and velocity
        # NEED TO CHANGE THIS TO HAVE ALL THE MOTORS AND DOUBLE CHECK THE INDICES
        for i in range(len(self.config.leg_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq

        for j in range(len(self.config.arm_waist_joint2motor_idx)):
            i = j + len(self.config.leg_joint2motor_idx)
            self.qj[i] = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[j]].q
            self.dqj[i] = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[j]].dq


        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)
        
        # q_tensor = torch.from_numpy(np.array([quat[1],quat[2],quat[3],quat[0]])).float().unsqueeze(0)    # shape becomes (1, 4)
        # v_tensor = torch.from_numpy(ang_vel).float()  # shape becomes (1, 3)

        # print("q_tensor ", q_tensor)
        # print("v_tensor ", v_tensor)

        # ang_vel = quat_rotate_inverse(q_tensor, v_tensor)
        # # ang_vel = ang_vel.squeeze(0)  # if you want a 1D result
        # print("ang_vel ", ang_vel)

        if self.config.imu_type == "torso":
            # h1 and h1_2 imu is on the torso
            # imu data needs to be transformed to the pelvis frame
            waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)

        # create observation
        gravity_orientation = get_gravity_orientation(quat)
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        qj_obs = (qj_obs - self.default_angles) * self.config.dof_pos_scale
        dqj_obs = dqj_obs * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale


        num_actions = self.config.num_actions


        roll_x, pitch_y, yaw_z =  euler_from_quaternion_new(quat)
        target_yaw = self.motions_env.target_yaw
        target_yaw = 0
        obs_demo = obs_demo1# Target motion from pre-recorded motion or from teleoperation  # NEEDS TO BE ADDED
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
        # 2.9852e-01,  3.0924e-01, -2.7392e-01,  4.0386e-01]).view(1, -1)

        # print("torch.tensor([ang_vel]) ", torch.tensor([ang_vel]).shape)
        # print("roll_x ", roll_x.shape)
        # print("pitch_y ", pitch_y.shape)
        # print("torch.sin(yaw_z - target_yaw) ", torch.sin(yaw_z - target_yaw).shape)
        # print("torch.cos(yaw_z - target_yaw) ", torch.cos(yaw_z - target_yaw).shape)
        # print("torch.tensor(qj_obs) ", torch.tensor(qj_obs).shape)
        # print("torch.tensor(dqj_obs) ", torch.tensor(dqj_obs).shape)
        # print("torch.tensor(self.action) ", torch.tensor(self.action).shape)
        # print("torch.tensor([-0.5]) ", torch.tensor([-0.5]).shape)
        # print("torch.tensor([-0.5]) ", torch.tensor([-0.5]).shape)
                            

        obs_buf = torch.cat((
            torch.tensor([ang_vel]).view(-1),  # Fix shape from [1, 1, 3] to [3]
            # ang_vel.view(-1) ,  # Fix shape from [1, 1, 3] to [3]
            roll_x.unsqueeze(0),  # Fix scalar to 1D
            pitch_y.unsqueeze(0),  # Fix scalar to 1D
            torch.sin(yaw_z - target_yaw).unsqueeze(0),  # Fix scalar to 1D
            torch.cos(yaw_z - target_yaw).unsqueeze(0),  # Fix scalar to 1D
            torch.tensor(qj_obs),
            torch.tensor(dqj_obs),
            # torch.tensor(self.action),
            torch.tensor(self.raw_action),
            torch.tensor([-0.5]),
            torch.tensor([-0.5]),
        ), dim=-1)




        # obs_buf = torch.cat((torch.tensor([ang_vel]),
        #                     roll_x.unsqueeze(0), pitch_y.unsqueeze(0), #imu_obs
        #                     torch.sin(yaw_z - target_yaw).unsqueeze(0),  # NEEDS TO BE ADDED
        #                     torch.cos(yaw_z - target_yaw).unsqueeze(0),  # NEEDS TO BE ADDED
        #                     torch.tensor(qj_obs).view(-1),
        #                     torch.tensor(dqj_obs).view(-1),
        #                     torch.tensor(self.action).view(-1),   # Last action
        #                     torch.tensor([-0.5]), torch.tensor([-0.5]), # self.contact_filt.float()*0-0.5         # should be a scalar if I'm not mistaken 
        #                     ),dim=-1)


        # obs_demo = # Target motion from pre-recorded motion or from teleoperation  # NEEDS TO BE ADDED

        priv_explicit = torch.zeros(3)


        # motor_strength_range = [0.8, 1.2]
        # str_rng = motor_strength_range
        # self.motor_strength = (str_rng[1] - str_rng[0]) * torch.rand(2, self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) + str_rng[0]

        priv_latent = torch.zeros(4+1+num_actions+num_actions)  # for now   # NEEDS TO BE CHANGED OR UNDERSTOOD


        if self.episode_length_buf <= 1:
            # For a new episode, initialize the history by repeating the urrent observation.
            history_len = 10
            self.obs_history_buf = torch.stack([obs_buf] * history_len, dim=0)
        else:
            # For an ongoing episode, shift the history by dropping the oldest observation and appending the new one.
            self.obs_history_buf = torch.cat([
                self.obs_history_buf[1:],      # Drop the first (oldest) observation.
                obs_buf.unsqueeze(0)           # Add the new observation at the end.
            ], dim=0)

        prop_hist_len = 4
        motion_features = self.obs_history_buf[-prop_hist_len:].flatten()

        # print("motion_features shape:", motion_features.shape)
        # print("obs_buf shape:", obs_buf.shape)
        # print("obs_demo shape:", obs_demo.shape)
        # print("priv_explicit shape:", priv_explicit.shape)
        # print("priv_latent shape:", priv_latent.shape)
        # print("self.obs_history_buf shape:", self.obs_history_buf.view(1, -1).shape)


        # Unsqueeze 1D tensors to make them 2D
        motion_features = motion_features.unsqueeze(0)  # Shape [1, 312]
        obs_buf = obs_buf.unsqueeze(0)  # Shape [1, 78]
        priv_explicit = priv_explicit.unsqueeze(0)  # Shape [1, 3]
        priv_latent = priv_latent.unsqueeze(0)  # Shape [1, 51]

        obs_demo = obs_demo.to('cpu')
        # self.obs_buf = torch.cat([motion_features.to(self.device), obs_buf.to(self.device), obs_demo.to(self.device), priv_explicit.to(self.device), priv_latent.to(self.device), obs_history_buf.view(1, -1).to(self.device)], dim=-1)
        self.obs_buf = torch.cat([motion_features, obs_buf, obs_demo, priv_explicit, priv_latent, self.obs_history_buf.view(1, -1)], dim=-1)



        # self.obs[:3] = ang_vel
        # self.obs[3:6] = gravity_orientation
        # self.obs[6:9] = self.cmd * self.config.cmd_scale * self.config.max_cmd
        # self.obs[9 : 9 + num_actions] = qj_obs
        # self.obs[9 + num_actions : 9 + num_actions * 2] = dqj_obs
        # self.obs[9 + num_actions * 2 : 9 + num_actions * 3] = self.action
        # self.obs[9 + num_actions * 3] = sin_phase
        # self.obs[9 + num_actions * 3 + 1] = cos_phase

        # # Get the action from the policy network
        # obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)


        obs_tensor =  self.obs_buf #.unsqueeze(0)?
    
        obs_tensor = obs_tensor[:, self.motions_cfg.env.n_feature:]

        self.action = self.policy(obs_tensor).detach().numpy().squeeze()
        print("self.action: ",  self.action)


        self.raw_action = self.action.copy()
        # Then in your run() method, after obtaining self.action from the policy:
        smoothing_factor = 0.05  # adjust between 0 (more smoothing) and 1 (no smoothing)
        self.smoothed_action[12:] = (
            smoothing_factor * self.action[12:] + (1 - smoothing_factor) * self.smoothed_action[12:]
        )
        self.action[12:] = self.smoothed_action[12:].copy()

        smoothing_factor = 0.1  # adjust between 0 (more smoothing) and 1 (no smoothing)
        self.smoothed_action[:12] = (
            smoothing_factor * self.action[:12] + (1 - smoothing_factor) * self.smoothed_action[:12]
        )
        self.action[:12] = self.smoothed_action[:12].copy()

        clip_actions = self.motions_cfg.normalization.clip_actions / self.motions_cfg.control.action_scale
        # self.action = np.clip(self.action, -0.5, 0.5)
        # self.action = np.clip(self.action, -1, 1)
        self.action = np.clip(self.action, -2, 2)
        # print("self.action after: ",  self.action)


        # transform action to target_dof_pos
        self.target_dof_pos = self.default_angles + self.action * self.config.action_scale

        # print("target_dof_pos before: ", target_dof_pos)


        # print("target_dof_pos after: ", target_dof_pos)
        self.counter += 1
        self.episode_length_buf = self.counter

        # Build low cmd
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            # Upper body only
            # self.low_cmd.motor_cmd[motor_idx].q = 0 #target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            motor_idx = self.config.arm_waist_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.target_dof_pos[i+len(self.config.leg_joint2motor_idx)]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        for i in range(len(self.config.wrist_joint2motor_idx)):
            motor_idx = self.config.wrist_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.config.wrist_target[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.wrist_kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.wrist_kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # send the command
        self.send_cmd(self.low_cmd)

        time.sleep(self.config.control_dt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="g1.yaml")
    args = parser.parse_args()

    # Load config
    config_path = f"/home/schakkal/expressive-humanoid/deploy/deploy_real_custom/configs/{args.config}"
    config = Config(config_path)

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)

    controller = Controller(config)

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()

    while True:
        try:
            # start_time = time.time()
            controller.run()
            # end_time = time.time()

            # # Calculate and print control frequency
            # loop_time = end_time - start_time
            # frequency = 1.0 / loop_time
            # print(f"Control Frequency: {frequency:.2f} Hz")

            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
