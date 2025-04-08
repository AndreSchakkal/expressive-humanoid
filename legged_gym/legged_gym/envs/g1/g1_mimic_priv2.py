from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch, torchvision

from legged_gym import LEGGED_GYM_ROOT_DIR, ASE_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.envs.base.legged_robot import LeggedRobot, euler_from_quaternion
from legged_gym.utils.math import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

import sys
sys.path.append(os.path.join(ASE_DIR, "ase"))
sys.path.append(os.path.join(ASE_DIR, "ase/utils"))
import cv2

from motion_lib import MotionLib
import torch_utils

class G1MimicPriv2(LeggedRobot):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True
        self.init_done = False
        self._parse_cfg(self.cfg)
        
        # Pre init for motion loading
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'
        
        self.init_motions(cfg)
        if cfg.motion.num_envs_as_motions:
            self.cfg.env.num_envs = self._motion_lib.num_motions()
        self.base_ang_vel_list = []

        BaseTask.__init__(self, self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        # print("self.dof_posself.dof_posself.dof_posself.dof_pos ", self.dof_pos)
        self._prepare_reward_function()
        self.init_done = True
        self.global_counter = 0
        self.total_env_steps_counter = 0

        self.init_motion_buffers(cfg)
        # self.rand_vx_cmd = 4*torch.rand((self.num_envs, ), device=self.device) - 2

        self.reset_idx(torch.arange(self.num_envs, device=self.device), init=True)
        # print("self.dof_posself.dof_posself.dof_posself.dof_pos ", self.dof_pos)
        self.post_physics_step()
        # print("self.dof_posself.dof_posself.dof_posself.dof_pos ", self.dof_pos)

    def _get_noise_scale_vec(self, cfg):
        print("self.cfg.env.n_proprio ", self.cfg.env.n_proprio)
        noise_scale_vec = torch.zeros(1, self.cfg.env.n_proprio, device=self.device)
        noise_scale_vec[:, :3] = self.cfg.noise.noise_scales.ang_vel
        noise_scale_vec[:, 3:5] = self.cfg.noise.noise_scales.imu
        noise_scale_vec[:, 7:7+self.num_dof] = self.cfg.noise.noise_scales.dof_pos
        noise_scale_vec[:, 7+self.num_dof:7+2*self.num_dof] = self.cfg.noise.noise_scales.dof_vel
        return noise_scale_vec
    
    def init_motions(self, cfg):
        ## GOES IN MOTIONLIB  -  NO USE
        self._key_body_ids = torch.tensor([3, 6, 9, 12], device=self.device)  #self._build_key_body_ids_tensor(key_bodies)

        # ['pelvis',
        # 'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link',
        # 'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link',
        # 'waist_yaw_link', 'waist_roll_link', 'torso_link',
        # 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_rubber_hand',
        # 'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_rubber_hand']

        ## LIST OF KEYBODY IDS
        self._key_body_ids_sim = torch.tensor([1, 4, 5, # Left Hip yaw, Knee, Ankle
                                                7, 10, 11,
                                                16, 19, 20, # Left Shoulder pitch, Elbow, hand
                                                21, 24, 25], device=self.device)         
# # ['pelvis', 'left_hip_yaw_link', 'left_hip_roll_link', 'left_hip_pitch_link', 'left_knee_link', 'left_ankle_link', 
# # 'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link', 'right_knee_link', 'right_ankle_link', 
# # 'torso_link', 
# # 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_hand_keypoint_link', 
# # 'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_hand_keypoint_link']
# self._key_body_ids_sim = torch.tensor([1, 4, 5, # Left Hip yaw, Knee, Ankle
#                                        6, 9, 10,
#                                        12, 15, 16, # Left Shoulder pitch, Elbow, hand
#                                        17, 20, 21], device=self.device)
        
        ## LIST OF INDICES OF THE KEYBODY IDS OF _key_body_ids_sim THAT ARE USED IN THE TRACKING REWARD
        self._key_body_ids_sim_subset = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], device=self.device)  # no knee and ankle
        # self._key_body_ids_sim_subset = torch.tensor([6, 7, 8, 9, 10, 11], device=self.device)  # no knee and ankle
# self._key_body_ids_sim_subset = torch.tensor([6, 7, 8, 9, 10, 11], device=self.device)  # no knee and ankle
       
        self._num_key_bodies = len(self._key_body_ids_sim_subset)


        ## GOES IN MOTIONLIB (represents ids of bodies in motion data data['skeleton_tree'])
        self._dof_body_ids = [1, 2, 3, # Hip, Knee, Ankle
                              4, 5, 6,
                              7,       # Torso
                              8, 9, 10, # Shoulder, Elbow, Hand
                              11, 12, 13]  # 13
# self._dof_body_ids = [1, 2, 3, # Hip, Knee, Ankle
#                       4, 5, 6,
#                       7,       # Torso
#                       8, 9, 10, # Shoulder, Elbow, Hand
#                       11, 12, 13]  # 13
        

        ## GOES IN MOTIONLIB (represents offsets of ids of bodies in robot links) ?????????? [maybe right]  # problem is there are offsets of 2 # maybe the solution would be to keep an offset of 3 and then remove it with valid_ids
        self._dof_offsets = [0, 3, 4, 7, 10, 11, 14, 
                             17, 
                             20, 21, 22, 25, 26, 27]  # 14
    # self._dof_offsets = [0, 3, 4, 6, 9, 10, 12, 
    #                         15, 
    #                         18, 19, 20, 23, 24, 25]  # 14
# self._dof_offsets = [0, 3, 4, 5, 8, 9, 10, 
#                      11, 
#                      14, 15, 16, 19, 20, 21]  # 14

        self._valid_dof_body_ids = torch.ones(len(self._dof_body_ids)+2*4+6, device=self.device, dtype=torch.bool)  ## CHANGE
# self._valid_dof_body_ids = torch.ones(len(self._dof_body_ids)+2*4, device=self.device, dtype=torch.bool)

        self._valid_dof_body_ids[-1] = 0    ## I THINK THIS IS THE right_rubber_hand
        self._valid_dof_body_ids[-6] = 0    ## I THINK THIS IS THE left_rubber_hand

        self._valid_dof_body_ids[6] = 0    ## DISCARD ADDITIONAL ROTATION FOR left_ankle yaw  #MAYBE
        self._valid_dof_body_ids[13] = 0    ## DISCARD ADDITIONAL ROTATION FOR right_ankle yaw
# self._valid_dof_body_ids[-1] = 0    ## I THINK THIS IS THE right_hand_keypoint_link
# self._valid_dof_body_ids[-6] = 0    ## I THINK THIS IS THE left_hand_keypoint_link

        self.dof_indices_sim = torch.tensor([0, 1, 2,    4, 5, 6,    7, 8, 9,    14, 15, 16,    17, 18, 19,    22, 23, 24], device=self.device, dtype=torch.long)
        self.dof_indices_motion = torch.tensor([1, 0, 2,    5, 4, 6,   8,7,9,    15, 14, 16,     18, 17, 19,   23, 22, 24], device=self.device, dtype=torch.long)
        # self.dof_indices_sim = torch.tensor([0, 1, 2, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19, 22, 23, 24], device=self.device, dtype=torch.long)
        # self.dof_indices_motion = torch.tensor([2, 0, 1, 6, 4, 5, 9, 7, 8, 16, 14, 15, 19, 17, 18, 24, 22, 23], device=self.device, dtype=torch.long)
# self.dof_indices_sim = torch.tensor([0, 1, 2, 5, 6, 7, 11, 12, 13, 16, 17, 18], device=self.device, dtype=torch.long)
# self.dof_indices_motion = torch.tensor([2, 0, 1, 7, 5, 6, 12, 11, 13, 17, 16, 18], device=self.device, dtype=torch.long)
        
        # self._dof_ids_subset = torch.tensor([0, 1, 2, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18], device=self.device)  # no knee and ankle
        
        
        ## LIST OF DOF IDS THAT ARE USED IN THE TRACKING REWARD
        # no ankle
        self._dof_ids_subset = torch.tensor([0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22], device=self.device)  # no knee and ankle
        # self._dof_ids_subset = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22], device=self.device)  # no knee and ankle
        # self._dof_ids_subset = torch.tensor([15, 16, 17, 18, 19, 20, 21, 22], device=self.device)  # no knee and ankle
# self._dof_ids_subset = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 18], device=self.device)  # no knee and ankle
        self._n_demo_dof = len(self._dof_ids_subset)

#['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint', 
#'left_knee_joint', 'left_ankle_joint', 
#'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint', 
#'right_knee_joint', 'right_ankle_joint', 
#'torso_joint', 
#'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 
#'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint']

        # ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
        # 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
        # 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
        # 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
        # 'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
        # 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint',
        # 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint']
        # self.dof_ids_subset = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], device=self.device, dtype=torch.long)
        # motion_name = "17_04_stealth"
        if cfg.motion.motion_type == "single":
            motion_file = os.path.join(ASE_DIR, f"ase/poselib/data/g1_retarget_npy/{cfg.motion.motion_name}.npy")
        else:
            assert cfg.motion.motion_type == "yaml"
            motion_file = os.path.join(ASE_DIR, f"ase/poselib/data/configs/{cfg.motion.motion_name}")
        
        self._load_motion(motion_file, cfg.motion.no_keybody)

    def init_motion_buffers(self, cfg):
        num_motions = self._motion_lib.num_motions()
        self._motion_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._motion_ids = torch.remainder(self._motion_ids, num_motions)
        if cfg.motion.motion_curriculum:
            self._max_motion_difficulty = 9
            # self._motion_ids = self._motion_lib.sample_motions(self.num_envs, self._max_motion_difficulty)
        else:
            self._max_motion_difficulty = 9
        self._motion_times = self._motion_lib.sample_time(self._motion_ids)
        self._motion_lengths = self._motion_lib.get_motion_length(self._motion_ids)
        self._motion_difficulty = self._motion_lib.get_motion_difficulty(self._motion_ids)
        # self._motion_features = self._motion_lib.get_motion_features(self._motion_ids)

        self._motion_dt = self.dt


        self._motion_num_future_steps = self.cfg.env.n_demo_steps
        self._motion_demo_offsets = torch.arange(0, self.cfg.env.n_demo_steps * self.cfg.env.interval_demo_steps, self.cfg.env.interval_demo_steps, device=self.device)
        self._demo_obs_buf = torch.zeros((self.num_envs, self.cfg.env.n_demo_steps, self.cfg.env.n_demo), device=self.device)
        self._curr_demo_obs_buf = self._demo_obs_buf[:, 0, :]
        self._next_demo_obs_buf = self._demo_obs_buf[:, 1, :]
        # self._curr_mimic_obs_buf = torch.zeros_like(self._curr_demo_obs_buf, device=self.device)

        self._curr_demo_root_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._curr_demo_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self._curr_demo_root_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self._curr_demo_keybody = torch.zeros((self.num_envs, self._num_key_bodies, 3), device=self.device)
        self._in_place_flag = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.dof_term_threshold = 3 * torch.ones(self.num_envs, device=self.device)
        self.keybody_term_threshold = 0.3 * torch.ones(self.num_envs, device=self.device)
        self.yaw_term_threshold = 0.5 * torch.ones(self.num_envs, device=self.device)
        self.height_term_threshold = 0.2 * torch.ones(self.num_envs, device=self.device)

        # self.step_inplace_ids = self.resample_step_inplace_ids()
    
    def _load_motion(self, motion_file, no_keybody=False):
        # assert(self._dof_offsets[-1] == self.num_dof + 2)  # +2 for hand dof not used
        self._motion_lib = MotionLib(motion_file=motion_file,
                                     dof_body_ids=self._dof_body_ids,
                                     dof_offsets=self._dof_offsets,
                                     key_body_ids=self._key_body_ids.cpu().numpy(), 
                                     device=self.device, 
                                     no_keybody=no_keybody, 
                                     regen_pkl=self.cfg.motion.regen_pkl)
        return
    
    def step(self, actions):
        actions = self.reindex(actions)

        actions.to(self.device)
        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)
        if self.cfg.domain_rand.action_delay:
            if self.global_counter % self.cfg.domain_rand.delay_update_global_steps == 0:
                if len(self.cfg.domain_rand.action_curr_step) != 0:
                    self.delay = torch.tensor(self.cfg.domain_rand.action_curr_step.pop(0), device=self.device, dtype=torch.float)
            if self.viewer:
                self.delay = torch.tensor(self.cfg.domain_rand.action_delay_view, device=self.device, dtype=torch.float)
            # self.delay = torch.randint(0, 3, (1,), device=self.device, dtype=torch.float)
            indices = -self.delay -1
            actions = self.action_history_buf[:, indices.long()] # delay for 1/50=20ms

        self.global_counter += 1
        self.total_env_steps_counter += 1
        clip_actions = self.cfg.normalization.clip_actions / self.cfg.control.action_scale

        # print("Raw actions:", actions)


        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.render()
        # print("actions:", actions)
        
        # print("Actions:", actions.detach().cpu().numpy())
        # print("Action mean:", actions.mean().item(), "Action std:", actions.std().item())

        # self.actions[:, [4, 9]] = torch.clamp(self.actions[:, [4, 9]], -0.5, 0.5)
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            
            # print("self.torques ", self.torques)
            
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        # for i in torch.topk(self.torques[self.lookat_id], 3).indices.tolist():
        #     print(self.dof_names[i], self.torques[self.lookat_id][i])
        
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        if self.cfg.depth.use_camera and self.global_counter % self.cfg.depth.update_interval == 0:
            self.extras["depth"] = self.depth_buffer[:, -2]  # have already selected last one
        else:
            self.extras["depth"] = None
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def resample_motion_times(self, env_ids):
        return self._motion_lib.sample_time(self._motion_ids[env_ids])
    
    def update_motion_ids(self, env_ids):
        self._motion_times[env_ids] = self.resample_motion_times(env_ids)
        self._motion_lengths[env_ids] = self._motion_lib.get_motion_length(self._motion_ids[env_ids])
        self._motion_difficulty[env_ids] = self._motion_lib.get_motion_difficulty(self._motion_ids[env_ids])

    def reset_idx(self, env_ids, init=False):
        if len(env_ids) == 0:
            return
        # RSI
        if self.cfg.motion.motion_curriculum:
            # ep_length = self.episode_length_buf[env_ids] * self.dt
            completion_rate = self.episode_length_buf[env_ids] * self.dt / self._motion_lengths[env_ids]
            completion_rate_mean = completion_rate.mean()
            # if completion_rate_mean > 0.8:
            #     self._max_motion_difficulty = min(self._max_motion_difficulty + 1, 9)
            #     self._motion_ids[env_ids] = self._motion_lib.sample_motions(len(env_ids), self._max_motion_difficulty)
            # elif completion_rate_mean < 0.4:
            #     self._max_motion_difficulty = max(self._max_motion_difficulty - 1, 0)
            #     self._motion_ids[env_ids] = self._motion_lib.sample_motions(len(env_ids), self._max_motion_difficulty)
            relax_ids = completion_rate < 0.3
            strict_ids = completion_rate > 0.9
            # self.dof_term_threshold[env_ids[relax_ids]] += 0.05
            self.dof_term_threshold[env_ids[strict_ids]] -= 0.05
            self.dof_term_threshold.clamp_(1.5, 3)

            self.height_term_threshold[env_ids[relax_ids]] += 0.01
            self.height_term_threshold[env_ids[strict_ids]] -= 0.01
            self.height_term_threshold.clamp_(0.03, 0.1)

            relax_ids = completion_rate < 0.6
            strict_ids = completion_rate > 0.9
            self.keybody_term_threshold[env_ids[relax_ids]] -= 0.05
            self.keybody_term_threshold[env_ids[strict_ids]] += 0.05
            self.keybody_term_threshold.clamp_(0.1, 0.4)

            relax_ids = completion_rate < 0.4
            strict_ids = completion_rate > 0.8
            self.yaw_term_threshold[env_ids[relax_ids]] -= 0.05
            self.yaw_term_threshold[env_ids[strict_ids]] += 0.05
            self.yaw_term_threshold.clamp_(0.1, 0.6)


        self.update_motion_ids(env_ids)

        motion_ids = self._motion_ids[env_ids]
        motion_times = self._motion_times[env_ids]
        root_pos, root_rot, dof_pos_motion, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        
        # Intialize dof state from default position and reference position
        dof_pos_motion, dof_vel = self.reindex_dof_pos_vel(dof_pos_motion, dof_vel)

        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)

        # reset robot states
        # print("dof_pos_motion ", dof_pos_motion)
        # print("dof_vel ", dof_vel)
        # print("root_vel ", root_vel)
        # print("root_rot ", root_rot)
        # print("root_pos[:, 2] ", root_pos[:, 2])
        self._reset_dofs(env_ids, dof_pos_motion, dof_vel)
        self._reset_root_states(env_ids, root_vel, root_rot, root_pos[:, 2])

        if init:
            self.init_root_pos_global = self.root_states[:, :3].clone()
            self.init_root_pos_global_demo = root_pos[:].clone()
            self.target_pos_abs = self.init_root_pos_global.clone()[:, :2]
        else:
            self.init_root_pos_global[env_ids] = self.root_states[env_ids, :3].clone()
            self.init_root_pos_global_demo[env_ids] = root_pos[:].clone()
            self.target_pos_abs[env_ids] = self.init_root_pos_global[env_ids].clone()[:, :2]

        self._resample_commands(env_ids)  # no resample commands
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.last_root_vel[:] = 0.
        self.feet_air_time[env_ids] = 0.
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.  # reset obs history buffer TODO no 0s
        self.contact_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.
        self.cur_goal_idx[env_ids] = 0
        self.reach_goal_timer[env_ids] = 0

        # fill extras
        self.extras["episode"] = {}
        self.extras["episode"]["curriculum_completion"] = completion_rate_mean
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.episode_length_buf[env_ids] = 0

        self.extras["episode"]["curriculum_motion_difficulty_level"] = self._max_motion_difficulty
        self.extras["episode"]["curriculum_dof_term_thresh"] = self.dof_term_threshold.mean()
        self.extras["episode"]["curriculum_keybody_term_thresh"] = self.keybody_term_threshold.mean()
        self.extras["episode"]["curriculum_yaw_term_thresh"] = self.yaw_term_threshold.mean()
        self.extras["episode"]["curriculum_height_term_thresh"] = self.height_term_threshold.mean()
        
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        return
                                                                                                                                                                                                                                                                                                                                                                   
    def _reset_dofs(self, env_ids, dof_pos, dof_vel):
        
        # dof_pos_default = self.default_dof_pos + torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dof), device=self.device) * self.default_dof_pos
        self.dof_pos[env_ids] = dof_pos
        self.dof_vel[env_ids] = dof_vel

        # self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(0., 0.5, (len(env_ids), self.num_dof), device=self.device)
        # self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
    def post_physics_step(self):
        # self._motion_sync()
        super().post_physics_step()

        # step motion lib
        self._motion_times += self._motion_dt
        self._motion_times[self._motion_times >= self._motion_lengths] = 0.
        self.update_demo_obs()
        # self.update_mimic_obs()
        
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.draw_rigid_bodies_demo()
            self.draw_rigid_bodies_actual()
            self._draw_goals()

        return

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        if self.common_step_counter % int(self.cfg.domain_rand.gravity_rand_interval) == 0:
            self._randomize_gravity()
        if self.common_step_counter % self.cfg.motion.resample_step_inplace_interval == 0:
            self.resample_step_inplace_ids()
    
    def resample_step_inplace_ids(self, ):
        self.step_inplace_ids = torch.rand(self.num_envs, device=self.device) < self.cfg.motion.step_inplace_prob
    
    def _randomize_gravity(self, external_force = None):
        if self.cfg.domain_rand.randomize_gravity and external_force is None:
            min_gravity, max_gravity = self.cfg.domain_rand.gravity_range
            external_force = torch.rand(3, dtype=torch.float, device=self.device,
                                        requires_grad=False) * (max_gravity - min_gravity) + min_gravity


        sim_params = self.gym.get_sim_params(self.sim)
        gravity = external_force + torch.Tensor([0, 0, -9.81]).to(self.device)
        self.gravity_vec[:, :] = gravity.unsqueeze(0) / torch.norm(gravity)
        sim_params.gravity = gymapi.Vec3(gravity[0], gravity[1], gravity[2])
        self.gym.set_sim_params(self.sim, sim_params)
    
    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.cfg.domain_rand.gravity_rand_interval = np.ceil(self.cfg.domain_rand.gravity_rand_interval_s / self.dt)
        self.cfg.motion.resample_step_inplace_interval = np.ceil(self.cfg.motion.resample_step_inplace_interval_s / self.dt)

    def _update_goals(self):
        # self.target_pos_abs = (self._curr_demo_root_pos - self.init_root_pos_global_demo + self.init_root_pos_global)[:, :2]
        # self.target_pos_rel = self.target_pos_abs - self.root_states[:, :2]
        reset_target_pos = self.episode_length_buf % (self.cfg.motion.global_keybody_reset_time // self.dt) == 0
        self.target_pos_abs[reset_target_pos] = self.root_states[reset_target_pos, :2]
        self.target_pos_abs += (self._curr_demo_root_vel * self.dt)[:, :2]
        self.target_pos_rel = global_to_local_xy(self.yaw[:, None], self.target_pos_abs - self.root_states[:, :2])
        # print(self.target_pos_rel[self.lookat_id])
        r, p, y = euler_from_quaternion(self._curr_demo_quat)
        self.target_yaw = y.clone()
        # self.desired_vel_scalar = torch.norm(self._curr_demo_obs_buf[:, self.num_dof:self.num_dof+2], dim=-1)

    
    def update_demo_obs(self):
        demo_motion_times = self._motion_demo_offsets + self._motion_times[:, None]  # [num_envs, demo_dim]
        # print(demo_motion_times)
        # print()
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, local_key_body_pos \
            = self._motion_lib.get_motion_state(self._motion_ids.repeat_interleave(self._motion_num_future_steps), demo_motion_times.flatten(), get_lbp=True)
        dof_pos, dof_vel = self.reindex_dof_pos_vel(dof_pos, dof_vel)

        # root_vel[:,[0,1,2]] = torch.tensor([0.,-5.,0.]).to('cuda:0')
        # print(root_vel, root_vel.shape)

        
        self._curr_demo_root_pos[:] = root_pos.view(self.num_envs, self._motion_num_future_steps, 3)[:, 0, :]
        self._curr_demo_quat[:] = root_rot.view(self.num_envs, self._motion_num_future_steps, 4)[:, 0, :]
        self._curr_demo_root_vel[:] = root_vel.view(self.num_envs, self._motion_num_future_steps, 3)[:, 0, :]
        self._curr_demo_keybody[:] = local_key_body_pos[:, self._key_body_ids_sim_subset].view(self.num_envs, self._motion_num_future_steps, self._num_key_bodies, 3)[:, 0, :, :]
        # self._in_place_flag = 0*(torch.norm(self._curr_demo_root_vel, dim=-1) < 0.2)
        self._in_place_flag = torch.norm(self._curr_demo_root_vel, dim=-1) < 0.2
        # for i in range(13):
        #     feet_pos_global = key_pos[:, i]# - root_pos + self.root_states[:, :3]
        #     pose = gymapi.Transform(gymapi.Vec3(feet_pos_global[self.lookat_id, 0], feet_pos_global[self.lookat_id, 1], feet_pos_global[self.lookat_id, 2]), r=None)
        #     gymutil.draw_lines(edge_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)

        demo_obs = build_demo_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos[:, self._dof_ids_subset], dof_vel, key_pos, local_key_body_pos[:, self._key_body_ids_sim_subset, :], self._dof_offsets)
        self._demo_obs_buf[:] = demo_obs.view(self.num_envs, self.cfg.env.n_demo_steps, self.cfg.env.n_demo)[:]
        # self._demo_obs_buf[:, :,  self._n_demo_dof:self._n_demo_dof+3] = torch.tensor([0.,0.,0.])

    
    def compute_obs_buf(self):
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        # print()
        # print("!!!!!!!!!")
        # print("omega", self.base_ang_vel  * self.obs_scales.ang_vel)
        # print("[roll_x, pitch_y]", imu_obs)  # imu_obs
        # print("[sin_yaw, cos_yaw]", torch.sin(self.yaw - self.target_yaw)[:, None],torch.cos(self.yaw - self.target_yaw)[:, None],)  # NEEDS TO BE ADDED
        # print("qj", self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos))
        # print("dqj",self.reindex(self.dof_vel * self.obs_scales.dof_vel))
        # print("action",self.reindex(self.action_history_buf[:, -1]))  # Last action

        # self.base_ang_vel_list.append(self.base_ang_vel) 
        # print("self.base_ang_vel ", self.base_ang_vel)
        # print("self.base_ang_vel  * self.obs_scales.ang_vel ", self.base_ang_vel  * self.obs_scales.ang_vel)

        # stacked = torch.stack(self.base_ang_vel_list)  # shape: (N, D) if each tensor is D-dimensional.
        # elementwise_max = torch.max(stacked, dim=0).values

        # print("max base ang vel:", elementwise_max)
        # print("max base ang vel ", torch.max(self.base_ang_vel_list))


        # cur_key_body_pos_local = global_to_local(self.base_quat, self.rigid_body_states[:, :, :3], self.root_states[:, :3]).view(self.num_envs, -1)


        return torch.cat((#motion_id_one_hot,
                            self.base_ang_vel  * self.obs_scales.ang_vel,   #[1,3]
                            imu_obs,    #[1,2]
                            torch.sin(self.yaw - self.target_yaw)[:, None],  #[1,1]
                            torch.cos(self.yaw - self.target_yaw)[:, None],  #[1,1]  
                            self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),
                            self.reindex(self.dof_vel * self.obs_scales.dof_vel),
                            #####
                            self.rigid_body_states[:, self._key_body_ids_sim[self._key_body_ids_sim_subset], :3].view(self.num_envs, -1),
                            self.root_states[:, 7:10],
                            #####
                            self.reindex(self.action_history_buf[:, -1]),
                            self.reindex_feet(self.contact_filt.float()*0-0.5),
                            ),dim=-1)
    


    def compute_obs_demo(self):
        obs_demo = self._next_demo_obs_buf.clone()#self._demo_obs_buf.clone().flatten(start_dim=1)
        # obs_demo[self._in_place_flag, self._n_demo_dof:self._n_demo_dof+3] = 0
        return obs_demo
    
    def compute_observations(self):
        # motion_id_one_hot = torch.zeros((self.num_envs, self._motion_lib.num_motions()), device=self.device)
        # motion_id_one_hot[torch.arange(self.num_envs, device=self.device), self._motion_ids] = 1.
        
        obs_buf = self.compute_obs_buf()
        # print("obs_buf ", obs_buf.shape)
        # print("self.noise_scale_vec ", self.noise_scale_vec.shape)

        if self.cfg.noise.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec * self.cfg.noise.noise_scale
        
        obs_demo = self.compute_obs_demo()
        # obs_demo = torch.tensor([[-2.4745e-02,  1.2176e-01,  8.1200e-02, -6.5168e-02, -6.0663e-02,
        #  -1.1331e-01, -9.6272e-02,  1.2682e-02, -3.0531e-02, -6.9623e-02,
        #   9.4022e-02, -1.0719e-02,  2.0356e-01, -2.7878e-01,  1.0762e+00,
        #  -9.1577e-01, -1.3040e+00,  2.5616e-01,  2.0081e-01,  3.9499e-03,
        #  -4.9789e-03,  1.5506e-03,  2.9615e-02,  2.1360e-02, -1.5168e-01,
        #  -9.8864e-03,  7.7811e-02,  7.3771e-01,  8.3883e-04,  6.2207e-02,
        #  -8.2667e-02,  2.9788e-03,  1.4755e-01, -4.1746e-01,  3.1189e-02,
        #   1.8327e-01, -7.1399e-01,  1.5378e-03, -6.6695e-02, -8.2658e-02,
        #   1.1219e-02, -1.4511e-01, -4.1800e-01,  3.4049e-02, -1.7265e-01,
        #  -7.1587e-01,  2.8161e-02,  1.2105e-01,  2.9693e-01,  3.1996e-02,
        #   1.8049e-01,  1.0672e-01,  9.9245e-02,  1.8431e-01, -1.0877e-01,
        #   2.4633e-02, -7.8363e-02,  3.1682e-01,  1.0982e-01, -2.8419e-01,
        #   2.9852e-01,  3.0924e-01, -2.7392e-01,  4.0386e-01],
        # [-3.8790e-01, -6.5256e-02, -2.3366e-01,  3.3138e-01,  2.4324e-02,
        #  -3.1360e-02, -5.9318e-03,  8.8575e-01,  7.0752e-02, -7.3739e-02,
        #   1.1033e-01,  3.5094e-01,  2.3271e-01,  1.7048e-01,  9.4172e-01,
        #   1.5077e-01, -2.1636e-01,  8.6633e-02,  1.2443e-01,  1.4202e+00,
        #   2.2207e-01, -2.8978e-02,  1.5557e-01, -1.8797e-01,  1.5873e-01,
        #   7.4327e-02, -3.1586e-02,  7.5555e-01, -4.7287e-02,  5.5354e-02,
        #  -1.0510e-01,  8.4925e-02,  1.0847e-01, -4.1533e-01,  1.3842e-01,
        #   1.1922e-01, -7.1014e-01, -4.7943e-02, -7.3542e-02, -1.0431e-01,
        #  -8.6280e-02, -1.3644e-01, -4.3810e-01, -3.0924e-01, -1.3511e-01,
        #  -6.3828e-01, -1.6893e-02,  1.1154e-01,  2.7858e-01, -1.0254e-01,
        #   1.8530e-01,  1.1079e-01, -9.5726e-02,  2.5189e-01, -1.0482e-01,
        #  -5.8799e-03, -8.8035e-02,  2.9335e-01, -4.1166e-02, -1.8941e-01,
        #   1.2348e-01,  1.7839e-01, -1.9449e-01,  7.5293e-02]]).to(self.device)
        # obs_demo = np.zeros(64)  # Target motion from pre-recorded motion or from teleoperation  # NEEDS TO BE ADDED
        # print("obs_demo ", obs_demo)

        # print("obs_demo1", obs_demo)

        # obs_demo[:, :] = 0
        # obs_demo[:, -3*len(self._key_body_ids_sim_subset)-1] = 1
        # obs_demo[:, self._n_demo_dof:self._n_demo_dof+8] = 1 #self.rand_vx_cmd
        # obs_demo[:, -3*len(self._key_body_ids_sim_subset):] = torch.tensor([ 0.0049,  0.1554,  0.4300,  
        #                                                                      0.0258,  0.2329,  0.1076,  
        #                                                                      0.3195,  0.2040,  0.0537,  
        #                                                                      0.0061, -0.1553,  0.4300,  
        #                                                                      0.0292, -0.2305,  0.1076,  
        #                                                                      0.3225, -0.1892,  0.0598], device=self.device)
        motion_features = self.obs_history_buf[:, -self.cfg.env.prop_hist_len:].flatten(start_dim=1)#self._demo_obs_buf[:, 2:, :].clone().flatten(start_dim=1) 
        priv_explicit = torch.cat((0*self.base_lin_vel * self.obs_scales.lin_vel,
                                #    global_to_local(self.base_quat, self.rigid_body_states[:, self._key_body_ids_sim[self._key_body_ids_sim_subset], :3], self.root_states[:, :3]).view(self.num_envs, -1),
                                  ), dim=-1)
        priv_latent = torch.cat((
            self.mass_params_tensor,
            self.friction_coeffs_tensor,
            self.motor_strength[0] - 1, 
            self.motor_strength[1] - 1
        ), dim=-1)
        # print()
        # print()
        # print()
        # print("priv_latent ", priv_latent)
        # print()
        # print()
        # print()
        # priv_explicit = torch.zeros(3).view(1,-1).to(self.device)

        
        # priv_latent = torch.zeros(4 + 1 + 2*23)  # for now  # NEEDS TO BE CHANGED OR UNDERSTOOD
        # priv_latent = torch.tensor([ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  9.7470e-01,                                                                                
        #                         6.8799e-02, -1.0730e-01,  8.7041e-02, -6.2713e-02,  1.6226e-01,          
        #                         5.8429e-02, -1.6696e-01,  2.8117e-03,  2.7620e-02,  1.5985e-01,          
        #                         1.2579e-01, -1.3900e-01, -1.3811e-01,  1.2472e-01, -3.5774e-02,          
        #                         -1.4318e-01,  1.1866e-01, -1.1778e-01, -3.9310e-02,  1.7975e-02,          
        #                         -1.3652e-01,  1.7772e-01,  1.9462e-01,  4.4546e-02, -6.2652e-03,          
        #                         -7.5512e-02, -1.8965e-01, -1.4859e-01,  1.6633e-01,  7.5209e-03,          
        #                         4.0745e-02, -3.0808e-02,  6.3970e-02, -1.4891e-01, -1.0964e-01,          
        #                         -1.6342e-01, -2.3097e-02,  4.1836e-02,  3.4393e-02, -1.0548e-01,
        #                         -2.3853e-02,  8.8947e-02,  8.6490e-02, -3.9195e-02,  8.9053e-02,
        #                         -1.5109e-01]).view(1,-1).to(self.device)


        # cur_key_body_pos_local = global_to_local(self.base_quat, self.rigid_body_states[:, self._key_body_ids_sim[self._key_body_ids_sim_subset], :3], self.root_states[:, :3]).view(self.num_envs, -1)
        # , cur_key_body_pos_local, self.root_states[:, 7:10].view(self.num_envs, -1)

        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.3 - self.measured_heights, -1, 1.)
            self.obs_buf = torch.cat([motion_features, obs_buf, obs_demo, heights, priv_explicit, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        else:
            self.obs_buf = torch.cat([motion_features, obs_buf, obs_demo, priv_explicit, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        

        # self.obs_buf = torch.tensor([[    -0.0557903982698917388916016,     -1.1220515966415405273437500,
        #       0.4402205049991607666015625,      0.0510412007570266723632812,
        #       0.0310374330729246139526367,      0.2680881321430206298828125,
        #       0.9633944034576416015625000,      0.0325090736150741577148438,
        #      -0.0521629936993122100830078,     -0.1275454908609390258789062,
        #      -0.3071124851703643798828125,      0.2520418167114257812500000,
        #       0.0107025634497404098510742,      0.1901922225952148437500000,
        #      -0.1505199372768402099609375,      0.0044078803621232509613037,
        #      -0.2256443053483963012695312,      0.0111294835805892944335938,
        #       0.1048782989382743835449219,      0.0161452889442443847656250,
        #       0.0026774692814797163009644,      0.0506901182234287261962891,
        #       0.0189582332968711853027344,      0.0012786077568307518959045,
        #      -0.2442948818206787109375000,      0.7892524600028991699218750,
        #      -0.0368890464305877685546875,     -0.1292335987091064453125000,
        #       0.1403569877147674560546875,      0.7906432747840881347656250,
        #       0.2259670495986938476562500,      0.0170157384127378463745117,
        #      -0.0632177889347076416015625,     -0.0060159387066960334777832,
        #       0.0984176769852638244628906,     -0.4452678859233856201171875,
        #       0.3791607022285461425781250,      0.0036483779549598693847656,
        #      -0.1179847493767738342285156,     -0.1793670654296875000000000,
        #       0.0173094160854816436767578,     -0.0563375651836395263671875,
        #       0.1394380331039428710937500,     -0.0661912336945533752441406,
        #       0.2531404197216033935546875,     -0.0358539037406444549560547,
        #      -0.1195269227027893066406250,      0.6520054936408996582031250,
        #      -0.3662268817424774169921875,     -0.0776600390672683715820312,
        #      -0.0406240373849868774414062,      0.3502987325191497802734375,
        #       0.2701234519481658935546875,     -0.4650021791458129882812500,
        #      -0.0240667499601840972900391,     -0.8515438437461853027343750,
        #      -0.6589136719703674316406250,      4.4570612907409667968750000,
        #       9.7113609313964843750000000,      1.9048637151718139648437500,
        #       0.3162832558155059814453125,     -0.2193512320518493652343750,
        #      -0.5879993438720703125000000,      9.4803142547607421875000000,
        #       3.8904640674591064453125000,      0.6696988344192504882812500,
        #       0.0030622519552707672119141,     -0.2225416004657745361328125,
        #      -2.9281847476959228515625000,      2.9904303550720214843750000,
        #       0.9965088367462158203125000,      1.6217541694641113281250000,
        #       0.7642003297805786132812500,      2.1409063339233398437500000,
        #       1.3981121778488159179687500,      3.7922210693359375000000000,
        #      -0.5000000000000000000000000,     -0.5000000000000000000000000,
        #       0.0726586356759071350097656,     -1.5362514257431030273437500,
        #       1.0208513736724853515625000,      0.0532127767801284790039062,
        #       0.0218555331230163574218750,      0.3198881745338439941406250,
        #       0.9474552869796752929687500,      0.0171815678477287292480469,
        #      -0.0807484686374664306640625,     -0.1746668368577957153320312,
        #      -0.2693960368633270263671875,      0.2289195656776428222656250,
        #       0.0205991920083761215209961,      0.2002887874841690063476562,
        #      -0.1651854068040847778320312,      0.0043023726902902126312256,
        #      -0.1865801662206649780273438,     -0.0257664173841476440429688,
        #       0.1511912792921066284179688,     -0.0583690106868743896484375,
        #      -0.0041921143420040607452393,      0.0398448705673217773437500,
        #       0.0262939520180225372314453,      0.0079895406961441040039062,
        #      -0.3080827593803405761718750,      0.8411225080490112304687500,
        #      -0.0533705726265907287597656,     -0.0968873128294944763183594,
        #       0.0900650918483734130859375,      0.7493371963500976562500000,
        #       0.2962763905525207519531250,     -0.1152969747781753540039062,
        #      -0.2365301400423049926757812,     -0.0041137072257697582244873,
        #       0.1317992061376571655273438,     -0.0274031497538089752197266,
        #       0.3798457086086273193359375,     -0.0607191137969493865966797,
        #      -0.0946117788553237915039062,      0.0989270731806755065917969,
        #      -0.2933935523033142089843750,      0.0037478448357433080673218,
        #      -0.5423406362533569335937500,     -0.0178528167307376861572266,
        #       0.3831067085266113281250000,      0.0669692680239677429199219,
        #      -0.0321504436433315277099609,     -0.0004037543258164077997208,
        #       0.2033002674579620361328125,     -0.3419035077095031738281250,
        #       0.0352506227791309356689453,      0.0697044506669044494628906,
        #      -0.0534834526479244232177734,      2.1894419193267822265625000,
        #      -0.1784185171127319335937500,     -0.1694147586822509765625000,
        #      -0.2576711773872375488281250,      1.9750608205795288085937500,
        #       8.2167930603027343750000000,     -1.7196539640426635742187500,
        #      -0.8595688939094543457031250,      0.6012303829193115234375000,
        #      -0.6693711280822753906250000,     10.6692781448364257812500000,
        #       2.7617115974426269531250000,     -1.1509296894073486328125000,
        #      -0.1017333716154098510742188,      0.1937062740325927734375000,
        #       2.3534219264984130859375000,     -1.7043046951293945312500000,
        #      -3.3426346778869628906250000,      5.3313326835632324218750000,
        #       0.1319857537746429443359375,     -2.5329711437225341796875000,
        #      -0.8594818711280822753906250,      2.1946063041687011718750000,
        #      -0.5000000000000000000000000,     -0.5000000000000000000000000,
        #      -0.0129625983536243438720703,     -0.9514032602310180664062500,
        #      -1.0096907615661621093750000,      0.0509643293917179107666016,
        #       0.0334757417440414428710938,      0.2699498534202575683593750,
        #       0.9628744125366210937500000,      0.0032522231340408325195312,
        #      -0.0699721127748489379882812,     -0.1211879923939704895019531,
        #      -0.2466068267822265625000000,      0.2093174755573272705078125,
        #      -0.0168630313128232955932617,      0.1665729880332946777343750,
        #      -0.1620474010705947875976562,      0.0638734772801399230957031,
        #      -0.1728421747684478759765625,     -0.0160708129405975341796875,
        #       0.1203276216983795166015625,      0.0330459885299205780029297,
        #      -0.0209409054368734359741211,      0.0061676823534071445465088,
        #       0.0329704172909259796142578,     -0.0084927193820476531982422,
        #      -0.2629885673522949218750000,      0.8201450705528259277343750,
        #      -0.0364614538848400115966797,     -0.0873415097594261169433594,
        #       0.1192922890186309814453125,      0.7655465602874755859375000,
        #       0.2220560610294342041015625,      0.0895600914955139160156250,
        #       0.3297962844371795654296875,      0.0378880985081195831298828,
        #      -0.1553386151790618896484375,      0.1361845880746841430664062,
        #       0.1984127908945083618164062,      0.0326541773974895477294922,
        #       0.2226752489805221557617188,     -0.1172080039978027343750000,
        #       0.0910780355334281921386719,     -0.1248708516359329223632812,
        #       0.6240521669387817382812500,     -0.0672810226678848266601562,
        #       0.2141441851854324340820312,     -0.2872840166091918945312500,
        #       0.0939564332365989685058594,     -0.2863917052745819091796875,
        #      -0.0723479986190795898437500,      0.1951807141304016113281250,
        #       0.1092086210846900939941406,     -0.3272659480571746826171875,
        #      -0.0482664629817008972167969,     -0.8783699274063110351562500,
        #      -0.0699234604835510253906250,     -0.8546839356422424316406250,
        #      -0.6151859760284423828125000,      2.6317338943481445312500000,
        #       9.3338785171508789062500000,      2.1465270519256591796875000,
        #       0.3332407474517822265625000,     -0.2030525207519531250000000,
        #      -0.4275144338607788085937500,      8.6745929718017578125000000,
        #       4.4197826385498046875000000,      1.2554845809936523437500000,
        #      -0.0461214520037174224853516,     -0.3515377342700958251953125,
        #      -4.9116530418395996093750000,      3.8693120479583740234375000,
        #       0.8732876777648925781250000,      1.4854606389999389648437500,
        #      -0.2543546855449676513671875,      2.5073966979980468750000000,
        #       1.8092387914657592773437500,      3.7983105182647705078125000,
        #      -0.5000000000000000000000000,     -0.5000000000000000000000000,
        #      -0.1747466921806335449218750,     -0.8044985532760620117187500,
        #       0.7279235124588012695312500,      0.0373590663075447082519531,
        #      -0.0140729593113064765930176,      0.2868064045906066894531250,
        #       0.9579885601997375488281250,      0.0307899937033653259277344,
        #      -0.0747657045722007751464844,     -0.1406897902488708496093750,
        #      -0.2118421047925949096679688,      0.1987788230180740356445312,
        #       0.0227221306413412094116211,      0.1999859362840652465820312,
        #      -0.1571142375469207763671875,      0.0580728761851787567138672,
        #      -0.1367218792438507080078125,     -0.0030544996261596679687500,
        #       0.1450724154710769653320312,     -0.0096735777333378791809082,
        #      -0.0070722913369536399841309,      0.0666960179805755615234375,
        #       0.0054444367997348308563232,     -0.0240318961441516876220703,
        #      -0.2747312486171722412109375,      0.8958238959312438964843750,
        #      -0.0956844463944435119628906,     -0.0716991350054740905761719,
        #       0.0983934924006462097167969,      0.7819210886955261230468750,
        #       0.1142815276980400085449219,     -0.0269750189036130905151367,
        #      -0.0903980657458305358886719,      0.0482798404991626739501953,
        #       0.0678482800722122192382812,      0.1561535596847534179687500,
        #       0.1337469667196273803710938,      0.0228492394089698791503906,
        #      -0.0754430443048477172851562,      0.1862028688192367553710938,
        #      -0.1000331044197082519531250,     -0.1632643789052963256835938,
        #      -0.3875363767147064208984375,      0.0553510300815105438232422,
        #       0.2319633513689041137695312,     -0.0397523790597915649414062,
        #       0.0537221543490886688232422,     -0.4101552665233612060546875,
        #       0.3174995779991149902343750,     -0.1875082552433013916015625,
        #       0.1343509107828140258789062,     -0.3723058402538299560546875,
        #      -0.1762131601572036743164062,      1.9927299022674560546875000,
        #       0.0221033953130245208740234,     -0.3189619779586791992187500,
        #      -0.4710716605186462402343750,      0.5210730433464050292968750,
        #       9.6689805984497070312500000,     -2.0927090644836425781250000,
        #      -0.8240192532539367675781250,      0.4500705599784851074218750,
        #      -0.5016323328018188476562500,      8.0164728164672851562500000,
        #       5.4716262817382812500000000,     -1.5268659591674804687500000,
        #      -0.0986293703317642211914062,      0.4733132421970367431640625,
        #       2.2169780731201171875000000,     -2.2754330635070800781250000,
        #      -2.5120196342468261718750000,      5.3406028747558593750000000,
        #      -0.5225818753242492675781250,     -4.4111747741699218750000000,
        #      -0.7143121361732482910156250,      2.6998765468597412109375000,
        #      -0.5000000000000000000000000,     -0.5000000000000000000000000,
        #      -0.1913302540779113769531250,     -0.8408308625221252441406250,
        #      -0.6602052450180053710937500,      0.0235968679189682006835938,
        #      -0.0165131948888301849365234,      0.2398319542407989501953125,
        #       0.9708144068717956542968750,      0.0512453205883502960205078,
        #      -0.0480788461863994598388672,     -0.0980399772524833679199219,
        #      -0.2414377182722091674804688,      0.2084671109914779663085938,
        #      -0.0059439861215651035308838,      0.1399032473564147949218750,
        #      -0.1338257789611816406250000,      0.1151348426938056945800781,
        #      -0.0499564409255981445312500,      0.0122395157814025878906250,
        #       0.1361065506935119628906250,      0.0845649838447570800781250,
        #      -0.0060871099121868610382080,      0.0650789514183998107910156,
        #      -0.0243624374270439147949219,     -0.0196295231580734252929688,
        #      -0.2219436019659042358398438,      0.8803597092628479003906250,
        #      -0.0927120223641395568847656,     -0.0784796774387359619140625,
        #       0.1620670109987258911132812,      0.8197462558746337890625000,
        #       0.2477714568376541137695312,      0.1146687418222427368164062,
        #       0.2686304450035095214843750,     -0.0932427644729614257812500,
        #      -0.0776933431625366210937500,      0.1861366033554077148437500,
        #       0.0869435295462608337402344,      0.0802802667021751403808594,
        #       0.1571294814348220825195312,      0.0970004722476005554199219,
        #       0.0731003209948539733886719,     -0.1665363758802413940429688,
        #       0.5073723196983337402343750,     -0.0040777227841317653656006,
        #       0.2053097039461135864257812,     -0.2360057681798934936523438,
        #       0.0090450914576649665832520,      0.2413762807846069335937500,
        #      -0.2018246948719024658203125,      0.1231352463364601135253906,
        #       0.0286068599671125411987305,     -0.0721517652273178100585938,
        #       0.0794389769434928894042969,     -0.7143577337265014648437500,
        #       0.2777816355228424072265625,     -1.0058977603912353515625000,
        #      -1.0965150594711303710937500,      1.2273244857788085937500000,
        #      10.2214031219482421875000000,      2.4022128582000732421875000,
        #       0.3414421379566192626953125,     -0.1612302064895629882812500,
        #       0.5009864568710327148437500,      1.2025182247161865234375000,
        #       7.2591443061828613281250000,      1.4989216327667236328125000,
        #       0.0556172989308834075927734,     -0.0670892447233200073242188,
        #      -4.9882450103759765625000000,      4.5325183868408203125000000,
        #       0.6270859241485595703125000,      1.4367442131042480468750000,
        #      -0.2231781184673309326171875,      2.3022558689117431640625000,
        #       1.9977220296859741210937500,      4.0985975265502929687500000,
        #      -0.5000000000000000000000000,     -0.5000000000000000000000000,
        #       0.0923269987106323242187500,     -0.0800469964742660522460938,
        #      -0.0453049987554550170898438,     -0.1206599995493888854980469,
        #       0.0684529989957809448242188,      0.0335359983146190643310547,
        #      -0.0422290004789829254150391,     -0.0611529983580112457275391,
        #       0.0472120009362697601318359,      0.0050201001577079296112061,
        #       0.0436130017042160034179688,     -0.0668490007519721984863281,
        #       0.1384200006723403930664062,     -0.3518199920654296875000000,
        #       1.2539000511169433593750000,     -0.0792360007762908935546875,
        #      -0.1173299998044967651367188,      0.2466699928045272827148438,
        #       1.1676000356674194335937500,      0.0075185000896453857421875,
        #      -0.0025311000645160675048828,      0.0015536999562755227088928,
        #       0.0002659400051925331354141,      0.0251250006258487701416016,
        #      -0.0007114899926818907260895,      0.0049025001935660839080811,
        #      -0.0250569991767406463623047,      0.7273899912834167480468750,
        #      -0.0064722998067736625671387,      0.0646810010075569152832031,
        #      -0.0814569965004920959472656,     -0.0162310004234313964843750,
        #       0.0982789993286132812500000,     -0.4165300130844116210937500,
        #      -0.0214230008423328399658203,      0.0750100016593933105468750,
        #      -0.7155900001525878906250000,      0.0034316999372094869613647,
        #      -0.0638199970126152038574219,     -0.0838169977068901062011719,
        #      -0.0237830001860857009887695,     -0.1030900031328201293945312,
        #      -0.4197100102901458740234375,     -0.0254629999399185180664062,
        #      -0.0936120003461837768554688,     -0.7195600271224975585937500,
        #       0.0080917002633213996887207,      0.0990569964051246643066406,
        #       0.3101499974727630615234375,      0.0241279993206262588500977,
        #       0.1640399992465972900390625,      0.1252799928188323974609375,
        #       0.0784410014748573303222656,      0.1820700019598007202148438,
        #      -0.0931219980120658874511719,      0.0180799998342990875244141,
        #      -0.1011200025677680969238281,      0.3106699883937835693359375,
        #       0.0428999997675418853759766,     -0.1591099947690963745117188,
        #       0.1256300061941146850585938,      0.1162000000476837158203125,
        #      -0.1657699942588806152343750,     -0.0878089964389801025390625,
        #       0.0000000000000000000000000,      0.0000000000000000000000000,
        #       0.0000000000000000000000000,      0.0000000000000000000000000,
        #       0.0000000000000000000000000,      0.0000000000000000000000000,
        #       0.0000000000000000000000000,      0.9746999740600585937500000,
        #       0.0687989965081214904785156,     -0.1072999984025955200195312,
        #       0.0870409980416297912597656,     -0.0627129971981048583984375,
        #       0.1622599959373474121093750,      0.0584289990365505218505859,
        #      -0.1669600009918212890625000,      0.0028117001056671142578125,
        #       0.0276200007647275924682617,      0.1598500013351440429687500,
        #       0.1257899999618530273437500,     -0.1389999985694885253906250,
        #      -0.1381099969148635864257812,      0.1247199997305870056152344,
        #      -0.0357739999890327453613281,     -0.1431799978017807006835938,
        #       0.1186600029468536376953125,     -0.1177799999713897705078125,
        #      -0.0393100008368492126464844,      0.0179750006645917892456055,
        #      -0.1365199983119964599609375,      0.1777199953794479370117188,
        #       0.1946199983358383178710938,      0.0445460006594657897949219,
        #      -0.0062652002088725566864014,     -0.0755119994282722473144531,
        #      -0.1896499991416931152343750,     -0.1485899984836578369140625,
        #       0.1663299947977066040039062,      0.0075209001079201698303223,
        #       0.0407450012862682342529297,     -0.0308079998940229415893555,
        #       0.0639699995517730712890625,     -0.1489100009202957153320312,
        #      -0.1096400022506713867187500,     -0.1634200066328048706054688,
        #      -0.0230969991534948348999023,      0.0418360009789466857910156,
        #       0.0343930013477802276611328,     -0.1054800003767013549804688,
        #      -0.0238530002534389495849609,      0.0889469981193542480468750,
        #       0.0864899978041648864746094,     -0.0391950011253356933593750,
        #       0.0890529975295066833496094,     -0.1510899960994720458984375,
        #      -0.4145100712776184082031250,      1.6425266265869140625000000,
        #      -0.3444730639457702636718750,      0.1240477561950683593750000,
        #       0.0397267900407314300537109,     -0.0057800686918199062347412,
        #       0.9999833106994628906250000,     -0.0436135455965995788574219,
        #      -0.0554779097437858581542969,     -0.0325340591371059417724609,
        #       0.1190583705902099609375000,     -0.0164064168930053710937500,
        #      -0.1742343455553054809570312,      0.1330738663673400878906250,
        #      -0.1671777218580245971679688,      0.0581821314990520477294922,
        #      -0.2754791676998138427734375,      0.1009088829159736633300781,
        #       0.0453933142125606536865234,      0.0680625066161155700683594,
        #      -0.0913169905543327331542969,      0.0684501156210899353027344,
        #       0.0214102342724800109863281,      0.1665427535772323608398438,
        #      -0.2879897356033325195312500,      0.8123525381088256835937500,
        #      -0.0077113187871873378753662,     -0.0826900079846382141113281,
        #       0.1622375249862670898437500,      0.7685677409172058105468750,
        #      -0.5921098589897155761718750,     -0.0124142514541745185852051,
        #      -0.4520320892333984375000000,      0.5384637713432312011718750,
        #      -1.7476371526718139648437500,     -1.9748235940933227539062500,
        #      -0.4730711579322814941406250,      0.1149609833955764770507812,
        #       0.1628255993127822875976562,      0.1213796138763427734375000,
        #      -0.0077348710037767887115479,      0.1388164013624191284179688,
        #       0.4018138051033020019531250,      0.0901145488023757934570312,
        #      -0.4096522331237792968750000,     -0.1252523511648178100585938,
        #      -0.1002392694354057312011719,      0.1647296249866485595703125,
        #      -0.1426574140787124633789062,      0.2049462348222732543945312,
        #      -0.1324206292629241943359375,      0.2965873181819915771484375,
        #       0.1558795422315597534179688,     -1.8552563190460205078125000,
        #      -0.2260617911815643310546875,     -0.1487042456865310668945312,
        #       0.2340030074119567871093750,     -0.9695094227790832519531250,
        #       1.4943186044692993164062500,      2.0957019329071044921875000,
        #      -0.2594992220401763916015625,     -0.2117990851402282714843750,
        #      -0.7070016264915466308593750,      3.4302277565002441406250000,
        #      -1.4394794702529907226562500,      1.6348440647125244140625000,
        #      -0.4247028529644012451171875,      0.0957648530602455139160156,
        #      -4.3269057273864746093750000,      4.1703610420227050781250000,
        #       0.4823386669158935546875000,      1.5430774688720703125000000,
        #      -0.3829332292079925537109375,      2.5261154174804687500000000,
        #       1.8419561386108398437500000,      3.7317197322845458984375000,
        #      -0.5000000000000000000000000,     -0.5000000000000000000000000,
        #      -0.1752460896968841552734375,      1.3589235544204711914062500,
        #       2.1047379970550537109375000,      0.1266568750143051147460938,
        #      -0.0165557377040386199951172,      0.0608149766921997070312500,
        #       0.9981490373611450195312500,     -0.0191041156649589538574219,
        #      -0.0619694255292415618896484,     -0.0453415289521217346191406,
        #       0.0747818648815155029296875,      0.1035551652312278747558594,
        #      -0.1443162709474563598632812,      0.2046342492103576660156250,
        #      -0.1836392432451248168945312,      0.0177587140351533889770508,
        #      -0.2695671319961547851562500,      0.0863257125020027160644531,
        #       0.0573154352605342864990234,     -0.0274401865899562835693359,
        #      -0.0917736664414405822753906,      0.1292354017496109008789062,
        #       0.0047935149632394313812256,      0.1508226245641708374023438,
        #      -0.3149097263813018798828125,      0.8629958033561706542968750,
        #      -0.0462034121155738830566406,     -0.0773230642080307006835938,
        #       0.1410575658082962036132812,      0.7483119964599609375000000,
        #      -0.5420632362365722656250000,     -0.0417143516242504119873047,
        #      -0.3058127164840698242187500,      0.4248641133308410644531250,
        #      -1.5553895235061645507812500,     -2.0683226585388183593750000,
        #      -0.2322608232498168945312500,      0.0044713579118251800537109,
        #      -0.3104729950428009033203125,      0.1850038617849349975585938,
        #      -0.0934719964861869812011719,     -0.3127168118953704833984375,
        #      -0.8252935409545898437500000,      0.0121619943529367446899414,
        #      -0.4494777619838714599609375,      0.3897711336612701416015625,
        #      -0.1420427858829498291015625,      0.2538599371910095214843750,
        #       0.0935436412692070007324219,     -0.1581166535615921020507812,
        #      -0.0714682936668395996093750,      0.3116651177406311035156250,
        #       0.0503962524235248565673828,      0.6747753620147705078125000,
        #      -0.7876816987991333007812500,      0.1649519354104995727539062,
        #      -0.1494762301445007324218750,     -0.4830598235130310058593750,
        #       3.1281874179840087890625000,      0.8497322201728820800781250,
        #      -1.7114437818527221679687500,      0.8329994678497314453125000,
        #      -0.5459045171737670898437500,      4.7823796272277832031250000,
        #      -0.6671900153160095214843750,     -0.9571863412857055664062500,
        #      -0.4140693843364715576171875,      0.9988133907318115234375000,
        #       2.1504323482513427734375000,     -1.6568133831024169921875000,
        #      -3.1251819133758544921875000,      4.9853253364562988281250000,
        #      -0.0433708131313323974609375,     -2.5882654190063476562500000,
        #      -0.8807485103607177734375000,      2.1960976123809814453125000,
        #      -0.5000000000000000000000000,     -0.5000000000000000000000000,
        #      -0.3976787030696868896484375,      1.6384958028793334960937500,
        #       0.0666980296373367309570312,      0.1194717064499855041503906,
        #      -0.0006658882484771311283112,      0.0760609135031700134277344,
        #       0.9971031546592712402343750,     -0.0423407331109046936035156,
        #      -0.0605522282421588897705078,     -0.0563874021172523498535156,
        #       0.0363943278789520263671875,      0.1077053621411323547363281,
        #       0.2591170668601989746093750,      0.1673026531934738159179688,
        #      -0.1794095933437347412109375,      0.0181505456566810607910156,
        #      -0.2338360548019409179687500,      0.0674748867750167846679688,
        #       0.0539624020457267761230469,      0.0378398336470127105712891,
        #      -0.0808264315128326416015625,      0.1291754394769668579101562,
        #      -0.0210568513721227645874023,      0.1308598816394805908203125,
        #      -0.3012409508228302001953125,      0.8143658041954040527343750,
        #      -0.0241110920906066894531250,     -0.0906001180410385131835938,
        #       0.1553230285644531250000000,      0.7650400400161743164062500,
        #      -0.4733823239803314208984375,     -0.1013112068176269531250000,
        #      -0.4790562689304351806640625,      0.4073675572872161865234375,
        #      -1.8492794036865234375000000,      2.0265996456146240234375000,
        #      -0.4286121428012847900390625,      0.1052282080054283142089844,
        #       0.1362633854150772094726562,      0.0889880359172821044921875,
        #       0.0725718364119529724121094,     -0.2643859088420867919921875,
        #       0.4212074875831604003906250,      0.1279764026403427124023438,
        #      -0.3221175074577331542968750,     -0.1496338844299316406250000,
        #      -0.1297519654035568237304688,      0.1726567745208740234375000,
        #      -0.1129259690642356872558594,      0.2047509253025054931640625,
        #      -0.1483640819787979125976562,      0.2740651965141296386718750,
        #       0.1782186478376388549804688,     -1.7319300174713134765625000,
        #      -0.6990405321121215820312500,     -0.2091941237449645996093750,
        #      -0.2569817900657653808593750,     -0.9513377547264099121093750,
        #       3.5002982616424560546875000,      2.0701079368591308593750000,
        #       0.0522022135555744171142578,     -0.3951648473739624023437500,
        #      -0.4624654054641723632812500,      4.4524035453796386718750000,
        #       0.1650938540697097778320312,      1.4470739364624023437500000,
        #      -0.2999167740345001220703125,      0.2275261878967285156250000,
        #      -3.8642773628234863281250000,      4.0433268547058105468750000,
        #       0.5411444902420043945312500,      1.5910434722900390625000000,
        #      -0.0747008025646209716796875,      2.3147728443145751953125000,
        #       1.9125323295593261718750000,      3.8109910488128662109375000,
        #      -0.5000000000000000000000000,     -0.5000000000000000000000000,
        #      -0.3390934765338897705078125,      0.6180462837219238281250000,
        #       2.3863730430603027343750000,      0.0975738018751144409179688,
        #      -0.0808101892471313476562500,      0.1499994993209838867187500,
        #       0.9886860847473144531250000,      0.0572417639195919036865234,
        #      -0.0689389258623123168945312,     -0.0460134036839008331298828,
        #      -0.0515509098768234252929688,     -0.0037300586700439453125000,
        #       0.2638660073280334472656250,      0.2736884951591491699218750,
        #      -0.1730543076992034912109375,      0.0338823981583118438720703,
        #      -0.2405104339122772216796875,      0.0706704407930374145507812,
        #       0.1050943210721015930175781,     -0.0283221080899238586425781,
        #      -0.0467020682990550994873047,      0.2604228258132934570312500,
        #      -0.0494534820318222045898438,      0.0959125086665153503417969,
        #      -0.3085030615329742431640625,      0.8331608176231384277343750,
        #      -0.0891724228858947753906250,     -0.1156353577971458435058594,
        #       0.1436382681131362915039062,      0.7355542182922363281250000,
        #      -0.2402075380086898803710938,     -0.1625866740942001342773438,
        #      -0.4595081508159637451171875,      0.2792014777660369873046875,
        #      -1.8860992193222045898437500,     -0.0691921487450599670410156,
        #       0.0172837059944868087768555,      0.0504081510007381439208984,
        #      -0.0498866923153400421142578,      0.0546172969043254852294922,
        #      -0.0433956719934940338134766,     -0.0628471374511718750000000,
        #      -0.7530230879783630371093750,      0.0737403631210327148437500,
        #      -0.1361385136842727661132812,      0.3375205099582672119140625,
        #      -0.2431208342313766479492188,      0.4809713363647460937500000,
        #      -0.0088894143700599670410156,     -0.3004139065742492675781250,
        #      -0.1669828295707702636718750,      0.3482741415500640869140625,
        #       0.0836070030927658081054688,      0.9782341122627258300781250,
        #      -1.2189347743988037109375000,      0.4269589185714721679687500,
        #      -1.0248006582260131835937500,      1.4757235050201416015625000,
        #       8.8456420898437500000000000,      1.7355985641479492187500000,
        #      -1.0482479333877563476562500,      0.8797641992568969726562500,
        #      -0.3615604639053344726562500,      7.9279980659484863281250000,
        #       0.7654345631599426269531250,     -0.5851240158081054687500000,
        #      -0.1588701605796813964843750,      0.9261130094528198242187500,
        #       1.4733132123947143554687500,     -1.7129113674163818359375000,
        #      -2.9810442924499511718750000,      4.5173115730285644531250000,
        #      -1.3502311706542968750000000,     -3.0644223690032958984375000,
        #      -0.7707746624946594238281250,      1.8044929504394531250000000,
        #      -0.5000000000000000000000000,     -0.5000000000000000000000000,
        #      -0.3167025148868560791015625,      1.5890419483184814453125000,
        #       0.5339698195457458496093750,      0.0745358094573020935058594,
        #       0.0057854340411722660064697,      0.1811263710260391235351562,
        #       0.9834598302841186523437500,      0.0171577483415603637695312,
        #      -0.0560315847396850585937500,     -0.0502377636730670928955078,
        #      -0.2025758922100067138671875,      0.2384016215801239013671875,
        #       0.2252940833568572998046875,      0.1846370398998260498046875,
        #      -0.1551094502210617065429688,      0.0395718403160572052001953,
        #      -0.2318079471588134765625000,      0.0565386116504669189453125,
        #       0.0931907296180725097656250,      0.0465559028089046478271484,
        #      -0.0066724941134452819824219,      0.1417512446641921997070312,
        #      -0.0374434925615787506103516,      0.0743566378951072692871094,
        #      -0.3680553436279296875000000,      0.8023047447204589843750000,
        #      -0.0239731390029191970825195,     -0.1329519748687744140625000,
        #       0.1251030862331390380859375,      0.7501159906387329101562500,
        #      -0.3042968809604644775390625,      0.0798159241676330566406250,
        #       0.0228958725929260253906250,     -0.0520045720040798187255859,
        #      -0.1307316571474075317382812,     -0.0707168057560920715332031,
        #      -0.2682897746562957763671875,      0.0594860687851905822753906,
        #      -0.1468626111745834350585938,     -0.0620919428765773773193359,
        #      -0.0698484405875205993652344,     -0.1734162867069244384765625,
        #       0.2973378002643585205078125,      0.0691553652286529541015625,
        #      -0.4411934912204742431640625,     -0.0047919596545398235321045,
        #      -0.0131094520911574363708496,     -0.0823539271950721740722656,
        #      -0.1894162446260452270507812,      0.2334754019975662231445312,
        #      -0.0912185385823249816894531,      0.2530194222927093505859375,
        #       0.2191126793622970581054688,     -0.4068344235420227050781250,
        #      -0.9544271230697631835937500,     -0.3555543720722198486328125,
        #      -1.0945864915847778320312500,      4.5633244514465332031250000,
        #      15.6672182083129882812500000,      1.9581295251846313476562500,
        #       0.5572302341461181640625000,     -0.2093592584133148193359375,
        #      -0.5318229794502258300781250,      8.1149435043334960937500000,
        #       3.3069796562194824218750000,      0.9281886219978332519531250,
        #      -0.0500464178621768951416016,      0.2024014890193939208984375,
        #      -2.7319643497467041015625000,      3.0214436054229736328125000,
        #       0.1318908035755157470703125,      1.8186612129211425781250000,
        #       0.9091939926147460937500000,      1.9806658029556274414062500,
        #       1.4381653070449829101562500,      3.7242844104766845703125000,
        #      -0.5000000000000000000000000,     -0.5000000000000000000000000,
        #      -0.1988602429628372192382812,      0.3061016201972961425781250,
        #       0.8286576271057128906250000,      0.0563971661031246185302734,
        #       0.0036339988000690937042236,      0.2526302039623260498046875,
        #       0.9675629138946533203125000,      0.0493306182324886322021484,
        #      -0.0615503787994384765625000,     -0.1029909849166870117187500,
        #      -0.2766937911510467529296875,      0.2162504941225051879882812,
        #       0.2301088571548461914062500,      0.2058112174272537231445312,
        #      -0.1488909572362899780273438,      0.0212053973227739334106445,
        #      -0.2157928347587585449218750,      0.0251841545104980468750000,
        #       0.1046526208519935607910156,     -0.0360856503248214721679688,
        #       0.0123798847198486328125000,      0.1193235516548156738281250,
        #      -0.0123246721923351287841797,      0.0473326109349727630615234,
        #      -0.3786850571632385253906250,      0.8433661460876464843750000,
        #      -0.0642810314893722534179688,     -0.1304915547370910644531250,
        #       0.1121144965291023254394531,      0.7673029303550720214843750,
        #       0.0478074476122856140136719,     -0.0733311399817466735839844,
        #      -0.2737498581409454345703125,     -0.2240227907896041870117188,
        #      -0.0391589663922786712646484,      0.5519010424613952636718750,
        #      -0.0505842268466949462890625,      0.0228309854865074157714844,
        #       0.0335964187979698181152344,      0.1630580872297286987304688,
        #      -0.1906354129314422607421875,     -0.1356258392333984375000000,
        #      -0.3941102623939514160156250,      0.0462151318788528442382812,
        #      -0.1661041378974914550781250,      0.1926800012588500976562500,
        #      -0.0817400589585304260253906,     -0.0951998159289360046386719,
        #       0.2122731506824493408203125,     -0.1856403201818466186523438,
        #      -0.0811969041824340820312500,      0.2193419784307479858398438,
        #      -0.0177355129271745681762695,      2.3756825923919677734375000,
        #      -0.6513727903366088867187500,      0.1852205246686935424804688,
        #      -0.4629589915275573730468750,      4.0587210655212402343750000,
        #      11.2473840713500976562500000,      0.2620953619480133056640625,
        #      -0.4013788402080535888671875,      0.5723547339439392089843750,
        #      -0.4060186147689819335937500,     12.3609895706176757812500000,
        #       2.3412671089172363281250000,     -0.8805527687072753906250000,
        #      -0.0443686135113239288330078,      0.0509020611643791198730469,
        #       1.5631890296936035156250000,     -1.5174994468688964843750000,
        #      -2.7474646568298339843750000,      4.4883098602294921875000000,
        #      -1.0984699726104736328125000,     -2.6847479343414306640625000,
        #      -0.5729723572731018066406250,      2.2995111942291259765625000,
        #      -0.5000000000000000000000000,     -0.5000000000000000000000000,
        #      -0.0557903982698917388916016,     -1.1220515966415405273437500,
        #       0.4402205049991607666015625,      0.0510412007570266723632812,
        #       0.0310374330729246139526367,      0.2680881321430206298828125,
        #       0.9633944034576416015625000,      0.0325090736150741577148438,
        #      -0.0521629936993122100830078,     -0.1275454908609390258789062,
        #      -0.3071124851703643798828125,      0.2520418167114257812500000,
        #       0.0107025634497404098510742,      0.1901922225952148437500000,
        #      -0.1505199372768402099609375,      0.0044078803621232509613037,
        #      -0.2256443053483963012695312,      0.0111294835805892944335938,
        #       0.1048782989382743835449219,      0.0161452889442443847656250,
        #       0.0026774692814797163009644,      0.0506901182234287261962891,
        #       0.0189582332968711853027344,      0.0012786077568307518959045,
        #      -0.2442948818206787109375000,      0.7892524600028991699218750,
        #      -0.0368890464305877685546875,     -0.1292335987091064453125000,
        #       0.1403569877147674560546875,      0.7906432747840881347656250,
        #       0.2259670495986938476562500,      0.0170157384127378463745117,
        #      -0.0632177889347076416015625,     -0.0060159387066960334777832,
        #       0.0984176769852638244628906,     -0.4452678859233856201171875,
        #       0.3791607022285461425781250,      0.0036483779549598693847656,
        #      -0.1179847493767738342285156,     -0.1793670654296875000000000,
        #       0.0173094160854816436767578,     -0.0563375651836395263671875,
        #       0.1394380331039428710937500,     -0.0661912336945533752441406,
        #       0.2531404197216033935546875,     -0.0358539037406444549560547,
        #      -0.1195269227027893066406250,      0.6520054936408996582031250,
        #      -0.3662268817424774169921875,     -0.0776600390672683715820312,
        #      -0.0406240373849868774414062,      0.3502987325191497802734375,
        #       0.2701234519481658935546875,     -0.4650021791458129882812500,
        #      -0.0240667499601840972900391,     -0.8515438437461853027343750,
        #      -0.6589136719703674316406250,      4.4570612907409667968750000,
        #       9.7113609313964843750000000,      1.9048637151718139648437500,
        #       0.3162832558155059814453125,     -0.2193512320518493652343750,
        #      -0.5879993438720703125000000,      9.4803142547607421875000000,
        #       3.8904640674591064453125000,      0.6696988344192504882812500,
        #       0.0030622519552707672119141,     -0.2225416004657745361328125,
        #      -2.9281847476959228515625000,      2.9904303550720214843750000,
        #       0.9965088367462158203125000,      1.6217541694641113281250000,
        #       0.7642003297805786132812500,      2.1409063339233398437500000,
        #       1.3981121778488159179687500,      3.7922210693359375000000000,
        #      -0.5000000000000000000000000,     -0.5000000000000000000000000,
        #       0.0726586356759071350097656,     -1.5362514257431030273437500,
        #       1.0208513736724853515625000,      0.0532127767801284790039062,
        #       0.0218555331230163574218750,      0.3198881745338439941406250,
        #       0.9474552869796752929687500,      0.0171815678477287292480469,
        #      -0.0807484686374664306640625,     -0.1746668368577957153320312,
        #      -0.2693960368633270263671875,      0.2289195656776428222656250,
        #       0.0205991920083761215209961,      0.2002887874841690063476562,
        #      -0.1651854068040847778320312,      0.0043023726902902126312256,
        #      -0.1865801662206649780273438,     -0.0257664173841476440429688,
        #       0.1511912792921066284179688,     -0.0583690106868743896484375,
        #      -0.0041921143420040607452393,      0.0398448705673217773437500,
        #       0.0262939520180225372314453,      0.0079895406961441040039062,
        #      -0.3080827593803405761718750,      0.8411225080490112304687500,
        #      -0.0533705726265907287597656,     -0.0968873128294944763183594,
        #       0.0900650918483734130859375,      0.7493371963500976562500000,
        #       0.2962763905525207519531250,     -0.1152969747781753540039062,
        #      -0.2365301400423049926757812,     -0.0041137072257697582244873,
        #       0.1317992061376571655273438,     -0.0274031497538089752197266,
        #       0.3798457086086273193359375,     -0.0607191137969493865966797,
        #      -0.0946117788553237915039062,      0.0989270731806755065917969,
        #      -0.2933935523033142089843750,      0.0037478448357433080673218,
        #      -0.5423406362533569335937500,     -0.0178528167307376861572266,
        #       0.3831067085266113281250000,      0.0669692680239677429199219,
        #      -0.0321504436433315277099609,     -0.0004037543258164077997208,
        #       0.2033002674579620361328125,     -0.3419035077095031738281250,
        #       0.0352506227791309356689453,      0.0697044506669044494628906,
        #      -0.0534834526479244232177734,      2.1894419193267822265625000,
        #      -0.1784185171127319335937500,     -0.1694147586822509765625000,
        #      -0.2576711773872375488281250,      1.9750608205795288085937500,
        #       8.2167930603027343750000000,     -1.7196539640426635742187500,
        #      -0.8595688939094543457031250,      0.6012303829193115234375000,
        #      -0.6693711280822753906250000,     10.6692781448364257812500000,
        #       2.7617115974426269531250000,     -1.1509296894073486328125000,
        #      -0.1017333716154098510742188,      0.1937062740325927734375000,
        #       2.3534219264984130859375000,     -1.7043046951293945312500000,
        #      -3.3426346778869628906250000,      5.3313326835632324218750000,
        #       0.1319857537746429443359375,     -2.5329711437225341796875000,
        #      -0.8594818711280822753906250,      2.1946063041687011718750000,
        #      -0.5000000000000000000000000,     -0.5000000000000000000000000,
        #      -0.0129625983536243438720703,     -0.9514032602310180664062500,
        #      -1.0096907615661621093750000,      0.0509643293917179107666016,
        #       0.0334757417440414428710938,      0.2699498534202575683593750,
        #       0.9628744125366210937500000,      0.0032522231340408325195312,
        #      -0.0699721127748489379882812,     -0.1211879923939704895019531,
        #      -0.2466068267822265625000000,      0.2093174755573272705078125,
        #      -0.0168630313128232955932617,      0.1665729880332946777343750,
        #      -0.1620474010705947875976562,      0.0638734772801399230957031,
        #      -0.1728421747684478759765625,     -0.0160708129405975341796875,
        #       0.1203276216983795166015625,      0.0330459885299205780029297,
        #      -0.0209409054368734359741211,      0.0061676823534071445465088,
        #       0.0329704172909259796142578,     -0.0084927193820476531982422,
        #      -0.2629885673522949218750000,      0.8201450705528259277343750,
        #      -0.0364614538848400115966797,     -0.0873415097594261169433594,
        #       0.1192922890186309814453125,      0.7655465602874755859375000,
        #       0.2220560610294342041015625,      0.0895600914955139160156250,
        #       0.3297962844371795654296875,      0.0378880985081195831298828,
        #      -0.1553386151790618896484375,      0.1361845880746841430664062,
        #       0.1984127908945083618164062,      0.0326541773974895477294922,
        #       0.2226752489805221557617188,     -0.1172080039978027343750000,
        #       0.0910780355334281921386719,     -0.1248708516359329223632812,
        #       0.6240521669387817382812500,     -0.0672810226678848266601562,
        #       0.2141441851854324340820312,     -0.2872840166091918945312500,
        #       0.0939564332365989685058594,     -0.2863917052745819091796875,
        #      -0.0723479986190795898437500,      0.1951807141304016113281250,
        #       0.1092086210846900939941406,     -0.3272659480571746826171875,
        #      -0.0482664629817008972167969,     -0.8783699274063110351562500,
        #      -0.0699234604835510253906250,     -0.8546839356422424316406250,
        #      -0.6151859760284423828125000,      2.6317338943481445312500000,
        #       9.3338785171508789062500000,      2.1465270519256591796875000,
        #       0.3332407474517822265625000,     -0.2030525207519531250000000,
        #      -0.4275144338607788085937500,      8.6745929718017578125000000,
        #       4.4197826385498046875000000,      1.2554845809936523437500000,
        #      -0.0461214520037174224853516,     -0.3515377342700958251953125,
        #      -4.9116530418395996093750000,      3.8693120479583740234375000,
        #       0.8732876777648925781250000,      1.4854606389999389648437500,
        #      -0.2543546855449676513671875,      2.5073966979980468750000000,
        #       1.8092387914657592773437500,      3.7983105182647705078125000,
        #      -0.5000000000000000000000000,     -0.5000000000000000000000000,
        #      -0.1747466921806335449218750,     -0.8044985532760620117187500,
        #       0.7279235124588012695312500,      0.0373590663075447082519531,
        #      -0.0140729593113064765930176,      0.2868064045906066894531250,
        #       0.9579885601997375488281250,      0.0307899937033653259277344,
        #      -0.0747657045722007751464844,     -0.1406897902488708496093750,
        #      -0.2118421047925949096679688,      0.1987788230180740356445312,
        #       0.0227221306413412094116211,      0.1999859362840652465820312,
        #      -0.1571142375469207763671875,      0.0580728761851787567138672,
        #      -0.1367218792438507080078125,     -0.0030544996261596679687500,
        #       0.1450724154710769653320312,     -0.0096735777333378791809082,
        #      -0.0070722913369536399841309,      0.0666960179805755615234375,
        #       0.0054444367997348308563232,     -0.0240318961441516876220703,
        #      -0.2747312486171722412109375,      0.8958238959312438964843750,
        #      -0.0956844463944435119628906,     -0.0716991350054740905761719,
        #       0.0983934924006462097167969,      0.7819210886955261230468750,
        #       0.1142815276980400085449219,     -0.0269750189036130905151367,
        #      -0.0903980657458305358886719,      0.0482798404991626739501953,
        #       0.0678482800722122192382812,      0.1561535596847534179687500,
        #       0.1337469667196273803710938,      0.0228492394089698791503906,
        #      -0.0754430443048477172851562,      0.1862028688192367553710938,
        #      -0.1000331044197082519531250,     -0.1632643789052963256835938,
        #      -0.3875363767147064208984375,      0.0553510300815105438232422,
        #       0.2319633513689041137695312,     -0.0397523790597915649414062,
        #       0.0537221543490886688232422,     -0.4101552665233612060546875,
        #       0.3174995779991149902343750,     -0.1875082552433013916015625,
        #       0.1343509107828140258789062,     -0.3723058402538299560546875,
        #      -0.1762131601572036743164062,      1.9927299022674560546875000,
        #       0.0221033953130245208740234,     -0.3189619779586791992187500,
        #      -0.4710716605186462402343750,      0.5210730433464050292968750,
        #       9.6689805984497070312500000,     -2.0927090644836425781250000,
        #      -0.8240192532539367675781250,      0.4500705599784851074218750,
        #      -0.5016323328018188476562500,      8.0164728164672851562500000,
        #       5.4716262817382812500000000,     -1.5268659591674804687500000,
        #      -0.0986293703317642211914062,      0.4733132421970367431640625,
        #       2.2169780731201171875000000,     -2.2754330635070800781250000,
        #      -2.5120196342468261718750000,      5.3406028747558593750000000,
        #      -0.5225818753242492675781250,     -4.4111747741699218750000000,
        #      -0.7143121361732482910156250,      2.6998765468597412109375000,
        #      -0.5000000000000000000000000,     -0.5000000000000000000000000]]).view(1,-1).to(self.device)        

        
        # print("self.obs_buf")
        # torch.set_printoptions(precision=25, threshold=torch.iinfo(torch.int32).max, sci_mode=False)
        # print(self.obs_buf)
        # print("self.obs_buf")
        # print(self.obs_buf.dtype)

        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
            torch.cat([
                self.obs_history_buf[:, 1:],
                obs_buf.unsqueeze(1)
            ], dim=1)
        )

        # print("obs_history_buf ", self.obs_history_buf.shape, self.obs_history_buf)


        self.contact_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([self.contact_filt.float()] * self.cfg.env.contact_buf_len, dim=1),
            torch.cat([
                self.contact_buf[:, 1:],
                self.contact_filt.float().unsqueeze(1)
            ], dim=1)
        )

    def _motion_sync(self):
        num_motions = self._motion_lib.num_motions()
        motion_ids = self._motion_ids
        # print(self._motion_times[self.lookat_id])
        # motion_times = self.episode_length_buf * self._motion_dt

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
           = self._motion_lib.get_motion_state(motion_ids, self._motion_times)
        
        root_pos[:, :2] = (self._curr_demo_root_pos - self.init_root_pos_global_demo + self.init_root_pos_global)[:, :2]
        root_vel = torch.zeros_like(root_vel)
        root_ang_vel = torch.zeros_like(root_ang_vel)
        dof_vel = torch.zeros_like(dof_vel)

        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        dof_pos, dof_vel = self.reindex_dof_pos_vel(dof_pos, dof_vel)

        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return
    
    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self.root_states[env_ids, 0:3] = root_pos
        self.root_states[env_ids, 3:7] = root_rot
        self.root_states[env_ids, 7:10] = root_vel
        self.root_states[env_ids, 10:13] = root_ang_vel

        self.dof_pos[env_ids] = dof_pos
        self.dof_vel[env_ids] = dof_vel
        return

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        # roll_cutoff = torch.abs(self.roll) > 1.0
        # pitch_cutoff = torch.abs(self.pitch) > 1.0
        # height_cutoff = self.root_states[:, 2] < 0.5

        dof_dev = self._reward_tracking_demo_dof_pos() < 0.1
        self.reset_buf |= dof_dev



        z = self.root_states[:, 2]
        z_threshold_buff = z < 0.6
        self.reset_buf |= z_threshold_buff

        # demo_dofs = self._curr_demo_obs_buf[:, :self.num_dof]
        # ref_deviation = torch.norm(self.dof_pos - demo_dofs, dim=1) >= self.dof_term_threshold
        # self.reset_buf |= ref_deviation
        
        # height_dev = torch.abs(self.root_states[:, 2] - self._curr_demo_root_pos[:, 2]) >= self.height_term_threshold
        # self.reset_buf |= height_dev

        # yaw_dev = self._reward_tracking_demo_yaw() < self.yaw_term_threshold
        # self.reset_buf |= yaw_dev

        # ref_keybody_dev = self._reward_tracking_demo_key_body() < 0.2
        # self.reset_buf |= ref_keybody_dev

        # ref_deviation = (torch.norm(self.dof_pos - demo_dofs, dim=1) >= 1.5) & \
        #                 (self._motion_difficulty < 3)
        # self.reset_buf |= ref_deviation
        
        # ref_keybody_dev = (self._reward_tracking_demo_key_body() < 0.3) & \
        #                   (self._motion_difficulty < 3)
        # self.reset_buf |= ref_keybody_dev

        motion_end = self.episode_length_buf * self.dt >= self._motion_lengths
        self.reset_buf |= motion_end

        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.time_out_buf |= motion_end

        self.reset_buf |= self.time_out_buf
        # self.reset_buf |= roll_cutoff
        # self.reset_buf |= pitch_cutoff
        # self.reset_buf |= height_cutoff

    ######### demonstrations #########
    # def get_demo_obs(self, ):
    #     demo_motion_times = self._motion_demo_offsets + self._motion_times[:, None]  # [num_envs, demo_dim]
    #     # get the motion state at the demo times
    #     root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
    #         = self._motion_lib.get_motion_state(self._motion_ids.repeat(self._motion_num_future_steps), demo_motion_times.flatten())
    #     dof_pos, dof_vel = self.reindex_dof_pos_vel(dof_pos, dof_vel)
        
    #     demo_obs = build_demo_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_pos, self._dof_offsets)
    #     return demo_obs
    
    # def get_curr_demo(self):
    #     root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
    #         = self._motion_lib.get_motion_state(self._motion_ids, self._motion_times)
    #     dof_pos, dof_vel = self.reindex_dof_pos_vel(dof_pos, dof_vel)
    #     demo_obs = build_demo_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_pos, self._dof_offsets)
    #     return demo_obs
    
    
    ######### utils #########
    
    def reindex_dof_pos_vel(self, dof_pos, dof_vel):
        dof_pos = reindex_motion_dof(dof_pos, self.dof_indices_sim, self.dof_indices_motion, self._valid_dof_body_ids)
        dof_vel = reindex_motion_dof(dof_vel, self.dof_indices_sim, self.dof_indices_motion, self._valid_dof_body_ids)
        return dof_pos, dof_vel

    def draw_rigid_bodies_demo(self, ):
        geom = gymutil.WireframeSphereGeometry(0.06, 32, 32, None, color=(0, 1, 0))
        local_body_pos = self._curr_demo_keybody.clone().view(self.num_envs, self._num_key_bodies, 3)
        if self.cfg.motion.global_keybody:
            curr_demo_xyz = torch.cat((self.target_pos_abs, self._curr_demo_root_pos[:, 2:3]), dim=-1)
        else:
            curr_demo_xyz = torch.cat((self.root_states[:, :2], self._curr_demo_root_pos[:, 2:3]), dim=-1)
        global_body_pos = local_to_global(self._curr_demo_quat, local_body_pos, curr_demo_xyz)
        for i in range(global_body_pos.shape[1]):
            pose = gymapi.Transform(gymapi.Vec3(global_body_pos[self.lookat_id, i, 0], global_body_pos[self.lookat_id, i, 1], global_body_pos[self.lookat_id, i, 2]), r=None)
            gymutil.draw_lines(geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)

    def draw_rigid_bodies_actual(self, ):
        geom = gymutil.WireframeSphereGeometry(0.06, 32, 32, None, color=(1, 0, 0))
        rigid_body_pos = self.rigid_body_states[:, self._key_body_ids_sim, :3].clone()
        for i in range(rigid_body_pos.shape[1]):
            pose = gymapi.Transform(gymapi.Vec3(rigid_body_pos[self.lookat_id, i, 0], rigid_body_pos[self.lookat_id, i, 1], rigid_body_pos[self.lookat_id, i, 2]), r=None)
            gymutil.draw_lines(geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)

    def _draw_goals(self, ):
        demo_geom = gymutil.WireframeSphereGeometry(0.2, 32, 32, None, color=(1, 0, 0))
        
        pose_robot = self.root_states[self.lookat_id, :3].cpu().numpy()
        # print(self._curr_demo_obs_buf[self.lookat_id, 2*self.num_dof:2*self.num_dof+3])
        # demo_pos = (self._curr_demo_root_pos - self.init_root_pos_global_demo + self.init_root_pos_global)[self.lookat_id]
        # pose = gymapi.Transform(gymapi.Vec3(demo_pos[0], demo_pos[1], demo_pos[2]), r=None)
        # gymutil.draw_lines(demo_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        if not self.cfg.depth.use_camera:
            sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 0.35, 0.25))
            # norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
            # target_vec_norm = self.target_pos_rel / (norm + 1e-5)
            norm = torch.norm(self._curr_demo_root_vel[:, :2], dim=-1, keepdim=True)
            target_vec_norm = self._curr_demo_root_vel[:, :2] / (norm + 1e-5)
            for i in range(5):
                pose_arrow = pose_robot[:2] + 0.1*(i+3) * target_vec_norm[self.lookat_id, :2].cpu().numpy()
                pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None)
                gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)
    
    ######### Rewards #########
    def compute_reward(self):
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew #if "demo" not in name else 0  # log demo rew but do not include in additative reward
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        if self.cfg.rewards.clip_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=-0.5)
        
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
        
    def _reward_tracking_demo_goal_vel(self):
        norm = torch.norm(self._curr_demo_root_vel[:, :3], dim=-1, keepdim=True)
        target_vec_norm = self._curr_demo_root_vel[:, :3] / (norm + 1e-5)
        cur_vel = self.root_states[:, 7:10]
        norm_squeeze = norm.squeeze(-1)
        rew = torch.minimum(torch.sum(target_vec_norm * cur_vel, dim=-1), norm_squeeze) / (norm_squeeze + 1e-5)

        rew_zeros = torch.exp(-4*torch.norm(cur_vel, dim=-1))
        small_cmd_ids = (norm<0.1).squeeze(-1)
        rew[small_cmd_ids] = rew_zeros[small_cmd_ids]
        # return torch.exp(-2 * torch.norm(cur_vel - self._curr_demo_root_vel[:, :2], dim=-1))
        return rew.squeeze(-1)

    def _reward_tracking_vx(self):
        rew = torch.minimum(self.base_lin_vel[:, 0], self.commands[:, 0]) / (self.commands[:, 0] + 1e-5)
        # print("vx rew", rew, self.base_lin_vel[:, 0], self.commands[:, 0])
        return rew
    
    def _reward_tracking_ang_vel(self):
        rew = torch.minimum(self.base_ang_vel[:, 2], self.commands[:, 2]) / (self.commands[:, 2] + 1e-5)
        return rew
    
    def _reward_tracking_demo_yaw(self):
        rew = torch.exp(-torch.abs(self.target_yaw - self.yaw))
        # print("yaw rew", rew, self.target_yaw, self.yaw)
        return rew

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        # print("lower dof pos error: ", self.dof_pos - self.dof_pos_limits[:, 0])
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        # print("upper dof pos error: ", self.dof_pos - self.dof_pos_limits[:, 1])
        return torch.sum(out_of_limits, dim=1)

    def _reward_tracking_demo_dof_pos(self):
        demo_dofs = self._curr_demo_obs_buf[:, :self._n_demo_dof]
        dof_pos = self.dof_pos[:, self._dof_ids_subset]
        rew = torch.exp(-0.7 * torch.norm((dof_pos - demo_dofs), dim=1))
        # print(rew[self.lookat_id].cpu().numpy())
        # print("dof_pos", dof_pos)
        # print("demo_dofs", demo_dofs)
        return rew

    # def _reward_tracking_demo_dof_vel(self):
    #     demo_dof_vel = self._curr_demo_obs_buf[:, self.num_dof:self.num_dof*2]
    #     rew = torch.exp(- 0.01 * torch.norm(self.dof_vel - demo_dof_vel, dim=1))
    #     return rew
    
    def _reward_stand_still(self):
        dof_pos_error = torch.norm((self.dof_pos - self.default_dof_pos)[:, :11], dim=1)
        dof_vel_error = torch.norm(self.dof_vel[:, :11], dim=1)
        rew = torch.exp(- 0.1*dof_vel_error) * torch.exp(- dof_pos_error) 
        rew[~self._in_place_flag] = 0
        return rew
    
    def _reward_tracking_lin_vel(self):
        demo_vel = self._curr_demo_obs_buf[:, self._n_demo_dof:self._n_demo_dof+3]
        # demo_vel[self._in_place_flag] = 0
        rew = torch.exp(- 4 * torch.norm(self.base_lin_vel - demo_vel, dim=1))
        return rew

    def _reward_tracking_demo_ang_vel(self):
        demo_ang_vel = self._curr_demo_obs_buf[:, self._n_demo_dof+3:self._n_demo_dof+6]
        rew = torch.exp(-torch.norm(self.base_ang_vel - demo_ang_vel, dim=1))
        return rew

    def _reward_tracking_demo_roll_pitch(self):
        demo_roll_pitch = self._curr_demo_obs_buf[:, self._n_demo_dof+6:self._n_demo_dof+8]
        cur_roll_pitch = torch.stack((self.roll, self.pitch), dim=1)
        rew = torch.exp(-torch.norm(cur_roll_pitch - demo_roll_pitch, dim=1))
        return rew
    
    def _reward_tracking_demo_height(self):
        demo_height = self._curr_demo_obs_buf[:, self._n_demo_dof+8]
        cur_height = self.root_states[:, 2]
        rew = torch.exp(- 4 * torch.abs(cur_height - demo_height))
        return rew
    
    def _reward_tracking_demo_key_body(self):
        # demo_key_body_pos_local = self._curr_demo_obs_buf[:, self.num_dof*2+8:].view(self.num_envs, self._num_key_bodies, 3)[:,self._key_body_ids_sim_subset,:].view(self.num_envs, -1)
        # cur_key_body_pos_local = global_to_local(self.base_quat, self.rigid_body_states[:, self._key_body_ids_sim[self._key_body_ids_sim_subset], :3], self.root_states[:, :3]).view(self.num_envs, -1)
        
        demo_key_body_pos_local = self._curr_demo_keybody.view(self.num_envs, self._num_key_bodies, 3)
        if self.cfg.motion.global_keybody:
            curr_demo_xyz = torch.cat((self.target_pos_abs, self._curr_demo_root_pos[:, 2:3]), dim=-1)
        else:
            curr_demo_xyz = torch.cat((self.root_states[:, :2], self._curr_demo_root_pos[:, 2:3]), dim=-1)
        demo_global_body_pos = local_to_global(self._curr_demo_quat, demo_key_body_pos_local, curr_demo_xyz).view(self.num_envs, -1)
        cur_global_body_pos = self.rigid_body_states[:, self._key_body_ids_sim[self._key_body_ids_sim_subset], :3].view(self.num_envs, -1)

        # cur_local_body_pos = global_to_local(self.base_quat, cur_global_body_pos.view(self.num_envs, -1, 3), self.root_states[:, :3]).view(self.num_envs, -1)
        # print(cur_local_body_pos)
        rew = torch.exp(-torch.norm(cur_global_body_pos - demo_global_body_pos, dim=1))
        # print("key body rew", rew[self.lookat_id].cpu().numpy())
        return rew

    def _reward_tracking_mul(self):
        rew_key_body = self._reward_tracking_demo_key_body()
        rew_roll_pitch = self._reward_tracking_demo_roll_pitch()
        rew_ang_vel = self._reward_tracking_demo_yaw()
        # rew_dof_vel = self._reward_tracking_demo_dof_vel()
        rew_dof_pos = self._reward_tracking_demo_dof_pos()
        # rew_goal_vel = self._reward_tracking_lin_vel()#self._reward_tracking_demo_goal_vel()
        rew = rew_key_body * rew_roll_pitch * rew_ang_vel * rew_dof_pos# * rew_dof_vel
        # print(self._curr_demo_obs_buf[:, self.num_dof:self.num_dof+3][self.lookat_id], self.base_lin_vel[self.lookat_id])
        return rew
    # def _reward_tracking_demo_vel(self):
    #     demo_vel = self.get_curr_demo()[:, self.num_dof:]
    def _reward_feet_drag(self):
        # print(contact_bool)
        # contact_forces = self.contact_forces[:, self.feet_indices, 2]
        # print(contact_forces[self.lookat_id], self.force_sensor_tensor[self.lookat_id, :, 2])
        # print(self.contact_filt[self.lookat_id])
        feet_xyz_vel = torch.abs(self.rigid_body_states[:, self.feet_indices, 7:10]).sum(dim=-1)
        dragging_vel = self.contact_filt * feet_xyz_vel
        rew = dragging_vel.sum(dim=-1)
        # print(rew[self.lookat_id].cpu().numpy(), self.contact_filt[self.lookat_id].cpu().numpy(), feet_xy_vel[self.lookat_id].cpu().numpy())
        return rew
    
    def _reward_energy(self):
        return torch.norm(torch.abs(self.torques * self.dof_vel), dim=-1)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        # rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        rew_airTime[self._in_place_flag] = 0
        return rew_airTime

    def _reward_feet_height(self):
        feet_height = self.rigid_body_states[:, self.feet_indices, 2]
        rew = torch.clamp(torch.norm(feet_height, dim=-1) - 0.2, max=0)
        rew[self._in_place_flag] = 0
        # print("height: ", rew[self.lookat_id])
        return rew
    
    def _reward_feet_force(self):
        rew = torch.norm(self.contact_forces[:, self.feet_indices, 2], dim=-1)
        rew[rew < 500] = 0
        rew[rew > 500] -= 500
        rew[self._in_place_flag] = 0
        # print(rew[self.lookat_id])
        # print(self.dof_names)
        return rew

    def _reward_dof_error(self):
        # dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, :11], dim=1)
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return dof_error
    


    def _reward_ankle_action(self):
        return torch.norm(self.actions[:, [4, 5, 10, 11]], dim=1)
        # return torch.norm(self.actions[:, [5, 11]], dim=1)
    def _reward_waist_roll_pitch(self):
        return torch.sum(torch.square(self.dof_pos[:, [13, 14]] - self.default_dof_pos[:, [13, 14]]), dim=1)
    def _reward_feet_velocity(self):
        feet_xyz_vel = torch.abs(self.rigid_body_states[:, self.feet_indices, 7:10]).sum(dim=-1)
        rew = feet_xyz_vel.sum(dim=-1)
        return rew



    def _reward_fix_legs(self):
        dof_pos = self.dof_pos[:, :12]
        # print()
        # # print("dof_pos ", dof_pos.shape, dof_pos)
        # print("dof_pos ", dof_pos[0,:])
        # print()
        rew = torch.exp(-0.7 * torch.norm((dof_pos), dim=1))
        # print(rew[self.lookat_id].cpu().numpy())
        # print("dof_pos", dof_pos)
        # print("demo_dofs", demo_dofs)
        return rew
#####################################################################
###=========================jit functions=========================###
#####################################################################

# @torch.jit.script
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
    # print()
    # print("dof_pos in build demo: ", dof_pos.shape, dof_pos)
    # print()
    # print("dof_pos",dof_pos) 
    # print("local_root_vel",local_root_vel) 
    # print("local_root_ang_vel",local_root_ang_vel) 
    # print("roll",roll[:, None]) 
    # print("pitch",pitch[:, None]) 
    # print("root_pos[:, 2:3]", root_pos[:, 2:3]) 
    # print("local_key_body_pos",local_key_body_pos.view(local_key_body_pos.shape[0], -1))

    # return torch.cat((dof_pos, local_root_vel, local_root_ang_vel, roll[:, None], pitch[:, None], root_pos[:, 2:3], torch.zeros_like(local_key_body_pos.view(local_key_body_pos.shape[0], -1))), dim=-1)
    return torch.cat((dof_pos, local_root_vel, local_root_ang_vel, roll[:, None], pitch[:, None], root_pos[:, 2:3], local_key_body_pos.view(local_key_body_pos.shape[0], -1)), dim=-1)

@torch.jit.script
def reindex_motion_dof(dof, indices_sim, indices_motion, valid_dof_body_ids):
    dof = dof.clone()
    dof[:, indices_sim] = dof[:, indices_motion]
    return dof[:, valid_dof_body_ids]

@torch.jit.script
def local_to_global(quat, rigid_body_pos, root_pos):
    num_key_bodies = rigid_body_pos.shape[1]
    num_envs = rigid_body_pos.shape[0]
    total_bodies = num_key_bodies * num_envs
    heading_rot_expand = quat.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, num_key_bodies, 1))
    flat_heading_rot = heading_rot_expand.view(total_bodies, heading_rot_expand.shape[-1])

    flat_end_pos = rigid_body_pos.reshape(total_bodies, 3)
    global_body_pos = quat_rotate(flat_heading_rot, flat_end_pos).view(num_envs, num_key_bodies, 3) + root_pos[:, None, :3]
    return global_body_pos

@torch.jit.script
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

@torch.jit.script
def global_to_local_xy(yaw, global_pos_delta):
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    rotation_matrices = torch.stack([cos_yaw, sin_yaw, -sin_yaw, cos_yaw], dim=2).view(-1, 2, 2)
    local_pos_delta = torch.bmm(rotation_matrices, global_pos_delta.unsqueeze(-1))
    return local_pos_delta.squeeze(-1)