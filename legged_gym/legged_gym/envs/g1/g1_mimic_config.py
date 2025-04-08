# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class G1MimicCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 6144

        n_demo_steps = 2
        # n_demo = 9 + 3 + 3 + 3 +6*3 + 9 + 8 #observe height
        n_demo = 9 + 3 + 3 + 3 +6*3 + 9 + 23 - 4 #without ankle
        # n_demo = 9 + 3 + 3 + 3 +6*3 + 9 + 23 #- 15 - 6*3 #observe height
        # n_demo = 9 + 3 + 3 + 3 +6*3 #observe height
        interval_demo_steps = 0.1

        n_scan = 0#132
        n_priv = 3
        n_priv_latent = 4 + 1 + 23*2
        n_proprio = 3 + 2 + 2 + 23*3 + 2 # one hot
        # n_priv_latent = 4 + 1 + 19*2
        # n_proprio = 3 + 2 + 2 + 19*3 + 2 # one hot

        history_len = 10

        prop_hist_len = 4
        n_feature = prop_hist_len * n_proprio

        num_observations = n_feature + n_proprio + n_demo + n_scan + history_len*n_proprio + n_priv_latent + n_priv

        episode_length_s = 50 # episode length in seconds
        
        num_policy_actions = 23
        # num_policy_actions = 19
    
    class motion:
        motion_curriculum = True
        motion_type = "yaml"
        # motion_name = "motions_autogen_all_no_run_jump.yaml"
        # motion_name = "motions_50.yaml"
        # motion_name = "motions_50+andre_additions.yaml"
        # motion_name = "motions_50+curated_additions.yaml"
        # motion_name = "motions_50+picking.yaml"
        motion_name = "picking.yaml"

        # motion_type = "single"
        # motion_name = "0-ACCAD_Female1General_c3d_A1 - Stand_poses"
        # motion_name = "res_4"
        # motion_name = "0-CMU_79_79_32_poses"

        # motion_name = "res_hi"
        # motion_name = "res_box"
        # motion_name = "res_heart1"
        # motion_name = "res_heart2"
        # motion_name = "res"
        # motion_name = "res_1"
        # motion_name = "res_2"
        # motion_name = "res_3"
        # motion_name = "res_5"
        # motion_name = "res_6"
        
        global_keybody = False
        global_keybody_reset_time = 2

        num_envs_as_motions = False

        # no_keybody = True #False
        no_keybody = False
        regen_pkl = False

        step_inplace_prob = 0.05
        resample_step_inplace_interval_s = 10


    class terrain( LeggedRobotCfg.terrain ):
        horizontal_scale = 0.1 # [m] influence computation time by a lot
        height = [0., 0.04]
    
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        # pos = [0.0, 0.0, 1.1] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'left_hip_pitch_joint' : -0.1, #0.,
            'left_hip_roll_joint' : 0.,
            'left_hip_yaw_joint' : 0.,
            'left_knee_joint' : 0.3, #0.,
            'left_ankle_pitch_joint' : -0.2, #0.,
            'left_ankle_roll_joint' : 0.,
            'right_hip_pitch_joint' : -0.1, #0.,
            'right_hip_roll_joint' : 0.,
            'right_hip_yaw_joint' : 0.,
            'right_knee_joint' : 0.3, #0.,
            'right_ankle_pitch_joint' : -0.2, #0.,
            'right_ankle_roll_joint' : 0.,
            'waist_yaw_joint' : 0.,
            'waist_roll_joint' : 0.,
            'waist_pitch_joint' : 0.,
            'left_shoulder_pitch_joint' : 0.,
            'left_shoulder_roll_joint' : 0.,
            'left_shoulder_yaw_joint' : 0.,
            'left_elbow_joint' : 0., #left_wrist_roll_joint left_wrist_pitch_joint left_wrist_yaw_joint
            'right_shoulder_pitch_joint' : 0.,
            'right_shoulder_roll_joint' : 0.,
            'right_shoulder_yaw_joint' : 0.,
            'right_elbow_joint' : 0.,  #right_wrist_roll_joint right_wrist_pitch_joint right_wrist_yaw_joint
        }


    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        # control_type = 'P'
        # stiffness = {'hip_yaw': 100,
        #              'hip_roll': 100,
        #              'hip_pitch': 100,
        #              'knee': 150,
        #              'ankle': 40,
        #              'torso': 200,
        #              'shoulder': 40,
        #              "elbow":40,
        #              "waist": 400,
        #              }  # [N*m/rad]
        # damping = {  'hip_yaw': 2,
        #              'hip_roll': 2,
        #              'hip_pitch': 2,
        #              'knee': 4,
        #              'ankle': 2,
        #              'torso': 4,
        #              'shoulder': 2,
        #              "elbow":2,
        #              "waist": 6,
        #              }  # [N*m/rad]  # [N*m*s/rad]   

        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 200,
                     'ankle_pitch': 20,
                     'ankle_roll': 20,
                     'shoulder_pitch': 90,
                     'shoulder_roll': 60,
                     'shoulder_yaw': 20,
                     "elbow":60,
                     "waist": 400,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2.5,
                     'hip_roll': 2.5,
                     'hip_pitch': 2.5,
                     'knee': 5.0,
                     'ankle_pitch': 0.2,
                     'ankle_roll': 0.1,
                     'torso': 4,
                     'shoulder_pitch': 2.0,
                     'shoulder_roll': 1.0,
                     'shoulder_yaw': 0.4,
                     "elbow":1.0,
                     "waist": 5.0,
                     }  # [N*m/rad]  # [N*m*s/rad]   


        action_scale = 0.25
        decimation = 4

    class normalization( LeggedRobotCfg.normalization):
        clip_actions = 10

    class asset( LeggedRobotCfg.asset ):
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/h1_blue_red_custom_collision.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/h1_custom_collision.urdf'
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/g1_29dof_rev_1_0_custom.urdf'
        # torso_name = "torso_link"
        torso_name = "waist_yaw_link"
        foot_name = "ankle_roll"
        # foot_name = "ankle"
        penalize_contacts_on = ["shoulder", "elbow", "hip", "torso", "waist"]
        # terminate_after_contacts_on = ["torso_link", ]#, "thigh", "calf"]
        terminate_after_contacts_on = []# ["waist_yaw_link"]# ["waist_yaw_link", ]#, "thigh", "calf"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            # tracking rewards
            alive = 1 #not in Exbody2
            # tracking_demo_goal_vel = 1.0
            # tracking_mul = 6
            tracking_lin_vel = 6
            # stand_still = 3
            # tracking_goal_vel = 4


            tracking_demo_yaw = 1
            tracking_demo_roll_pitch = 1
            # orientation = -2  # not in Exbody2
            tracking_demo_dof_pos = 3
            # tracking_demo_dof_vel = 1.0
            tracking_demo_key_body = 2
            # tracking_demo_height = 1  # useful if want better height tracking
            
            # tracking_demo_lin_vel = 1
            # tracking_demo_ang_vel = 0.5


            # regularization rewards
            # lin_vel_z = -1.0
            # ang_vel_xy = -0.4 #!
            # orientation = -2.           # not in Exbody2 # projected gravity
            dof_acc = -3e-7
            collision = -0.1            # not in Exbody2
            action_rate = -0.1
            # delta_torques = -1.0e-7
            # torques = -1e-5
            # energy =-1e-3

            # hip_pos = -0.2              # not in Exbody1
            
            dof_error = -0.1                     ## dof deviation in  Exbody1
            feet_stumble = -2
            # feet_edge = -1
            # feet_drag = -0.1            # not in Exbody2
            dof_pos_limits = -10.0              ## dof limit violation in  Exbody1
            feet_air_time = 10
            # feet_height = 2.            # not in Exbody2
            feet_force = -3e-3

            waist_roll_pitch = -0.5 #-0.5      # not in Exbody1
            ankle_action = -0.1          # not in Exbody1
            # feet_velocity = -0.1         # not in Exbody1

            # fix_legs = 0

        only_positive_rewards = False
        clip_rewards = True
        soft_dof_pos_limit = 0.95
        base_height_target = 0.25

        
    class domain_rand( LeggedRobotCfg.domain_rand ):
        randomize_gravity = False ###True
        gravity_rand_interval_s = 10
        gravity_range = [-0.1, 0.1]
    
    class noise():
        add_noise = True
        noise_scale = 0.5 # scales other values
        class noise_scales():
            dof_pos = 0.01
            dof_vel = 0.15
            ang_vel = 0.3
            imu = 0.2
    

class G1MimicCfgPPO( LeggedRobotCfgPPO ):
    class runner( LeggedRobotCfgPPO.runner ):
        runner_class_name = "OnPolicyRunnerMimic"
        policy_class_name = 'ActorCriticMimic'
        algorithm_class_name = 'PPOMimic'
    
    class policy( LeggedRobotCfgPPO.policy ):
        continue_from_last_std = False
        text_feat_input_dim = G1MimicCfg.env.n_feature
        text_feat_output_dim = 16
        feat_hist_len = G1MimicCfg.env.prop_hist_len
        # actor_hidden_dims = [1024, 512]
        # critic_hidden_dims = [1024, 512]
    
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.005

    class estimator:
        train_with_estimated_states = False
        learning_rate = 1.e-4
        hidden_dims = [128, 64]
        priv_states_dim = G1MimicCfg.env.n_priv
        priv_start = G1MimicCfg.env.n_feature + G1MimicCfg.env.n_proprio + G1MimicCfg.env.n_demo + G1MimicCfg.env.n_scan
        
        prop_start = G1MimicCfg.env.n_feature
        prop_dim = G1MimicCfg.env.n_proprio

class G1MimicDistillCfgPPO( G1MimicCfgPPO ):
    class distill:
        num_demo = 3
        num_steps_per_env = 24
        
        num_pretrain_iter = 0

        activation = "elu"
        learning_rate = 1.e-4
        student_actor_hidden_dims = [1024, 1024, 512]

        num_mini_batches = 4