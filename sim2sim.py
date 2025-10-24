# -*- coding: utf8 -*-

'''
加载一个预训练的策略模型，在MuJoCo仿真环境中控制四足机器人执行运动任务
'''

import time
import mujoco.viewer
import mujoco
# MuJoCo模拟环境
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from base_pose_estimator import BasePoseEstimator
from velocity_estimator import VelocityEstimator


def rotate_vector_by_quaternion(vector: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    '''
    rotate a vector by quaternion

    parameters:
        vector: np.ndarray, translation vector, [3]
        quaternion: np.ndarray, rotation quaternion, [4], [w, x, y, z]
    
        returns:
            v: rotated vector, R @ t, [3]
    '''
    qw, qx, qy, qz = quaternion
    qx = -qx
    qy = -qy
    qz = -qz
    qq = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return np.dot(qq, vector)


def get_gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    '''
    get gravity orientation from quaternion

    parameters:
        quaternion: [4], [w, x, y, z]
    returns:
        v: gravitation orientation, [3]
    '''
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def joint_torque(action, joint_pos_rel, joint_vel_rel):
    '''


    parameters:
        action: np.ndarray, shape = [12]
        joint_pos_rel: np.ndarray, shape = [12]
        joint_vel_rel: np.ndarray, shape = [12]
    returns:
        effort: np.ndarray, shape = [12],
    '''
    def clip_effort(effort: np.ndarray, 
                    joint_vel: np.ndarray, 
                    saturation_effort: float = 23.5, 
                    effort_limit: float = 23.5, 
                    velocity_limit: float = 30.0):
        '''


        parameters:
            effort: np.ndarray, shape = [12],
            joint_vel: np.ndarray, shape = [12],
        '''
        max_effort = saturation_effort * (1.0 - joint_vel / velocity_limit)
        max_effort = np.clip(max_effort, a_min = 0.0, a_max = effort_limit)

        min_effort = saturation_effort * (-1.0 - joint_vel / velocity_limit)
        min_effort = np.clip(min_effort, a_min = -effort_limit, a_max = 0.0)
        
        return np.clip(effort, a_min = min_effort, a_max = max_effort)

    def compute(target_joint_pos: np.ndarray, 
                joint_pos: np.ndarray, 
                joint_vel: np.ndarray, 
                stiffness: float = 50.0, 
                damping: float = 0.5):
        '''
        PD (proportion-difference) controller 


        parameters:
            target_joint_pos: np.ndarray, shape = [12], targte joint position
            joint_pos: np.ndarray, shape = [12], current joint position
            joint_vel: np.ndarray, shape = [12], current joint velocity
        returns:
            effort: np.ndarray, shape = [12],
        '''
        error_pos = target_joint_pos - joint_pos
        error_vel = -joint_vel
        computed_effort = stiffness * error_pos + damping * error_vel
        return clip_effort(computed_effort, joint_vel)

    return compute(action, joint_pos_rel, joint_vel_rel)


def pd_control(target_q: np.ndarray, 
               q: np.ndarray, 
               kp: float, 
               target_dq: float, 
               dq: np.ndarray, 
               kd: float):
    '''


    parameters:
        target_q: np.ndarray, shape = [12]
        q: np.ndarray, shape = [12]
        dq: np.ndarray, shape = [12]

    '''
    return joint_torque(target_q, q, dq)


def isaac2mujoco(inputs: np.ndarray) -> np.ndarray:
    '''

    parameters:
        inputs: np.ndarray, [12]
    returns:
        output: np.ndarray, [12]
    '''
    return inputs.reshape([3, 4]).T.flatten()


def mujoco2isaac(inputs: np.ndarray) -> np.ndarray:
    '''

    parameters:
        inputs: np.ndarray, [12]
    returns:
        output: np.ndarray, [12]
    '''
    return inputs.reshape([4, 3]).T.flatten()







def main():
    '''
    '''
    config = {
        "policy_path": "./policy.pt",
        "xml_path": "./robots/go2/scene.xml",
        "simulation_duration": 3000,
        "simulation_dt": 0.002,
        "control_decimation": 10,
        "kps": 50.0,
        "kds": 0.1,
        "default_angles": [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5],
        # len = 12

        "ang_vel_scale": 1.0,
        "dof_pos_scale": 1.0,
        "dof_vel_scale": 1.0,
        "action_scale": 0.2,
        "cmd_scale": 1.0,
        "num_actions": 12,
        "num_obs": 48,
        "cmd_init": [0.5, 0.0, 2.0], 
        # 前向速度、横向速度、转向速度. 向前1.5m/s，没有横向和旋转速度
    }

    policy_path = config["policy_path"]
    xml_path = config["xml_path"]

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = config["simulation_dt"]

    viewer = mujoco.viewer.launch_passive(m, d)
    policy = torch.jit.load(policy_path)

    estimator = BasePoseEstimator(m)
    vel_estimator = VelocityEstimator(m)

    default_angles = np.array(config["default_angles"], dtype=np.float32)
    # [12]

    num_actions = config["num_actions"]
    num_obs = config["num_obs"]

    action = np.zeros(num_actions, dtype=np.float32)
    # shape = [12]
    target_dof_pos = default_angles.copy()
    # [12]
    obs = np.zeros(num_obs, dtype=np.float32)
    # [48]
    cmd = np.array(config["cmd_init"], dtype=np.float32)
    # [3]

    counter = 0
    start = time.time()

    while viewer.is_running() and time.time() - start < config["simulation_duration"]:
        step_start = time.time()

        # PD控制计算关节力矩
        tau = pd_control(target_dof_pos, 
                         d.qpos[7:], 
                         config["kps"], 
                         np.zeros_like(config["kds"]), 
                         d.qvel[6:], 
                         config["kds"])
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        # 执行仿真步

        counter += 1
        # 定期更新策略（每10步）
        if counter % config["control_decimation"] == 0:
            qj = d.qpos[7:]
            # [12]
            dqj = d.qvel[6:]
            # [12]
            quat = d.qpos[3:7]
            # [4]
            omega = d.qvel[3:6]
            # [3]

            qj = (qj - default_angles) * config["dof_pos_scale"]
            dqj = dqj * config["dof_vel_scale"]
            gravity_orientation = get_gravity_orientation(quat)
            omega = omega * config["ang_vel_scale"]

            # 构建观测向量
            obs[:3] = rotate_vector_by_quaternion(d.qvel[:3], quat)
            # velocity, 基座线速度（机体坐标系）
            obs[3:6] = d.qvel[3:6]
            # angle velocity, 基座角速度
            obs[6:9] = gravity_orientation
            # gravity orientation
            obs[9:12] = cmd
            # command（前向速度、横向速度、转向速度）
            obs[12:12 + num_actions] = mujoco2isaac(qj)
            # joint position （相对默认姿态）
            obs[12 + num_actions:12 + 2 * num_actions] = mujoco2isaac(dqj)
            # joint velocity
            obs[12 + 2 * num_actions:12 + 3 * num_actions] = mujoco2isaac(action)
            # previous action

            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            # [1, 48]

            # 策略网络推理
            action = policy(obs_tensor).detach().numpy().squeeze()
            # [12]
            action = isaac2mujoco(action)

            # 更新目标关节位置
            target_dof_pos = action * config["action_scale"] + default_angles
            
        if counter % 200 == 0:
            # pos and pose estimation
            foot_pos = estimator.get_foot_pos_from_data(d)
            joint_angles = d.qpos[7:(7 + 12)].copy()
            base_pos_gt = d.qpos[:3].copy()
            base_quat_gt = d.qpos[3:7].copy()
            base_rot_mat = R.from_quat(base_quat_gt).as_matrix()

            base_pos_est1, base_pos_est_var1 = estimator(foot_pos, joint_angles, base_quat_gt)
            (base_pos_est2, base_pos_est_var2), base_rot_mat_est = estimator(foot_pos, joint_angles)

            base_pos_err1 = np.sum((base_pos_est1 - base_pos_gt) ** 2)
            base_pos_err2 = np.sum((base_pos_est2 - base_pos_gt) ** 2)
            rot_err = np.arccos((np.trace(base_rot_mat @ base_rot_mat_est.T) - 1) / 2)

            # velocity estimation
            base_angle_vel = d.qvel[3:6].copy()
            base_vel_gt = d.qvel[:3].copy()
            joint_vel = d.qvel[6:(6 + 12)].copy()
            base_vel_est, base_vel_est_var = vel_estimator(base_pos_gt, 
                                                           base_quat_gt, 
                                                           base_angle_vel, 
                                                           joint_angles, 
                                                           joint_vel, 
                                                           foot_pos)
            base_vel_err = np.sum((base_vel_est - base_vel_gt) ** 2)
            
            
            print(f"known base_quat: pos err = {base_pos_err1:.2e}"
                  f", pos var = {base_pos_est_var1:.2e}"
                  "\n"
                  f"unknown base_quat: pos err = {base_pos_err2:.2e}"
                  f", pos var = {base_pos_est_var2:.2e}, "
                  f"rot err = {rot_err:.2e} rad\n"
                  f"vel err = {base_vel_err:.2e}"
                  f", vel var = {base_vel_est_var:.2e}"
                  f"\n"
                  )
        
        # foot_pos = estimator.get_foot_pos_from_data(d)
        # print(foot_pos)
        viewer.sync()
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
    return


if __name__ == "__main__":
    main()

