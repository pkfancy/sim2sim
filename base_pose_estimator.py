# -*- coding: utf8 -*-


'''
'''

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

class BasePoseEstimator:
    '''
    '''
    def __init__(self, model):
        '''
        '''
        self.model = model
        # data = mujoco.MjData(self.model)
        
        # foot ids
        self.foot_site_names = ["FL", "FR", "RL", "RR"]
        # from .xml file
        self.foot_site_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                              for name in self.foot_site_names]
        # 
        
        # joint ids
        self.joint_names = [f"{s1}{s2}_{part}_joint" 
                            for s1 in "FR" 
                            for s2 in "LR" 
                            for part in ["hip", "thigh", "calf"]]
        # from .xml file
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) 
                          for name in self.joint_names]
        return
    
    def forward_kinematics(self, 
                           base_pos: np.ndarray, 
                           base_quat: np.ndarray, 
                           joint_angles: np.ndarray):
        '''
        get the positions of 4 feet by forward kinematics of the robot
        given the base posision, base quaternion and joint angles

        params:
            base_pos: np.ndarray, base position, shape = [3]
            base_quat: np.ndarray, base rotation quaternion, shape = [4]
            joint_angles: np.ndarray, angles of each joint of each leg, shape = [4, 3]
        returns:
            foot_pos: np.ndarray, foot positions, shape = [4, 3]
                ["FL", "FR", "RL", "RR"]
        '''
        data = mujoco.MjData(self.model)
        # data.qpos = [base_pos (shape = [3]), base_quat (shape = [4]), joint_angles (shape = [12])]
        data.qpos[:3] = base_pos
        data.qpos[3:7] = base_quat
        data.qpos[7:] = joint_angles.reshape([-1])
        
        # set the robot state according to data
        mujoco.mj_kinematics(self.model, data)
        
        foot_pos = self.get_foot_pos_from_data(data)        
        return foot_pos
    
    def get_foot_pos_from_data(self, data):
        '''
        get foot positions from data

        params:
            data: mujoco.MjData
        returns:
            foot_pos: np.ndarray, foot positions, shape = [4, 3]
                ["FL", "FR", "RL", "RR"]
        '''
        foot_pos = np.array([data.geom_xpos[foot_id].copy() 
                             for foot_id in self.foot_site_ids 
                             if foot_id != -1])
        return foot_pos
    

    def __call__(self, 
                 foot_pos: np.ndarray, 
                 joint_angles: np.ndarray, 
                 base_quat: np.ndarray = None):
        '''
        Given the foot positions and joint angles, 
        solve the position and rotation of the robot base.
        
        To this end, we use a robot copy and set the robot legs with the given joint angles, 
        then the copy robot and the real robot have only a difference of Euler transform, 
        that is, a translation and a rotation, which is the position and rotation of the robot base.
        
        when base_quat is known, base_pos = foot_i_pos - foot_i_pos_base, for i = 1, 2, 3, 4
        when base_quat is unknown, foot_i_pos = R @ foot_i_pos_base + base_pos, for i = 1, 2, 3, 4
        R and t can be solved by centralization and SVD
        foot_pos_mean = R @ foot_pos_base_mean + base_pos
        (foot_i_pos - foot_pos_mean) = R @ (foot_i_pos_base - foot_pos_base_mean), for i = 1, 2, 3, 4

        params:
            foot_pos: np.ndarray, shape = [4, 3]
            joint_angles: np.ndarray, shape = [4, 3]
            base_quat: np.ndarray = None, shape = [4]
        returns:
            base_pos_est: np.ndarray, shape = [3]
            base_pos_est_err: float
            # base_quat_est: np.ndarray, shape = [4]
            base_rot_mat: np.ndarray, [3, 3]
        '''
        base_pos0 = np.zeros([3])
        if base_quat is not None:
            foot_pos_base = self.forward_kinematics(base_pos = base_pos0, 
                                                    base_quat = base_quat, 
                                                    joint_angles = joint_angles)
            base_pos_arr = foot_pos - foot_pos_base
            base_pos_est = np.mean(base_pos_arr, axis = 0)
            base_pos_est_var = np.mean(np.sum((base_pos_arr - base_pos_est) ** 2, axis = 1))
            return base_pos_est, base_pos_est_var
        else:
            base_quat0 = np.array([1, 0, 0, 0])
            foot_pos_base = self.forward_kinematics(base_pos = base_pos0, 
                                                    base_quat = base_quat0, 
                                                    joint_angles = joint_angles)
            
            foot_pos_mean = np.mean(foot_pos, axis = 0)
            foot_pos1 = foot_pos - foot_pos_mean

            foot_pos_base_mean = np.mean(foot_pos_base, axis = 0)
            foot_pos_base1 = foot_pos_base - foot_pos_base_mean

            U1, e1, V1 = np.linalg.svd(foot_pos1)
            U2, e2, V2 = np.linalg.svd(foot_pos_base1)
            R = (V1.T @ V2).T
            # base_quat_est = Rotation.from_matrix(R).as_quat()

            base_pos_arr = foot_pos - foot_pos_base @ R
            base_pos_est = np.mean(base_pos_arr, axis = 0)
            base_pos_est_var = np.mean(np.sum((base_pos_arr - base_pos_est) ** 2, axis = 1))
            # return (base_pos_est, base_pos_est_err), base_quat_est
            return (base_pos_est, base_pos_est_var), R


def compute_dist_mat(x1: np.ndarray, x2: np.ndarray):
    '''
    '''
    dist = (np.sum(x1 ** 2, axis = 1, keepdims = True)
            + np.sum(x2 ** 2, axis = 1, keepdims = True).T 
            - 2 * x1 @ x2.T)
    return dist


def normalize(x, axis = None):
    '''
    '''
    eps = 1e-5
    x1 = x / (np.sqrt(np.sum(x ** 2, axis)) + eps)
    return x1


def main():
    '''
    '''
    # base_pos0 = np.zeros([3])
    # base_quat0 = np.array([1, 0, 0, 0])
    # foot_pos_base = estimator.forward_kinematics(base_pos0, base_quat0, joint_angles)

    # foot_pos_mean = np.mean(foot_pos, axis = 0)
    # foot_pos1 = foot_pos - foot_pos_mean

    # foot_pos_base_mean = np.mean(foot_pos_base, axis = 0)
    # foot_pos_base1 = foot_pos_base - foot_pos_base_mean

    # foot_pos1 - foot_pos_base1 @ Rotation.from_quat(base_quat_gt).as_matrix().T


    # compute_dist_mat(foot_pos, foot_pos)
    # compute_dist_mat(foot_pos_base, foot_pos_base)
    return

if __name__ == "__main__":
    main()
