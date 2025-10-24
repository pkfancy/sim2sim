# -*- coding: utf8 -*-

'''
'''

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation


class VelocityEstimator:
    '''
    '''
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
        
        # reset
        # data.qpos[:3] = 0
        # data.qpos[3:7] = np.array([1, 0, 0, 0])
        # data.qpos[7:] = 0
        # mujoco.mj_kinematics(self.model, data)

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
    
    def get_stance_feet(self, 
                        foot_pos: np.ndarray, 
                        ground_height_th: float = 1e-2):
        '''

        params:
            foot_pos: np.ndarray, [4, 3]
            ground_height_th: float = 5e-2
        returns:
            stance_mask: np.ndarray of bools, [4]
        '''
        stance_mask = foot_pos[:, 2] < ground_height_th
        return stance_mask
    
    def compute_foot_jacobians(self, 
                               base_pos: np.ndarray, 
                               base_quat: np.ndarray, 
                               joint_angles: np.ndarray):
        '''


        params:
            joint_angles: np.ndarray, [4, 3]
        returns:
            jacobians: np.ndarray, shape [4, 3, 12]
        '''
        data = mujoco.MjData(self.model)
        data.qpos[:3] = base_pos
        data.qpos[3:7] = base_quat
        data.qpos[7:(7 + 12)] = joint_angles.reshape([-1])
        mujoco.mj_kinematics(self.model, data)
        
        jacobians = []
        for foot_id in self.foot_site_ids:
            jac_pos = np.zeros([3, self.model.nv])
            jac_rot = np.zeros([3, self.model.nv])
            mujoco.mj_jacGeom(self.model, data, jac_pos, jac_rot, foot_id)
            jacobians.append(jac_pos[:, 6:(6 + 12)])
            # skip bas postion and rotation, get only 12 joint params
            # shape = [3, 12]
        jacobians = np.array(jacobians)
        return jacobians
    

    def __call__(self, 
                 
                 base_pos: np.ndarray, 
                 base_quat: np.ndarray, 
                 base_ang_vel: np.ndarray, 
                 
                 joint_angles: np.ndarray, 
                 joint_vel: np.ndarray, 
                 
                 foot_pos: np.ndarray):
        '''


        params:
            base_ang_vel: np.ndarray, [3]
            base_quat: np.ndarray, [4]
            joint_angles: np.ndarray, [4, 3]
        returns:
            base_vel_est: np.ndarray, [3]
        '''
        stance_mask = self.get_stance_feet(foot_pos)
        assert np.any(stance_mask)
        
        jacobians = self.compute_foot_jacobians(base_pos, base_quat, joint_angles)
        
        base_vel_arr = []
        for i, is_stance in enumerate(stance_mask):
            if is_stance:
                J = jacobians[i]
                foot_joint_vel_in_base = J @ joint_vel

                foot_pos_in_base = foot_pos[i] - base_pos
                foot_rot_vel_in_base = np.cross(base_ang_vel, foot_pos_in_base)
                
                base_vel0 = -(foot_joint_vel_in_base + foot_rot_vel_in_base)
                base_vel_arr.append(base_vel0)
        
        base_vel_arr = np.array(base_vel_arr)
        base_vel_est = np.mean(base_vel_arr, axis = 0)
        base_vel_est_var = np.mean(np.sum((base_vel_arr - base_vel_est) ** 2, axis = 1))
        return base_vel_est, base_vel_est_var


def main():
    '''
    '''
    return

if __name__ == "__main__":
    main()
