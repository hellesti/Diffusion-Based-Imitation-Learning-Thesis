import pathlib
import sys

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)

import numpy as np
from diffusion_policy.common.replay_buffer import ReplayBuffer
import NetFT
from diffusion_policy.common.precise_sleep import precise_wait
import time

#We define the end-effector coordinate system such that in the initialization pose, the gripper's axes [+x,+y,+z] ... 
#We measure the intended pose (tool flange z vertical, facing down)
#Then we measure the rotation when the gripper is positioned equivalently.
#positional offset is found from geometry

#The tool offset is defined such that R_02*tool_offset_rot_gripper = R_01
#therefore tool_offset_rot_gripper=R_20*R_01
#and tool_offset_rot_suctioncup=R_30*R_01
import scipy.spatial.transform as st


R_01 = (st.Rotation.from_rotvec(np.array([2.2235,-2.2260, -0.0086]))).as_matrix() #base to flange rotvec when flange is vertical
R_02 = (st.Rotation.from_rotvec(np.array([1.0283,-3.7372,0.4158]))).as_matrix() #base to flange rotvec when gripper is vertical
R_03 = (st.Rotation.from_rotvec(np.array([2.6349,0.7249,1.0924]))).as_matrix() #base to flange rotvec when suctioncup is vertical

R=(R_02.T)@R_01
#tool_offset_rot_gripper = st.Rotation.from_matrix(R).as_rotvec()

T = np.hstack((R,np.array([0,0,14.15 + 39.66 + 16.033]).reshape(3,1)))
T = np.vstack((T, np.array([0,0,0,1])))
T = T @np.array([[1,0,0,0],[0,1,0,0],[0,0,1,11.34 + 176.8],[0,0,0,1]])
gripper_tcp_offset = np.hstack(( T[:3,3]/1000,st.Rotation.from_matrix(T[:3,:3]).as_rotvec())) 
print("gripper:",list(gripper_tcp_offset))

R = (R_03.T)@R_01

T = np.hstack((R,np.array([0,0,14.15 + 39.66 + 16.033]).reshape(3,1)))
T = np.vstack((T, np.array([0,0,0,1])))
T = T @np.array([[1,0,0,0],[0,1,0,0],[0,0,1,11.34 + 186.6],[0,0,0,1]])
suctioncup_tcp_offset = np.hstack((T[:3,3]/1000,st.Rotation.from_matrix(T[:3,:3]).as_rotvec())) 
print("stick/suctioncup:",list(suctioncup_tcp_offset))

# tool_offset_rot_suctioncup = st.Rotation.from_matrix((R_03.T)@R_01).as_rotvec()
# print(tool_offset_rot_gripper)
# print(tool_offset_rot_suctioncup)

##Gripper fot T reset
R=(R_02.T)@R_01
#tool_offset_rot_gripper = st.Rotation.from_matrix(R).as_rotvec()

T = np.hstack((R,np.array([0,0,14.15 + 39.66 + 16.033]).reshape(3,1)))
T = np.vstack((T, np.array([0,0,0,1])))
T = T @np.array([[1,0,0,-39],[0,1,0,0],[0,0,1,11.34 + 176.8],[0,0,0,1]]) #x offset of 39mm makes tcp be in centre of T-block when resetting it, allowing for easy geometry
t_gripper_tcp_offset = np.hstack(( T[:3,3]/1000,st.Rotation.from_matrix(T[:3,:3]).as_rotvec())) 
print("special pusht scripting offset:",list(t_gripper_tcp_offset))