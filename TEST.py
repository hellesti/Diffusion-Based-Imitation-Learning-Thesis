import numpy as np
import NetFT
from diffusion_policy.common.precise_sleep import precise_wait
import time

from diffusion_policy.common.replay_buffer import ReplayBuffer

dir = "data/demo_rectangle_peg_vertical/replay_buffer.zarr"
rb = ReplayBuffer.copy_from_path(zarr_path=dir)

#dir2 = "data/outputs/2024.05.16/14.46.28_train_diffusion_unet_image_real_pusht_image_ft/checkpoints/eval_latest/replay_buffer.zarr"
#rb2 = ReplayBuffer.copy_from_path(zarr_path=dir2)

rb
#rb2




# # # target_pose = np.array([1.0,2.0,3.0,4.0,5.0,6.0])


# # # x_lim = [0.377, 1.145]
# # # y_lim = [-0.40, 0.420]
# # # z_lim = [0.002, 1.000]

# # # xr_lim = [0,0]
# # # yr_lim = [0,0]
# # # zr_lim = [-np.pi, np.pi]

# # # target_pose = np.clip(target_pose, [x_lim[0], y_lim[0], z_lim[0], xr_lim[0], yr_lim[0], zr_lim[0]], [x_lim[1], y_lim[1], z_lim[1], xr_lim[1], yr_lim[1], z_lim[1]])

# # # print(target_pose)


# # # import NetFT
# # # ft = NetFT.Sensor("192.168.1.13")
# # # a = ft.getMeasurement()
# # # print(a)

# # # ft = NetFT.Sensor("192.168.1.13")
# # # t_start = time.monotonic()
# # # iter_idx = 0
# # # dt = 1/60
# # # while True:
# # #     t_next_loop = (t_start + (iter_idx+1)*dt)
# # #     meas = ft.getMeasurement()
# # #     print(meas)
# # #     precise_wait(t_next_loop) q

# import modern_robotics as mr
# import numpy as np

# def c(x):
#     return np.cos(x)
# def s(x):
#     return np.sin(x)
# def deg_to_rad(x):
#     return x*np.pi/180

# def rotX(th):
#     rot_x = np.array([[1,0,0],
#                       [0,c(th),-s(th)],
#                       [0,s(th), c(th)]])
#     return rot_x

# def rotY(th):
#     rot_y = np.array([[c(th), 0, s(th)],
#                       [0,1,0],
#                       [-s(th),0, c(th)]])
#     return rot_y

# def rotZ(th):
#     rot_z = np.array([[c(th),-s(th),0],
#                       [s(th),c(th),0],
#                       [0,0,1]])
#     return rot_z





# theta1 = deg_to_rad(32)  #deg #z
# p1 = np.array([0,0,14.15 + 39.66 + 16.033]) #MILLIMETERS
# R1 = rotZ(theta1)
# T1 = mr.RpToTrans(R1, p1)



# theta2 = deg_to_rad(45) #x
# p2 = np.array([0,0,0]) #MILLIMETERS
# R2 = rotX(theta2)
# T2 = mr.RpToTrans(R2, p2)

# p3 = np.array([0,0,37.5])
# R3 = np.identity(3) #no rotation
# T3 = mr.RpToTrans(R3, p3)



# theta4 = deg_to_rad(-45) #
# p4 = np.array([0,0,0]) #MILLIMETERS
# R4 = rotX(theta4)
# T4 = mr.RpToTrans(R4, p4)

# p5 = np.array([0,0,37.5])
# R5 = np.identity(3) #no rotation
# T5 = mr.RpToTrans(R5, p5)


# ee1 = T1@T2@T3
# print(mr.se3ToVec(ee1))
# print()
# ee2 = T1@T4@T5
# print(mr.se3ToVec(ee2))


# import scipy.spatial.transform as st

# R_0a = st.Rotation.from_rotvec(np.array([-0.6279996, 2.28650355, -0.25471646])).as_matrix()   #Base ee offset, gripper vertical
# R_0b = st.Rotation.from_rotvec(np.array([-2.21776891,  2.21941352,  0.00935483])).as_matrix()  #Base ee offset, tool flange vertical
# R_0c = st.Rotation.from_rotvec(np.array([ 2.63474894,  0.72414768,  1.09210849])).as_matrix()  #suctioncup rot

# R_a0 = R_0a.T
# R_c0 = R_0c.T

# R_ab = R_a0@R_0b
# R_cb = R_c0@R_0b

# print(st.Rotation.from_matrix(R_ab).as_rotvec())
# print(st.Rotation.from_matrix(R_cb).as_rotvec())


# frame = R_ab@(st.Rotation.from_euler('x', -np.pi).as_matrix())
# frame2 = R_cb@(st.Rotation.from_euler('x', -np.pi).as_matrix())
# print(st.Rotation.from_matrix(frame).as_rotvec())
# print(st.Rotation.from_matrix(frame2).as_rotvec())
# #print(R_ab)

# print("\neulertest:\n")
# initial_euler_rot =[-3.14096641, -0.00701439, 1.56957194]
# target_pose = np.hstack([[0,0,0],st.Rotation.from_euler("xyz",initial_euler_rot).as_rotvec()])
# # in euler representation:
# # we assume that the end-effector frame is initially rotated one half-revolution around x (direction is irrelevant, same result regardless).
# # we also assume that the rotation around y is small.
# # We validate our assumptions and (post) rotate around x to obtain a frame whose x-axis is aligned with that of the end-effector frame. 
# assert np.abs(initial_euler_rot[0])-np.pi <0.01
# assert np.abs(initial_euler_rot[1]) < 0.01
# action_transform = st.Rotation.from_euler("yx", [-initial_euler_rot[1],-initial_euler_rot[0]])
# rotated_ee_frame = (st.Rotation.from_rotvec(target_pose[3:6]))*action_transform
# print(rotated_ee_frame.as_rotvec())
# print(rotated_ee_frame.as_euler("xyz"))

# assert rotated_ee_frame.as_rotvec()[0] < 0.01
# assert rotated_ee_frame.as_rotvec()[1] < 0.01
# # This new frame should (granted correct assumptions) be close to a pure z-rotation, which can be represented as a rotation vector [0,0,theta]. 
#     # We use this as the action representation when working with only z-rotations. It allows us to compress the rotation information from three channels to one channel while using rotation vectors as the rotation representation

# print("array misalignment")
# pass
# a1 = [1.71377399e+09, 1.71377399e+09, 2]
# a2 = [1.71377399e+09, 1.71377399e+09, 1.71377399e+09, 1.71377399e+09,
#     1.71377399e+09, 1.71377399e+09, 1.71377399e+09, 1.71377399e+09,
#     1.71377399e+09, 1.71377399e+09, 1.71377399e+09, 1.71377399e+09,
#     1.71377399e+09, 1.71377399e+09, 1.71377399e+09, 1.71377399e+09,
#     1.71377399e+09, 1.71377399e+09, 1.71377399e+09, 1.71377399e+09,
#     1.71377399e+09, 1.71377399e+09, 1.71377399e+09, 1.71377399e+09,
#     1.71377399e+09, 1.71377399e+09, 1.71377399e+09, 1.71377399e+09,
#     1.71377399e+09, 1]
# a = [a1,a2]
# #print(a)
# test = []
# for i in range(len(a)):
#     test.append(a[i][-1])
# last_timestamp_i = np.argmin(test)
# print(last_timestamp_i)

import omegaconf
config_path = "diffusion_policy/config/control/pusht10.yaml"
config_path_2 = "diffusion_policy/config/control/suctioncup_3dof.yaml"
cfg = omegaconf.OmegaConf.load(config_path)
cfg2 = omegaconf.OmegaConf.load(config_path_2)

term_dist = cfg.eval_termination_condition.dist
term_time = cfg.eval_termination_condition.time
term_pos = cfg.eval_termination_condition.pos
print(term_pos)
print(type(term_pos))
print(isinstance(term_pos, list))
print(not (None in [term_dist, term_time]))
print([term_pos, term_dist, term_time])
if (not None in [term_pos, term_dist, term_time]):
    print("hello")

term_dist2 = cfg2.eval_termination_condition.dist
term_time2 = cfg2.eval_termination_condition.time
term_pos2 = cfg2.eval_termination_condition.pos
print(not (term_pos2 is None))
print(not (None in [term_dist2, term_time2]))
print([term_pos2, term_dist2, term_time2])
if (not None in [term_pos2, term_dist2, term_time2]):
    print("hello")

pass
# from diffusion_policy.real_world.keystroke_counter import (
#     KeystrokeCounter, Key, KeyCode
# )   

# keycodes = list(cfg.keybidiffusion_policynds.keys())
# for i, a in enumerate(keycodes):
#     keycodes[i] = KeyCode(char=a)


# print(None in [0, 1, 0])

# if not (None in [None, 1, 0]):
#     print("not None")

# pass
# print(chr(-1))

shape = (2,)
if len(shape) == 1:
    shape = (shape[0],1)

shape