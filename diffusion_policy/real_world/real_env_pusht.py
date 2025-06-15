from diffusion_policy.real_world.base_real_env import BaseRealEnv
from diffusion_policy.real_world.gripper_api.gripper_modbus_tcp import Gripper
import numpy as np
import time
import math
import random
from typing import Optional
from diffusion_policy.common.timestamp_accumulator import (
    TimestampObsAccumulator, 
    TimestampActionAccumulator
)
import scipy.spatial.transform as st
import cv2

OBS_KEY_MAP = {
    # robot
    'ActualTCPPose': 'robot_eef_pose',
    'ActualTCPSpeed': 'robot_eef_pose_vel',
    'ActualQ': 'robot_joint',
    'ActualQd': 'robot_joint_vel',

    # ft sensor
    'ft_ee_wrench': 'ft_ee_wrench',

    # timestamps
    'step_idx': 'step_idx',
    'timestamp': 'timestamp'
}

class RealEnv(BaseRealEnv):

    def __init__(self, 
            # required params
            output_dir,
            robot_ip,
            tool_address,
            ft_sensor_ip,
            ft_transform_matrix,
            shm_manager,
            tool_init_pos,
            max_obs_buffer_size=30,
            **kwargs):
        super().__init__(
            output_dir=output_dir, 
            robot_ip=robot_ip, 
            tool_address=tool_address,
            ft_sensor_ip=ft_sensor_ip, 
            ft_transform_matrix = ft_transform_matrix,
            shm_manager=shm_manager,
            max_obs_buffer_size=max_obs_buffer_size,
            obs_key_map=OBS_KEY_MAP,
            #TCP offset pose for push-T is aligned with the gripper due to geometry convenience when resetting the T-block. This is fine, but means that models trained on pushT are not by-default compatible with models trained in stick env.
            tcp_offset_pose=[0.09397123355467275, 0.10192506785823813, 0.20287813598409524, -0.7133045977295701, 0.40382386980572454, -0.9743099431676225], #intentionally placed here, to prevent cnfig from modifying it
            init_joints=True,
            j_init = np.array([0,-90,-90,-135,90,59]) / 180 * np.pi, #intentionally placed here, to prevent cnfig from modifying it
            **kwargs)
        
        
        gripper = Gripper(gripper_ip=tool_address)
        gripper.activate()
        time.sleep(1)
        gripper.gotobool(False)

        #J-positions[joint space] may be stolen from here and used elsewhere but do not take L-positions[end-effector positions], because the end-effector position may change
        self.gripper = gripper
        self.storage_pos_J = np.array([0,-75,-130,-110,90,59]) / 180 * np.pi
        self.startpos_L = [0.28259833867399126, -0.3362941536022525, 0.2427819371223478, 1.2133864164084023, -1.2062432765955133, -1.2150160074254073]
        self.gripper_vertical_J = np.array([0,-90,-90,-135,90,59-180]) / 180 * np.pi
        self.above_pos = np.array([0.814+.039,0,0.076])
        self.centered_t_pos = np.array([0.794,0,0.030])
        self.photo_pos_J = np.array([8.48,-111.74,-31.27,-69.73,92.34,-112.68]) / 180 * np.pi
        self.T_startpos = None
        
    # ========= async env API ===========
    def exec_actions(self, 
        actions: np.ndarray, 
        timestamps: np.ndarray, 
        tool_actions = None,
        stages: Optional[np.ndarray]=None,
        action_transform: Optional[st.Rotation]=st.Rotation.from_matrix(np.identity(3))): #This parameter transforms actions for execution but not for recording. Can be chosen conveniently to reduce action dimentionality when under-actuated. (rot. in ee frame)
        super().exec_actions(actions=actions,
                             timestamps=timestamps,
                             stages=stages,
                             action_transform=action_transform)
    # ========== recording API ==========
            
    def start_episode(self, start_time=None):
        episode_id = self.replay_buffer.n_episodes
        super().start_episode(start_time=start_time)
        print(f'Episode {episode_id} started!')
    
    def end_episode(self):
        if self.obs_accumulator is not None: 
            super().end_episode()
            self.log_T_values()
            self.take_eval_photo()
            assert self.robot.moveJ(self.j_init, vel=2.0)
            self.goto_startpos()

        return True

    # ======== start-stop API =============

    @property
    def is_ready(self):
        return self.realsense.is_ready and self.robot.is_ready and self.ft_sensor.is_ready

    def start(self, wait=True):
        super().start(wait=wait)
    
    def stop(self, wait=True):
        super().stop(wait = wait)
        self.gripper.gotobool(False)
        self.gripper.close()
        if wait:
            self.stop_wait()
    
    def start_wait(self):
        super().start_wait()

    def stop_wait(self):
        super().stop_wait()
    # ========= context manager ===========
            
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()        

    def get_obs(self) -> dict:
        "observation dict"
        assert self.is_ready

        # get data
        # 30 Hz, camera_receive_timestamp
        k = math.ceil(self.n_obs_steps * (self.video_capture_fps / self.frequency))
        self.last_realsense_data = self.realsense.get(
            k=k, 
            out=self.last_realsense_data)

        last_robot_data = self.robot.get_all_state() # 125 hz, robot_receive_timestamp
        last_ft_data = self.ft_sensor.get_all_state() # 60 Hz, ft_sensor_receive_timestamp
        # align camera obs timestamps
        dt = 1 / self.frequency
        last_timestamp = np.max([x['timestamp'][-1] for x in self.last_realsense_data.values()]) #Last timestamp of camera observations

        #array of timestamps separated by dt, ending with last camera timestamp
        obs_align_timestamps = last_timestamp - (np.arange(self.n_obs_steps)[::-1] * dt)       

        camera_obs = self.align_camera_obs(obs_align_timestamps)
        robot_timestamps = last_robot_data['robot_receive_timestamp']
        robot_obs_raw, robot_obs = self.align_timestamps(robot_timestamps, last_robot_data, obs_align_timestamps)
        ft_timestamps = last_ft_data['ft_sensor_receive_timestamp'] 
        ft_obs_raw, ft_obs = self.align_timestamps(ft_timestamps, last_ft_data, obs_align_timestamps)
        
        #### Aligns the last frame of several raw obs for easy appending to obs accumulator 
        data_arr = [robot_obs_raw, ft_obs_raw]
        timestamps_arr = [robot_timestamps, ft_timestamps]

        data_arr_new, timestamps_arr_new = self.align_for_obs_accumulator(data_arr, timestamps_arr)
        robot_obs_raw = data_arr_new[0]
        robot_timestamps = timestamps_arr_new[0]
        additional_data = data_arr_new[1:]
        additional_timestamps = timestamps_arr_new[1:]
        assert robot_timestamps[-1] == additional_timestamps[0][-1]

        if self.obs_accumulator is not None:
            self.obs_accumulator.put(
                robot_obs_raw,
                robot_timestamps,
                additional_data=additional_data,
                additional_timestamps=additional_timestamps
            )

        # return obs
        obs_data = dict(camera_obs)
        obs_data.update(robot_obs)
        obs_data.update(ft_obs)
        obs_data['timestamp'] = obs_align_timestamps
        return obs_data
    
    def take_eval_photo(self):
        photo_pos = self.photo_pos_J
        assert self.robot.moveJ(photo_pos, vel=2)

        #eecam.set_manual_focus(0) #TODO put in real value

        time.sleep(0.5) #TODO necessary? empty buffer instead? maybe wait for focus completion mp.event

        eval_img = self.realsense.get()[0]["color"] 
        m=eval_img.max()
        if (m -1.0 < 0.05):
            eval_img *=250
        #eecam.set_manual_focus(0) #TODO put real value
        eval_img = cv2.cvtColor(eval_img, cv2.COLOR_RGB2BGR)

        episode_id = self.replay_buffer.n_episodes-1

        this_video_dir = self.video_dir.joinpath(str(episode_id))
        eval_img_path = str(this_video_dir.joinpath("eval.png"))
        cv2.imwrite(eval_img_path, eval_img) 
        
        return True

    def do_T_reset(self, seed_ep_num = False): 
        assert self.robot.moveJ(self.gripper_vertical_J, vel=3)
        #above = np.add(self.clasping_T_pos_L, np.array([0,0,0.100,0,0,0]))t
        #move above T

        state = self.get_robot_state()
        pose = state['TargetTCPPose']
        rot = (st.Rotation.from_euler('z', np.pi/2) * st.Rotation.from_rotvec(pose[3:6])).as_rotvec()
        assert self.robot.moveL(np.hstack([self.above_pos,rot]))
        clasping_t_pos = self.above_pos.copy()
        clasping_t_pos[2] = 0.030
        #lower around T
        assert self.robot.moveL(np.hstack([clasping_t_pos, rot]))
        #drag T towards robot base and close gripper
        dragged_t_pos = clasping_t_pos.copy()
        dragged_t_pos[0] -= 0.100
        assert self.robot.moveL(np.hstack([dragged_t_pos, rot]), vel = .2, accel = .1)
        #CLOSE GRIPPER
        assert self.gripper.gotobool(True, wait_timeout=3)
        #assert self.gripper.wait_for_grab_success(timeout=2)
        time.sleep(1)
        centered_t_pos = self.centered_t_pos
        #
        state = self.get_robot_state()
        pose = state['TargetTCPPose']
        episode_id = self.replay_buffer.n_episodes
        if seed_ep_num: 
            np.random.seed(episode_id)
        else:
            np.random.seed(int(time.time()))
        #assert self.robot.moveL(np.hstack([centered_t_pos, rot]), accel=.5)
        scripted_theta = np.random.uniform(low=-np.pi*3/2, high=np.pi/2)
        
        invalid_solution = True
        while invalid_solution:
            centre_dx = np.random.uniform(-0.23, 0.13)
            centre_dy = np.random.uniform(-0.23, 0.23)
            if ((-.5<centre_dx<.5) and (-.15<centre_dy<.15)):
                invalid_solution = True
            else:
                invalid_solution = False

        assert (0.23>=centre_dx>=-0.23)
        assert (0.23>=centre_dy>=-0.23) 

        scripted_pos = np.array([centre_dx, centre_dy, 0.01])
        self.T_startpos = [centre_dx, centre_dy, scripted_theta]

        if np.abs(scripted_theta)>0.9*np.pi:
            #add waypoint in the middle
            halfrot = scripted_theta/2
            halfpos = scripted_pos/2
            halfrot_vec = (st.Rotation.from_euler('z', halfrot) * st.Rotation.from_rotvec(pose[3:6])).as_rotvec()
            assert self.robot.moveL(np.hstack([centered_t_pos+halfpos, halfrot_vec]), accel=0.5)

        theta_vec = (st.Rotation.from_euler('z', scripted_theta) * st.Rotation.from_rotvec(pose[3:6])).as_rotvec()
        assert self.robot.moveL(np.hstack([centered_t_pos+scripted_pos, theta_vec]), accel=0.5)
        
        state = self.get_robot_state()
        pose = state['TargetTCPPose']
        down = pose.copy()
        down[2] = 0.03
        assert self.robot.moveL(down, vel=0.1, accel=0.1)
        assert self.gripper.gotobool(False, wait_timeout=3)

        state = self.get_robot_state()
        pose = state['TargetTCPPose']
        up = pose.copy()
        up[2] = 0.100
        assert self.robot.moveL(up, vel=.5, accel=.5)
        if (scripted_theta < -3*np.pi/4): #This block is to stop cable from getting snagged
            unrot_theta = -3*np.pi/4 -scripted_theta
            assert np.abs(unrot_theta) < np.pi
            unrot = (st.Rotation.from_euler('z', unrot_theta) * st.Rotation.from_rotvec(up[3:6])).as_rotvec()
            unrot_pose= np.hstack([up[:3],unrot])
            assert self.robot.moveL(unrot_pose,accel=1)
        assert self.robot.moveJ(self.j_init, vel=2.0)

        return 
    
    def goto_startpos(self, vel=1.05, accel=1.4):
        print("Start pos used: ", self.startpos_L)
        assert self.robot.moveL(self.startpos_L, vel=vel, accel=accel)
        return True

    def log_T_values(self):
        state = self.get_robot_state()
        endpos = state["ActualTCPPose"]
        T_startpos = self.T_startpos
        assert T_startpos is not None
        strings0 = [(str(x)+"\n") for x in endpos]
        strings1 = [(str(x)+"\n") for x in T_startpos]

        episode_id = self.replay_buffer.n_episodes-1
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        eval_log_path = str(this_video_dir.joinpath("log.txt"))
        
        with open(eval_log_path, "w") as file:
            file.writelines(strings0)
            file.writelines(strings1)