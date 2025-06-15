# [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
#  [0.0, 0.7071067811865476, 0.7071067811865475, 0.0, 0.0, 0.0], 
#  [0.0, -0.7071067811865475, 0.7071067811865476, 0.0, 0.0, 0.0], 
#  [0.0, 0.16756598320827715, 0.15153298320827713, 1.0, 0.0, 0.0], 
#  [-0.2256370430227639, 0.0, 0.0, 0.0, 0.7071067811865476, 0.7071067811865475], 
#  [0.011337043022763914, 0.0, 0.0, 0.0, -0.7071067811865475, 0.7071067811865476]]

from diffusion_policy.real_world.base_real_env import BaseRealEnv
from diffusion_policy.real_world.realtime_suctioncup_controller import SuctioncupController
import numpy as np
import math
import time
from typing import Optional
from diffusion_policy.common.timestamp_accumulator import (
    TimestampObsAccumulator, 
    TimestampActionAccumulator
)
import scipy.spatial.transform as st

OBS_KEY_MAP = {
    # robot
    'ActualTCPPose': 'robot_eef_pose',
    'ActualTCPSpeed': 'robot_eef_pose_vel',
    'ActualQ': 'robot_joint',
    'ActualQd': 'robot_joint_vel',

    # suctioncup
    'suctioncup_state': 'suctioncup_state',

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
            shm_manager,
            tool_init_pos,
            max_obs_buffer_size=30,
            **kwargs):
        super().__init__(
            output_dir=output_dir, 
            robot_ip=robot_ip, 
            tool_address=tool_address,
            ft_sensor_ip=ft_sensor_ip, 
            shm_manager=shm_manager,
            max_obs_buffer_size=max_obs_buffer_size,
            obs_key_map=OBS_KEY_MAP,
            tcp_offset_pose=[-0.11998584527161234, -0.07205312949515096, 0.20981437944741316, -0.46507897471551723, -0.8215429239922134, 1.9837910837159003], #intentionally placed here, to prevent cnfig from modifying it
            init_joints=True,
            j_init = np.array([0,-90,-115,-110,90,59]) / 180 * np.pi, #intentionally placed here, to prevent cnfig from modifying it
            **kwargs)

        suctioncup_controller = SuctioncupController(
            shm_manager=shm_manager,
            ttyusb=tool_address,
            launch_timeout=5,
            suctioncup_init_pos = tool_init_pos,
            verbose=False,
            get_max_k=max_obs_buffer_size
        )
        self.suctioncup_controller = suctioncup_controller
        self.storage_pos_J = np.array([0,-75,-130,-110,90,59]) / 180 * np.pi

        
    # ========= async env API ===========
    def exec_actions(self, 
            actions: np.ndarray, 
            timestamps: np.ndarray, 
            stages: Optional[np.ndarray]=None,
            action_transform: Optional[st.Rotation]=st.Rotation.from_matrix(np.identity(3))): #This parameter transforms actions for execution but not for recording. Can be chosen conveniently to reduce action dimentionality when under-actuated. (rot. in ee frame)
        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)
        if stages is None:
            stages = np.zeros_like(timestamps, dtype=np.int64)
        elif not isinstance(stages, np.ndarray):
            stages = np.array(stages, dtype=np.int64)

        assert len(actions[-1]) == 7 # always (6dof + suctioncup) in this env.
        # convert action to pose
        self.receive_time = time.time()
        is_new = timestamps > self.receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]
        new_stages = stages[is_new]
        new_actions_copy = new_actions.copy()

        # schedule waypoints
        for i in range(len(new_actions)):
            this_action = new_actions_copy[i]
            suctioncup_action = this_action[-1]

            self.robot.schedule_waypoint(
                pose=np.hstack([this_action[:3],(st.Rotation.from_rotvec(this_action[3:6])*(action_transform.inv())).as_rotvec()]),
                target_time=new_timestamps[i]
            )

            self.suctioncup_controller.schedule_waypoint(
                pose=int(suctioncup_action),
                target_time=new_timestamps[i]
            )

        # record actions
        if self.action_accumulator is not None:
            #print("record:", new_actions)
            self.action_accumulator.put(
                new_actions,
                new_timestamps
            )

        if self.stage_accumulator is not None:
            self.stage_accumulator.put(
                new_stages,
                new_timestamps
            )

    # ========== recording API ==========
            

    # ======== start-stop API =============

    @property
    def is_ready(self):
        return self.realsense.is_ready and self.robot.is_ready and self.suctioncup_controller.is_ready and self.ft_sensor.is_ready

    def start(self, wait=True):
        self.suctioncup_controller.start(wait=False)
        super().start(wait=wait)
    
    def stop(self, wait=True):
        super().stop(wait = wait)
        self.suctioncup_controller.stop()
        if wait:
            self.stop_wait()
    
    def start_wait(self):
        self.suctioncup_controller.start_wait()
        super().start_wait()

    def stop_wait(self):
        self.suctioncup_controller.stop_wait()
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
        last_suctioncup_data = self.suctioncup_controller.get_all_state() # 30 Hz, gripper_receive_timestamp
        last_ft_data = self.ft_sensor.get_all_state() # 60 Hz, ft_sensor_receive_timestamp

        # align camera obs timestamps
        dt = 1 / self.frequency
        last_timestamp = np.max([x['timestamp'][-1] for x in self.last_realsense_data.values()]) #Last timestamp of camera observations

        #array of timestamps separated by dt, ending with last camera timestamp
        obs_align_timestamps = last_timestamp - (np.arange(self.n_obs_steps)[::-1] * dt)       

        camera_obs = self.align_camera_obs(obs_align_timestamps)
        robot_timestamps = last_robot_data['robot_receive_timestamp']
        robot_obs_raw, robot_obs = self.align_timestamps(robot_timestamps, last_robot_data, obs_align_timestamps)
        suctioncup_timestamps = last_suctioncup_data['suctioncup_receive_timestamp'] 
        suctioncup_obs_raw, suctioncup_obs = self.align_timestamps(suctioncup_timestamps, last_suctioncup_data, obs_align_timestamps)
        ft_timestamps = last_ft_data['ft_sensor_receive_timestamp'] 
        ft_obs_raw, ft_obs = self.align_timestamps(ft_timestamps, last_ft_data, obs_align_timestamps)
        
        #### Aligns the last frame of several raw obs for easy appending to obs accumulator 
        data_arr = [robot_obs_raw, suctioncup_obs_raw, ft_obs_raw]
        timestamps_arr = [robot_timestamps, suctioncup_timestamps, ft_timestamps]

        data_arr, timestamps_arr = self.align_for_obs_accumulator(data_arr, timestamps_arr)
        robot_obs_raw = data_arr[0]
        robot_timestamps = timestamps_arr[0]
        additional_data = data_arr[1:]
        additional_timestamps = timestamps_arr[1:]
        assert robot_timestamps[-1] == additional_timestamps[0][-1] == additional_timestamps[1][-1]

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
        obs_data.update(suctioncup_obs) 
        obs_data.update(ft_obs)
        obs_data['timestamp'] = obs_align_timestamps
        return obs_data
    
