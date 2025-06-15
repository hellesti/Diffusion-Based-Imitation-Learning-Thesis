
#abstract class: 

from typing import Optional
import pathlib
import numpy as np
import time
import shutil
import math
import random

import cv2
import scipy.spatial.transform as st
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.real_world.rtde_interpolation_controller import RTDEInterpolationController
#from diffusion_policy.real_world.multi_usbcam import MultiUSBCam, SingleUSBCam
from diffusion_policy.real_world.multi_realsense import MultiRealsense, SingleRealsense
from diffusion_policy.real_world.video_recorder import VideoRecorder
from diffusion_policy.common.timestamp_accumulator import (
    TimestampObsAccumulator, 
    TimestampActionAccumulator
)
from diffusion_policy.real_world.realtime_ft_sensor import FTSensor
from diffusion_policy.real_world.multi_camera_visualizer import MultiCameraVisualizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform, optimal_row_cols)


DEFAULT_OBS_KEY_MAP = {
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

class BaseRealEnv:
    def __init__(self, 
            # required params
            output_dir,
            robot_ip,
            tool_address,
            ft_sensor_ip,
            ft_transform_matrix,
            ft_sensor_frequency=60,
            # env params
            frequency=10,
            n_obs_steps=2,
            # obsfalse
            obs_image_resolution=(640,480),
            max_obs_buffer_size=30,
            camera_serial_numbers=None,
            obs_key_map=DEFAULT_OBS_KEY_MAP,
            obs_float32=False,
            # action
            max_pos_speed=0.25, #.33 #0.25
            max_rot_speed=0.6, #.8 #0.6
            # robot
            #tcp_offset_pose: SCARY -- ONLY CHANGE IF YOU KNOW WHAT YOU ARE DOING
            tcp_offset_pose=[0,0,0,0,0,0],#[0,0,(14.15 + 39.66 + 16.033)/1000, -0.7133046, 0.40382387, -0.97430994],#2.0983501, -1.18898251, -0.4925532],#[0,0,0,0,0,0],#[0,0,(14.15 + 39.66 + 16.033)/1000, -0.71439057,  0.40409574, -0.97545355],
            init_joints=True, 
            tool_init_pos=None,
            # video capture params
            video_capture_fps=30,
            video_capture_resolution=(1280,720),
            # saving params
            record_raw_video=True,
            thread_per_video=2,
            video_crf=21,
            # vis params
            enable_multi_cam_vis=True,
            multi_cam_vis_resolution=(1280,720),
            j_init = np.array([0,-90,-90-45,0,90,59-180+180]) / 180 * np.pi,
            # shared memory
            shm_manager=None
            ):
        assert frequency <= video_capture_fps
        output_dir = pathlib.Path(output_dir)
        print("DIR:")
        print(output_dir)
        assert output_dir.parent.is_dir()
        video_dir = output_dir.joinpath('videos')
        video_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())
        replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=zarr_path, mode='a')

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
        if camera_serial_numbers is None:
            camera_serial_numbers = SingleRealsense.get_connected_devices_serial()
            print(f"camera_serial_numbers: {camera_serial_numbers}")

        color_tf = get_image_transform(
            input_res=video_capture_resolution,
            output_res=obs_image_resolution, 
            # obs output rgb
            bgr_to_rgb=True)
        color_transform = color_tf
        if obs_float32:
            color_transform = lambda x: color_tf(x).astype(np.float32) / 255

        def transform(data):
            data['color'] = color_transform(data['color'])
            return data
        
        rw, rh, col, row = optimal_row_cols(
            n_cameras=len(camera_serial_numbers),
            in_wh_ratio=obs_image_resolution[0]/obs_image_resolution[1],
            max_resolution=multi_cam_vis_resolution
        )
        vis_color_transform = get_image_transform(
            input_res=video_capture_resolution,
            output_res=(rw,rh),
            bgr_to_rgb=False
        )
        def vis_transform(data):
            data['color'] = vis_color_transform(data['color'])
            return data

        recording_transfrom = None
        recording_fps = video_capture_fps
        recording_pix_fmt = 'bgr24'
        if not record_raw_video:
            recording_transfrom = transform
            recording_fps = frequency
            recording_pix_fmt = 'rgb24'

        video_recorder = VideoRecorder.create_h264(
            fps=recording_fps, 
            codec='h264',
            input_pix_fmt=recording_pix_fmt, 
            crf=video_crf,
            thread_type='FRAME',
            thread_count=thread_per_video)

        realsense = MultiRealsense(
            serial_numbers=camera_serial_numbers,
            shm_manager=shm_manager,
            resolution=video_capture_resolution,
            capture_fps=video_capture_fps,
            put_fps=video_capture_fps,
            # send every frame immediately after arrival
            # ignores put_fps
            put_downsample=False,
            record_fps=recording_fps,
            enable_color=True,
            enable_depth=False,
            enable_infrared=False,
            get_max_k=max_obs_buffer_size,
            transform=transform,
            vis_transform=vis_transform,
            recording_transform=recording_transfrom,
            video_recorder=video_recorder,
            verbose=False
            )
        
        multi_cam_vis = None
        if enable_multi_cam_vis:
            multi_cam_vis = MultiCameraVisualizer(
                realsense=realsense,
                row=row,
                col=col,
                rgb_to_bgr=False
            )

        cube_diag = np.linalg.norm([1,1,1])

        if not init_joints:
            raise Exception("init_joints is mandatory")
            j_init = None

        robot = RTDEInterpolationController(
            shm_manager=shm_manager,
            robot_ip=robot_ip,
            frequency=125, # UR5 CB3 RTDE
            lookahead_time=0.1,
            gain=300,
            max_pos_speed=max_pos_speed*cube_diag,
            max_rot_speed=max_rot_speed*cube_diag,
            launch_timeout=10,
            tcp_offset_pose=tcp_offset_pose,
            payload_mass = None,
            payload_cog=None,
            joints_init=j_init,
            joints_init_speed=1.05,
            soft_real_time=False,
            verbose=False,
            receive_keys=None,
            get_max_k=max_obs_buffer_size
            )

        ft_sensor = FTSensor(
            shm_manager=shm_manager,
            ft_sensor_ip=ft_sensor_ip,
            ft_transform_matrix=ft_transform_matrix,
            frequency=ft_sensor_frequency,
            launch_timeout=3,
            verbose=False,
            get_max_k=max_obs_buffer_size
        )

        self.realsense = realsense
        self.robot = robot
        self.ft_sensor = ft_sensor
        self.multi_cam_vis = multi_cam_vis
        self.video_capture_fps = video_capture_fps
        self.frequency = frequency
        self.n_obs_steps = n_obs_steps
        self.max_obs_buffer_size = max_obs_buffer_size
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.obs_key_map = obs_key_map
        # recording
        self.output_dir = output_dir
        self.video_dir = video_dir
        self.replay_buffer = replay_buffer
        # temp memory buffers
        self.last_realsense_data = None
        # recording buffers
        self.obs_accumulator = None
        self.action_accumulator = None
        self.stage_accumulator = None
        self.start_time = None
        self.receive_time = None
        self.storage_pos_J = np.array([0,-75,-155,50,90,-121]) / 180 * np.pi
        self.startpos_L = np.array([4.68600000e-01, -3.75149999e-01,  0.0019, -1.71654971e-03, -3.14087749e+00, -1.12881027e-02]) #0.0019
        self.j_init = j_init

    # =========== miscellaneous =========== 
    def set_ft_zero_point(self):
        self.ft_sensor.set_zero_point()

    # ===== Functions used in get_obs =====
    def align_timestamps(self, timestamps, last_data, align_to_timestamps):
        this_timestamps = timestamps
        this_idxs = list()
        for t in align_to_timestamps:
            is_before_idxs = np.nonzero(this_timestamps < t)[0]
            this_idx = 0
            if len(is_before_idxs) > 0:
                this_idx = is_before_idxs[-1]
            this_idxs.append(this_idx)

        #make obs_raw
        obs_raw = dict()
        for k, v in last_data.items():
            if k in self.obs_key_map:
                obs_raw[self.obs_key_map[k]] = v 
        
        #make robot_obs
        obs = dict()
        for k, v in obs_raw.items():
            obs[k] = v[this_idxs]

        return obs_raw, obs
   
    def align_camera_obs(self, obs_align_timestamps):
        camera_obs = dict()
        for camera_idx, value in self.last_realsense_data.items():
            this_timestamps = value['timestamp']
            this_idxs = list()
            for t in obs_align_timestamps:
                is_before_idxs = np.nonzero(this_timestamps < t)[0]
                this_idx = 0
                if len(is_before_idxs) > 0:
                    this_idx = is_before_idxs[-1] # Take last element in camera obs which is before 
                this_idxs.append(this_idx) 
            # remap key
            camera_obs[f'camera_{camera_idx}'] = value['color'][this_idxs]
        return camera_obs
    
    def align_for_obs_accumulator(self, data_arr, timestamps_arr): # aligns the last data point in each channel to have the same timestamp 
        last_col = []
        for i in range(len(timestamps_arr)): 
            last_col.append(timestamps_arr[i][-1])
        last_timestamp_i = np.argmin(last_col)
        last_timestamp = timestamps_arr[last_timestamp_i][-1]
        for i, elem in enumerate(data_arr):
            if i == last_timestamp_i:
                pass
            else:
                n_later_ts = np.count_nonzero(last_timestamp < timestamps_arr[i])
                n_remove = n_later_ts-1
                if n_remove:
                    timestamps_arr[i] = timestamps_arr[i][:-n_remove]
                    for key, val in elem.items(): 
                        elem[key] = val[:-n_remove] #sync data with timestamps
                if n_later_ts:
                    timestamps_arr[i][-1] = last_timestamp
        return data_arr, timestamps_arr
    
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

        # convert action to pose
        self.receive_time = time.time()
        is_new = timestamps > self.receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]
        new_stages = stages[is_new]
        new_actions_copy = new_actions.copy()

        # schedule waypoints
        for i in range(len(new_actions)):
            #print("target pose transformed back:",np.hstack([new_actions[i][:3],(st.Rotation.from_rotvec(new_actions[i][3:6])*(action_transform.inv())).as_rotvec()]))
            self.robot.schedule_waypoint(
                pose=np.hstack([new_actions_copy[i][:3],(st.Rotation.from_rotvec(new_actions_copy[i][3:6])*(action_transform.inv())).as_rotvec()]),
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

        
    def get_robot_state(self):
        return self.robot.get_state()

    # recording API
    def start_episode(self, start_time=None):
        "Start recording and return first obs"
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        assert self.is_ready

        # prepare recording stuff
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        this_video_dir.mkdir(parents=True, exist_ok=True)
        n_cameras = self.realsense.n_cameras
        video_paths = list()
        for i in range(n_cameras):
            video_paths.append(
                str(this_video_dir.joinpath(f'{i}.mp4').absolute()))
        
        # start recording on realsense
        self.realsense.restart_put(start_time=start_time)
        self.realsense.start_recording(video_path=video_paths, start_time=start_time)

        # create accumulators
        self.obs_accumulator = TimestampObsAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        self.stage_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )

    def end_episode(self):
        "Stop recording"
        assert self.is_ready
        
        # stop video recorder
        self.realsense.stop_recording()
        print("Stopped recording")
        if self.obs_accumulator is not None:
            # recording
            assert self.action_accumulator is not None
            assert self.stage_accumulator is not None

            # Since the only way to accumulate obs and action is by calling
            # get_obs and exec_actions, which will be in the same thread.
            # We don't need to worry new data come in here.
            obs_data = self.obs_accumulator.data
            obs_timestamps = self.obs_accumulator.timestamps #Returns a self.dt (10Hz)-spaced array of times
            action_timestamps = self.action_accumulator.timestamps
            obs_timestamps = self.obs_accumulator.timestamps #Returns a self.dt (10Hz)-spaced array of times
            
            n_steps = min(len(obs_timestamps), len(action_timestamps))
            actions = self.action_accumulator.actions
            stages = self.stage_accumulator.actions
            if n_steps > 0:
                episode = dict()
                episode['timestamp'] = obs_timestamps[:n_steps]
                episode['action'] = actions[:n_steps]
                episode['stage'] = stages[:n_steps]
                for key, value in obs_data.items():
                    episode[key] = value[:n_steps]
                self.replay_buffer.add_episode(episode, compressors='disk')
                episode_id = self.replay_buffer.n_episodes - 1
                print(f'Episode {episode_id} saved!')
            self.obs_accumulator = None
            self.action_accumulator = None
            self.stage_accumulator = None
        return True

    def drop_episode(self):
        self.end_episode()
        self.replay_buffer.drop_episode()
        episode_id = self.replay_buffer.n_episodes
        print(episode_id)
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        if this_video_dir.exists():
            shutil.rmtree(str(this_video_dir))
        print(f'Episode {episode_id} dropped!')

    # ======== start-stop API =============

    def start(self, wait=True):
        self.realsense.start(wait=False)
        self.robot.start(wait=False)
        self.ft_sensor.start(wait=False)

        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start(wait=False)
        if wait:
            self.start_wait()
        #self.goto_startpos()

    def stop(self, wait=True): 
        self.end_episode()
        self.goto_storage_pos()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop(wait=False)
        self.robot.stop(wait=False)
        self.ft_sensor.stop()
        self.realsense.stop(wait=False)

    def start_wait(self):
        self.realsense.start_wait()
        self.robot.start_wait()
        self.ft_sensor.start_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start_wait()
    
    def stop_wait(self):
        self.robot.stop_wait()
        self.realsense.stop_wait()
        self.ft_sensor.stop_wait()

        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop_wait()

    #scripting

    def go_up(self, distance):
        state = self.get_robot_state()
        current_pose = state['TargetTCPPose']
        desired_pose = current_pose + np.array([0,0,distance,0,0,0])
        assert self.robot.moveL(desired_pose)
        return True
    
    def goto_storage_pos(self):
        assert self.robot.moveJ(self.j_init)
        assert self.robot.moveJ(self.storage_pos_J)
        return True