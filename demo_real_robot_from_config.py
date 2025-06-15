"""
Usage:
(robodiff)$ python demo_real_robot.py -o <demo_save_dir> --robot_ip <ip_of_ur5>

Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start recording.
Press "S" to stop recording.
Press "Q" to exit program.
Press "Backspace" to delete the previously recorded episode.
"""

# %%
import omegaconf
import time
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.common.env_util import propagate_random_walk
import click
from click.core import ParameterSource
import cv2
import numpy as np
import scipy.spatial.transform as st
from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)   
from diffusion_policy.common.rot_limit_util import TCPLimiter, pose_is_within_limits, rot_dist #special decomposition which is well-behaved for limiting rotation
from diffusion_policy.real_world.real_env_gripper import RealEnv as RealEnvGripper
from diffusion_policy.real_world.real_env_suctioncup import RealEnv as RealEnvSuctioncup 
#from diffusion_policy.real_world.real_env_stick import RealEnv as RealEnvStick 
from diffusion_policy.real_world.real_env_pusht import RealEnv as RealEnvPushT


@click.command()
@click.option('--config', '-c', required=True, help="Name of config file (in diffusion_policy/config/control)")

#Options override the config
@click.option('--output', '-o', default = None, help="Directory to save demonstration dataset.")
@click.option('--robot_ip', '-ri', default = None, help="UR5's IP address e.g. 192.168.1.10")
@click.option('--tool_address', '-gi', default = None, help="tool's IP address e.g. 192.168.1.12")
@click.option('--ft_sensor_ip', '-fi', default =None, help="ft sensor's IP address e.g. 192.168.1.13")
@click.option('--vis_camera_idx', default=None, type=int, help="Which RealSense camera to visualize.")
@click.option('--tool_init_pos', '-ti', default=None, help="Tool init pose. True: closed/active, False: open/inactive")
@click.option('--frequency', '-f', default=None, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=None, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.") #changed from .01 to .025

def main(config, output, robot_ip, tool_address, ft_sensor_ip, vis_camera_idx, tool_init_pos, frequency, command_latency):

    # Check if any click options are defaults 
    # Override default options with config options
    # Load config vals for launch options
    
    config_path = "diffusion_policy/config/control/"+config 

    control_cfg = omegaconf.OmegaConf.load(config_path)

    launch_options =   {"output":output, "robot_ip":robot_ip, "tool_address":tool_address, "ft_sensor_ip":ft_sensor_ip, "vis_camera_idx":vis_camera_idx, "tool_init_pos":tool_init_pos, "frequency":frequency, "command_latency":command_latency}
    
        
    click_context= click.get_current_context()
    for key, val in launch_options.items():
        if click_context.get_parameter_source(key) == ParameterSource.DEFAULT: #if option is default value, use config
            launch_options[key] = control_cfg.demo_launch_options[key]
        #else: keep value which has been input by user; do nothing

    output = launch_options["output"]
    robot_ip = launch_options["robot_ip"]
    tool_address = launch_options["tool_address"]
    ft_sensor_ip = launch_options["ft_sensor_ip"]
    vis_camera_idx = launch_options["vis_camera_idx"]
    tool_init_pos = launch_options["tool_init_pos"]
    frequency = launch_options["frequency"]
    command_latency = launch_options["command_latency"]

    if control_cfg.real_env == "gripper":
        RealEnv = RealEnvGripper
    elif control_cfg.real_env == "suctioncup":
        RealEnv = RealEnvSuctioncup
    elif control_cfg.real_env == "stick":
        raise NotImplementedError
        RealEnv = RealEnvStick
    elif control_cfg.real_env == "pusht":
        RealEnv = RealEnvPushT

    ft_transform_matrix = control_cfg.ft_transform_matrix

    output = "data/"+output
    
    dt = 1/frequency

    # DISTURBANCE PARAMS #
    # Easy Mode: bound: 0.00005, scale_factor: 0.0005
    # Medium Mode: bound: 000275 , scale_factor: 00275
    # Hard Mode: bound: 0.0005, scale_factor: 0.005
    disturbance_active = False
    disturbance_pos = np.array([0.0,0.0])
    disturbance_vel = np.array([0.0,0.0])
    disturbance_bounds = np.array([[
        -0.0005, 0.0005
    ],
    [
        -0.0005, 0.0005
    ]])
    smoothness = 1.0
    bound_strength=100.0
    scale_factors = np.array([0.005,0.005])
    propagate_disturbance = lambda pos,vel: propagate_random_walk(pos, vel, dt, disturbance_bounds, smoothness, scale_factors, bound_strength=bound_strength)

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            Spacemouse(shm_manager=shm_manager, deadzone=[0.000,0.000,0.000,0.000,0.000,0.000]) as sm, \
            RealEnv( #0.040 deadzone previously used
                output_dir=output, 
                robot_ip=robot_ip, 
                tool_address=tool_address, #ip address or usb port, depending on tool 
                ft_sensor_ip=ft_sensor_ip,
                ft_transform_matrix=ft_transform_matrix,
                # recording resolution
                obs_image_resolution=(1280,720),
                frequency=frequency,
                enable_multi_cam_vis=True,
                record_raw_video=True,
                tool_init_pos=tool_init_pos,
                # number of threads per camera view for video recording (H.264)
                thread_per_video=3,
                # video recording quality, lower is better (but slower).
                video_crf=21,
                shm_manager=shm_manager,
            ) as env:
            cv2.setNumThreads(1)


            # realsense exposure
            env.realsense.set_exposure(exposure=80, gain=0)
            # realsense white balance
            env.realsense.set_white_balance(white_balance=5900)
            
            if control_cfg.real_env == "gripper":
                env.goto_startpos()

            time.sleep(1.0)
            print('Ready!')
            
            env.set_ft_zero_point()
            state = env.get_robot_state()
            target_pose = state['TargetTCPPose']
            target_tool_pose = tool_init_pos
            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False
            initial_pose = target_pose
            R_bd = (st.Rotation.from_rotvec(target_pose[3:6])).as_matrix()
            tcp_limiter = TCPLimiter(control_cfg.tcp_limits, R_bd)
            initial_euler_rot = (st.Rotation.from_rotvec(target_pose[3:6])).as_euler('xyz')
            action_transform = st.Rotation.from_matrix(np.identity(3))
            skip_this_cycle = False
            

            if control_cfg.use_rot == "z":
                # in euler representation:
                # we assume that the end-effector frame is initially rotated one half-revolution around x (direction is irrelevant, same result regardless).
                # we also assume that the rotation around y is small.
                # We validate our assumptions and (post) rotate around x to obtain a frame whose x-axis is aligned with that of the end-effector frame. 
                assert np.abs(initial_euler_rot[0])-np.pi <0.01
                assert np.abs(initial_euler_rot[1]) < 0.01
                action_transform = st.Rotation.from_euler("yx", [-initial_euler_rot[1],-initial_euler_rot[0]]) #Ideal scenario for euler angles: x close to +-pi and y small
                rotated_ee_frame = (st.Rotation.from_rotvec(target_pose[3:6]))*action_transform

                assert rotated_ee_frame.as_rotvec()[0] < 1e-10
                assert rotated_ee_frame.as_rotvec()[1] < 1e-10
                # This new frame should (granted correct assumptions) be close to a pure z-rotation, which can be represented as a rotation vector [0,0,theta]. 
                # We use this as the action representation when working with only z-rotations. It allows us to compress the rotation information from three channels to one channel while using rotation vectors as the rotation representation
                pass
                

            state = env.get_robot_state()
            temp_pose = state['TargetTCPPose']
            #print(temp_pose)
            assert pose_is_within_limits(temp_pose, control_cfg.tcp_limits, R_bd)               
            
            while not stop:
                # calculate timing
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                # pump obs
                obs = env.get_obs()

                # handle key presses
                keycodes = list(control_cfg.keybinds.keys())
                for i, a in enumerate(keycodes):
                    keycodes[i] = KeyCode(char=a)

                press_events = key_counter.get_press_events()

                commands = []
                for key_stroke in press_events: #TODO make it so some or all of these can all be set in the config
                    if key_stroke == Key.backspace:
                        # Delete the most recent recorded episode
                        if click.confirm('Are you sure to drop an episode?'):
                            env.drop_episode()
                            key_counter.clear()
                            is_recording = False

                    elif key_stroke in keycodes:
                        command=control_cfg.keybinds[key_stroke.char] #access dict value of bind
                        commands.append(command)
                        print(command)
                    
                stage = key_counter[Key.space]

                # visualize
                vis_img = obs[f'camera_{vis_camera_idx}'][-1,:,:,::-1].copy()
                episode_id = env.replay_buffer.n_episodes
                text = f'Episode: {episode_id}, Stage: {stage}'
                if is_recording:
                    text += ', Recording!'
                cv2.putText(
                    vis_img,
                    text,
                    (10,30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    thickness=2,
                    color=(255,255,255)
                )

                cv2.imshow('default', vis_img)
                cv2.pollKey()

                precise_wait(t_sample)
                # get teleop command
                sm_state = sm.get_button_state()
                if sm.is_button_pressed(0):
                    commands.append(control_cfg.sm_keybinds.button_0)
                elif sm.is_button_pressed(1):
                    commands.append(control_cfg.sm_keybinds.button_1)
                
                if len(commands) > 0:
                    for command in commands:
                        #if self.env.gripper_controller.

                        if command == "stop":
                            # Exit program
                            stop = True

                        elif command == "start_recording":
                            if not is_recording:
                                # Start recording
                                env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
                                key_counter.clear()
                                is_recording = True
                                print('Recording!')
                            else:
                                print("Invalid input. Already recording.")

                        elif command == "stop_recording":
                            if is_recording:
                                # Stop recording
                                env.end_episode() #wait until done. Then, make sure next servo instruction is reasonable
                                #time.sleep(2)
                                key_counter.clear()
                                is_recording = False
                                state = env.get_robot_state()
                                target_pose = state['TargetTCPPose']
                                #print(target_pose)
                                print('Stopped.')
                                skip_this_cycle = True
                            else:
                                print("Invalid input. Not recording.")
                        
                        elif command == "toggle_disturbance":
                            if not disturbance_active:
                                disturbance_active = True
                                print("disturbance activated")
                            else:
                                disturbance_active = False
                                print("disturbance deactivated")

                        elif command == "do_t_reset":
                            if not is_recording:
                                assert control_cfg.real_env == "pusht"
                                env.do_T_reset()
                                env.goto_startpos()
                                state = env.get_robot_state()
                                temp_pose = state['TargetTCPPose']
                                assert pose_is_within_limits(temp_pose, control_cfg.tcp_limits, R_bd)
                                key_counter.clear()
                                target_pose = state['TargetTCPPose'] #If endpos differs from startpos, this resets servo target pos to movement endpos
                                skip_this_cycle = True

                            else:
                                print("Invalid input. Stop recording before doing T-reset.")
                        elif command == "activate_tool":
                            target_tool_pose = True
                        elif command == "deactivate_tool":
                            target_tool_pose = False
                        elif command == "print_ee_pose":    
                            state = env.get_robot_state()
                            target_pose = state['TargetTCPPose']
                            print(f"pos: {target_pose.tolist()}")
                        elif command == "print_ee_joints":
                            state = env.get_robot_state()
                            target_joints = state['TargetQ']
                            print(f"joints: {target_joints.tolist()}")

                if not skip_this_cycle:

                    sm_state = sm.get_motion_state_transformed() #get sm state again
                    #print(sm_state)
                    dpos = sm_state[:3] * (env.max_pos_speed / frequency) #reduce to 2dof for push-t
                    # Add disturbance:
                    if disturbance_active:
                        disturbance_pos, disturbance_vel,_ = propagate_disturbance(disturbance_pos, disturbance_vel)
                        dpos[:2] += disturbance_pos

                    drot_xyz = sm_state[3:6] * (env.max_rot_speed / frequency) #remove for push-t
                    
                    ##remove channels according to config

                    if control_cfg.real_env == "pusht":
                        drot_xyz = [0,0,0]
                        dpos[2] = 0
                        assert control_cfg.use_rot == None
                    elif control_cfg.use_rot == None:
                        drot_xyz = [0,0,0]
                    elif control_cfg.use_rot == "z":
                        drot_xyz[:2] = [0,0]
                    elif control_cfg.use_rot == "xyz":
                        pass
                    
                
                    drot = st.Rotation.from_euler('xyz', drot_xyz)
                    target_pose[:3] += dpos
                    target_pose[3:6] = (drot * st.Rotation.from_rotvec(
                        target_pose[3:6])).as_rotvec()
                    
                    ### BEGIN POS, ROT LIMITATION ### #TODO package this into a convenient function?
                    
                    target_pose = tcp_limiter.get_limited_pose(target_pose)
                    
                    ### END POS, ROT LIMITATION ###

                    #all ee joint limit checks are done prior to this. Now, given that we are working with pure z rot, we:
                        # 0. rotate by ee action transform
                        # 1. adjust rotation vector to be on the format [0,0,theta] by removing small offset in x and y dimensions
                        # 2. record action on the format [0,0,theta] (this happens in env.exec_actions)
                        # 3. reverse ee action transform rotation (this happens in env.exec_actions)
                        # 4. perform action
                        # warning: Step 3 will not yield the exact pre-transform orientation due to step 1 
                        #when running eval, we skip step 0,1,2.

                    target_pose_rotated = target_pose.copy() #By default action transform is identity -- yielding this 

                    if control_cfg.use_rot == "z":
                        target_pose_rotated[3:6] = (st.Rotation.from_rotvec(target_pose[3:6])*action_transform).as_rotvec()
                        assert target_pose_rotated[3] < 1e-2
                        target_pose_rotated[3] = 0
                        assert target_pose_rotated[4] < 1e-2
                        target_pose_rotated[4] = 0

                    # This new frame should (granted correct assumptions) be close to a pure z-rotation, which can be represented as a rotation vector [0,0,theta]. 
                        # We use this as the action representation when working with only z-rotations. It allows us to compress the rotation information from three channels to one channel while using rotation vectors as the rotation representation
                    A = ((st.Rotation.from_rotvec(target_pose_rotated[3:6]))*(action_transform.inv())).as_matrix()
                    B = (st.Rotation.from_rotvec(target_pose[3:6])).as_matrix()
                    assert rot_dist(A,B) < 0.02 #If the joint-limited rotation so far is safe, then it is also safe to rotate, clip, and reverse the rotation since the orientations are close to each other
                    
                    if target_tool_pose is not None: 
                        action = np.hstack([target_pose_rotated,target_tool_pose])
                    else:
                        action = target_pose_rotated
                    env.exec_actions(
                        actions=[action], 
                        timestamps=[t_command_target-time.monotonic()+time.time()],
                        stages=[stage],
                        action_transform = action_transform)
                    
                elif skip_this_cycle:
                    t=time.monotonic()
                    if t>t_cycle_end:
                        print("long action. skipping cycles.")
                        n_dt_skipped = (t-t_cycle_end)//dt+1
                        print("skipped:", n_dt_skipped)
                        iter_idx += n_dt_skipped
                        t_cycle_end = t_start + (iter_idx + 1) * dt


                skip_this_cycle=False


                precise_wait(t_cycle_end) # This, combined with long actions within one loop, is believed to be the cause of lag after scripted movements. Therefore, we wait 
                iter_idx += 1

# %%
if __name__ == '__main__':
    main()