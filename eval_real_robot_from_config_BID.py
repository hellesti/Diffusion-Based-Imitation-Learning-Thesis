"""
Usage:
(robodiff)$ python eval_real_robot.py -i <ckpt_path> -o <save_dir> --robot_ip <ip_of_ur5>

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
from click.core import ParameterSource
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
import skvideo.io
from omegaconf import OmegaConf
import scipy.spatial.transform as st
from diffusion_policy.real_world.real_env_gripper import RealEnv as RealEnvGripper 
from diffusion_policy.real_world.real_env_suctioncup import RealEnv as RealEnvSuctioncup 
#from diffusion_policy.real_world.real_env_stick import RealEnv as RealEnvStick 
from diffusion_policy.real_world.real_env_pusht import RealEnv as RealEnvPushT
from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution, 
    get_real_obs_dict)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.common.rot_limit_util import TCPLimiter, pose_is_within_limits, rot_dist
import os
from diffusion_policy.sampler.multi import bidirectional_sampler

OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('--config', '-c', required=False, default="gripper_3dof.yaml" , help="Name of config file (in diffusion_policy/config/control)")

@click.option('--input', '-i', default=None, help='Path to checkpoint')
@click.option('--bad_checkpoint', '-bc', default=None, help='path to early underfitting checkpoint')
@click.option('--output', '-o', default=None, help='Directory to save recording')
@click.option('--robot_ip', '-ri', default=None, help="UR5's IP address e.g. 192.168.1.10")
@click.option('--tool_address', '-ta', default=None, help="tool's IP address e.g. 192.168.0.12")
@click.option('--ft_sensor_ip', '-fi', default =None, help="ft sensor's IP address e.g. 192.168.1.13")
@click.option('--tool_init_pos', '-ti', default=None, help="tool's IP address e.g. 192.168.0.12")
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--vis_camera_idx', default=None, type=int, help="Which RealSense camera to visualize.")
@click.option('--steps_per_inference', '-si', default=None, type=int, help="Action horizon for inference.") #6
@click.option('--max_duration', '-md', default=None, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=None, type=float, help="Control frequency in Hz.") 
@click.option('--command_latency', '-cl', default=None, type=float, help="Latency between receiving SpaceMouse command to executing on Robot in Sec.") #0.01

def main(config, input, bad_checkpoint, output, robot_ip, tool_address, ft_sensor_ip, tool_init_pos, match_dataset, match_episode, vis_camera_idx, steps_per_inference, max_duration, frequency, command_latency):
    # Check if any click options are defaults 
    # Override default options with config options
    # Load config vals for launch options
    
    config_path = "diffusion_policy/config/control/"+config 

    control_cfg = OmegaConf.load(config_path)

    launch_options={"input":input, 
                    "bad_checkpoint":bad_checkpoint,
                    "output":output,
                    "robot_ip":robot_ip,
                    "tool_address":tool_address, 
                    "ft_sensor_ip":ft_sensor_ip,
                    "tool_init_pos":tool_init_pos,
                    "match_dataset":match_dataset,
                    "match_episode":match_episode,   
                    "vis_camera_idx":vis_camera_idx,
                    "steps_per_inference":steps_per_inference,
                    "max_duration":max_duration,
                    "frequency":frequency,
                    "command_latency":command_latency}
    
    click_context= click.get_current_context()
    for key, val in launch_options.items():
        if click_context.get_parameter_source(key) == ParameterSource.DEFAULT: #if option is default value, use config
            launch_options[key] = control_cfg.eval_launch_options[key]
        #else: keep value which has been input by user; do nothing

    input = launch_options["input"]
    bad_checkpoint = launch_options['bad_checkpoint']
    output = launch_options["output"]
    robot_ip = launch_options["robot_ip"]
    tool_address = launch_options["tool_address"]
    ft_sensor_ip = launch_options["ft_sensor_ip"]
    tool_init_pos = launch_options["tool_init_pos"]
    match_dataset = launch_options["match_dataset"]
    match_episode = launch_options["match_episode"]
    vis_camera_idx = launch_options["vis_camera_idx"]
    steps_per_inference = launch_options["steps_per_inference"]
    max_duration = launch_options["max_duration"]
    frequency = launch_options["frequency"]
    command_latency = launch_options["command_latency"]

    if control_cfg.real_env == "gripper":
        RealEnv = RealEnvGripper
    elif control_cfg.real_env == "suctioncup":
        RealEnv = RealEnvSuctioncup
    elif control_cfg.real_env == "stick":
        RealEnv = RealEnvStick
    elif control_cfg.real_env == "pusht":
        RealEnv = RealEnvPushT

    ft_transform_matrix = control_cfg.ft_transform_matrix

    # load match_dataset
    match_camera_idx = 0
    episode_first_frame_map = dict()
    if match_dataset is not None:
        match_dir = pathlib.Path(match_dataset)
        match_video_dir = match_dir.joinpath('videos')
        for vid_dir in match_video_dir.glob("*/"):
            episode_idx = int(vid_dir.stem)
            match_video_path = vid_dir.joinpath(f'{match_camera_idx}.mp4')
            if match_video_path.exists():
                frames = skvideo.io.vread(
                    str(match_video_path), num_frames=1)
                episode_first_frame_map[episode_idx] = frames[0]
    print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")
    
    # load checkpoint
   # ckpt_path = "data/outputs/"+input
    ckpt_path = input
    output = str(pathlib.Path(ckpt_path).parents[0]) +"/"+ output + "_" + str(pathlib.Path(ckpt_path).stem)

    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    # @Helle: Det under var noen greier vi måtte overstyre når jeg testet metoden min siden jeg var dum og endret noen navn
    if "history" in cfg.name:
        cfg._target_ = "diffusion_policy.workspace.train_bc_transformer_image_workspace.TrainBCTransformerImageWorkspace"
        cfg.name = "train_noisy_bc_transformer_real_image"
        cfg.policy._target_ = "diffusion_policy.policy.noisy_bc_transformer_image_policy.NoisyBCTransformerImagePolicy"
        cfg.policy.model_partial._target_ = "diffusion_policy.model.history_conditioned.bc_transformer.BCTransformer"

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    exclude_keys = ("scaler",)
    workspace.load_payload(payload, exclude_keys=exclude_keys, include_keys=None)

        # load bad checkpoint
   # ckpt_path = "data/outputs/"+input
    bad_ckpt_path = bad_checkpoint

    payload_bad_ckpt = torch.load(open(bad_ckpt_path, 'rb'), pickle_module=dill)
    cfg_bad_ckpt = payload_bad_ckpt['cfg']

    cls = hydra.utils.get_class(cfg_bad_ckpt._target_)
    workspace_bad_ckpt = cls(cfg_bad_ckpt)
    workspace_bad_ckpt: BaseWorkspace
    exclude_keys = ("scaler",)
    workspace_bad_ckpt.load_payload(payload_bad_ckpt, exclude_keys=exclude_keys, include_keys=None)

    # hacks for method-specific setup.
    action_offset = 0
    delta_action = False

    # diffusion model
    policy: BaseImagePolicy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    bad_policy: BaseImagePolicy
    bad_policy = workspace_bad_ckpt.model
    if cfg_bad_ckpt.training.use_ema:
        bad_policy = workspace_bad_ckpt.ema_model


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy.eval().to(device)
    bad_policy.eval().to(device)

    # set inference params
    policy.num_inference_steps = 8
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
    bad_policy.num_inference_steps = 8
    bad_policy.n_action_steps = bad_policy.horizon - bad_policy.n_obs_steps + 1

    # --- BID parameters ---
    nsample = 15
    nmode = 3
    decay = 0.5


    # setup experiment
    dt = 1/frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)
    print("action_offset:", action_offset)

    with SharedMemoryManager() as shm_manager:
        with Spacemouse(shm_manager=shm_manager, deadzone=[0.000,0.000,0.000,0.000,0.000,0.000]) as sm, RealEnv(
            output_dir=output, 
            robot_ip=robot_ip, 
            tool_address=tool_address,
            ft_sensor_ip=ft_sensor_ip,
            ft_transform_matrix=ft_transform_matrix,
            frequency=frequency,
            n_obs_steps=n_obs_steps,
            obs_image_resolution=obs_res,
            obs_float32=True,
            enable_multi_cam_vis=True,
            record_raw_video=True,
            tool_init_pos=tool_init_pos,
            # number of threads per camera view for video recording (H.264)
            thread_per_video=3,
            # video recording quality, lower is better (but slower).
            video_crf=21,
            shm_manager=shm_manager) as env:

            cv2.setNumThreads(1)

            # Should be the same as demo
            # realsense exposure
            env.realsense.set_exposure(exposure=80, gain=0)
            # realsense white balance
            env.realsense.set_white_balance(white_balance=5900)

            print("Waiting for realsense")
            env.set_ft_zero_point()

            time.sleep(1.0)
            state = env.get_robot_state()
            target_pose = state['TargetTCPPose']
            if control_cfg.real_env not in  ["pusht", "stick"]:
                target_pose = np.hstack([target_pose,tool_init_pos])

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

            print("Warming up policy inference")
            obs = env.get_obs()
            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                #Fjernet result og action herrifra

                # result = policy.predict_action(
                #     obs_dict,
                #     sample=True,
                #     num_samples=nsample,
                #     nmode=nmode,
                #     decay=decay,
                #     sampler='bid',
                #     reference=reference
                # )
                result = bidirectional_sampler(strong=policy, weak=bad_policy, obs_dict=obs_dict, prior=None, num_sample=nsample, num_mode=nmode)
                action = result['action'][0].detach().cpu().numpy()
                action_prior = None

                # Validate action shape
                if control_cfg.real_env == "pusht":
                    assert action.shape[-1] == 2
                if control_cfg.real_env == "suctioncup":
                    assert action.shape[-1] == 4
                if control_cfg.real_env == "gripper":
                    assert (action.shape[-1] == 4 or action.shape[-1] == 5 or action.shape[-1] == 7)
                del result

            if control_cfg.real_env == "gripper":
                env.goto_startpos()

            target_tool_pose = tool_init_pos
            print('Ready!')

            while True:
                # ========= human control loop ==========
                print("Human in control!")
                state = env.get_robot_state()
                target_pose = state['TargetTCPPose'] #TODO implement tool here
                
                if control_cfg.real_env not in ["pusht", "stick"]:
                    target_pose = np.hstack([target_pose,target_tool_pose])
                t_start = time.monotonic()
                iter_idx = 0
                while True:
                    #TODO needs to skip cycles after scripted movements, like demo_real_robot_from_config
                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    # pump obs
                    obs = env.get_obs()

                    # visualize
                    episode_id = env.replay_buffer.n_episodes
                    vis_img = obs[f'camera_{vis_camera_idx}'][-1]
                    match_episode_id = episode_id
                    if match_episode is not None:
                        match_episode_id = match_episode
                    if match_episode_id in episode_first_frame_map:
                        match_img = episode_first_frame_map[match_episode_id]
                        ih, iw, _ = match_img.shape
                        oh, ow, _ = vis_img.shape
                        tf = get_image_transform(
                            input_res=(iw, ih), 
                            output_res=(ow, oh), 
                            bgr_to_rgb=False)
                        match_img = tf(match_img).astype(np.float32) / 255
                        vis_img = np.minimum(vis_img, match_img)

                    text = f'Episode: {episode_id}'
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255,255,255)
                    )
                    cv2.imshow('default', vis_img[...,::-1])
                    
                    key_stroke = cv2.pollKey()
                    if (key_stroke != -1) and chr(key_stroke) in control_cfg.keybinds:
                        command=control_cfg.keybinds[chr(key_stroke)] #access dict value of bind
                    else: 
                        command = None

                        sm_state = sm.get_button_state()
                        if sm.is_button_pressed(0):
                            command=(control_cfg.sm_keybinds.button_0)
                        elif sm.is_button_pressed(1):
                            command=(control_cfg.sm_keybinds.button_1)

                    if command == "stop":
                        # Exit program
                        env.end_episode()
                        exit(0)
                    elif command == "start_recording":
                        # Exit human control loop
                        # hand control over to the policy
                        break
                    elif command == "do_t_reset":
                        assert control_cfg.real_env == "pusht"
                        env.do_T_reset(seed_ep_num=True)
                        env.goto_startpos()
                        state = env.get_robot_state()
                        temp_pose = state['TargetTCPPose']
                        assert pose_is_within_limits(temp_pose, control_cfg.tcp_limits, R_bd)
                        target_pose = state['TargetTCPPose'] #If endpos differs from startpos, this resets servo target pos to movement endpos
                        skip_this_cycle = True
                    elif command == "activate_tool":
                        assert control_cfg.real_env in ["gripper", "suctioncup"]
                        target_pose[-1] = True
                    elif command == "deactivate_tool":
                        assert control_cfg.real_env in ["gripper", "suctioncup"]
                        target_pose[-1] = False
                    elif command == "print_ee_pose":    
                        state = env.get_robot_state()
                        target_pose = state['TargetTCPPose']
                        print(f"pos: {target_pose.tolist()}")
                    elif command == "print_ee_joints":
                        state = env.get_robot_state()
                        target_joints = state['TargetQ']
                        print(f"joints: {target_joints.tolist()}")
                    elif command == None:
                        pass
                    else:
                        print("Key not bound")

                    precise_wait(t_sample)
                    # get teleop command

                    if not skip_this_cycle:
                        sm_state = sm.get_motion_state_transformed()
                        #print(sm_state)
                        dpos = sm_state[:3] * (env.max_pos_speed / frequency)
                        drot_xyz = sm_state[3:6] * (env.max_rot_speed / frequency)
        
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
                        
                        target_pose=tcp_limiter.get_limited_pose(target_pose)
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

                        action = target_pose_rotated

                        env.exec_actions(
                            actions=[action], 
                            timestamps=[t_command_target-time.monotonic()+time.time()],
                            stages=None,
                            action_transform=action_transform)
                        precise_wait(t_cycle_end)
                    
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

                # ========== policy control loop ==============
                try:
                    # start episode
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/30
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    term_area_start_timestamp = float('inf')
                    perv_target_pose = None
                    inference_latencies = []
                    while True:
                        #s0 = time.time()
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get obs
                        print('get_obs')
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta)
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            #Fjernet result og action herifra

                            # result = policy.predict_action(
                            #     obs_dict,
                            #     sample=True,
                            #     num_samples=nsample,
                            #     nmode=nmode,
                            #     decay=decay,
                            #     sampler='bid',
                            #     reference=reference
                            # )
                            result = bidirectional_sampler(strong=policy, weak=bad_policy, obs_dict=obs_dict, prior=action_prior, num_sample=nsample, num_mode=nmode)
                            action_prior = result['action_pred'][:, policy.n_action_steps:]
                            action = result['action'][0].detach().cpu().numpy()


                            latency = time.time() - s
                            print('Inference latency:', latency)
                            inference_latencies.append(latency)
                        
                        # convert policy action to env actions
                        if delta_action: 
                            assert (control_cfg.real_env == "pusht")
                            assert len(action) == 1
                            if perv_target_pose is None:
                                perv_target_pose = obs['robot_eef_pose'][-1]
                            this_target_pose = perv_target_pose.copy()
                            this_target_pose[[0,1]] += action[-1]
                            perv_target_pose = this_target_pose
                            this_target_poses = np.expand_dims(this_target_pose, axis=0)
                        else: # default behaviour:
                            this_target_poses = np.zeros((len(action), len(target_pose)), dtype=np.float64)
                            this_target_poses[:] = target_pose
                            if control_cfg.real_env == "pusht":
                                this_target_poses[:,[0,1]] = action
                            elif (control_cfg.real_env == "suctioncup" or control_cfg.real_env == "gripper"):
                                if len(action[0]) == 4:
                                    action[:, -1] = np.round(action[:, -1], decimals=0) # only 0 or 1
                                    this_target_poses[:,[0,1,2,6]] = action
                                elif len(action[0]) == 5:
                                    action[:, -1] = np.round(action[:, -1], decimals=0)
                                    this_target_poses[:,[0,1,2,5,6]] = action
                                    this_target_poses[:,[3,4]] = np.array([0,0])
                                elif len(action[0]) == 7:
                                    action[:, -1] = np.round(action[:, -1], decimals=0)
                                    this_target_poses[:,[0,1,2,3,4,5,6]] = action
                                else:
                                    raise Exception("Invalid action length")
                            elif control_cfg.real_env == "stick":
                                raise NotImplementedError
                            else:
                                raise NotImplementedError
                            
                        action_timestamps = (np.arange(len(action), dtype=np.float64) + action_offset
                            ) * dt + obs_timestamps[-1]
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        if np.sum(is_new) == 0:
                            # exceeded time budget, still do something
                            this_target_poses = this_target_poses[[-1]]
                            # schedule on next available step
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            print('Over budget', action_timestamp - curr_time)
                            action_timestamps = np.array([action_timestamp])
                        else:
                            this_target_poses = this_target_poses[is_new]
                            action_timestamps = action_timestamps[is_new]
                        
                        for i in range(len(this_target_poses)):
                            if control_cfg.use_rot == "z":
                                this_target_poses[i][3:6] = ((st.Rotation.from_rotvec(this_target_poses[i][3:6]))*(action_transform.inv())).as_rotvec()
                            this_target_poses[i] = tcp_limiter.get_limited_pose(this_target_poses[i])
                            if control_cfg.use_rot == "z":
                                this_target_poses[i][3:6] =  ((st.Rotation.from_rotvec(this_target_poses[i][3:6]))*action_transform).as_rotvec()

                        # execute actions #TODO add post-rot for zrot only control configurations

                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps,
                            stages=None,
                            action_transform=action_transform #during inference, pre transform has already been done in demo. We simply add the post-transform to undo it.
                        )
                        
                        """ print(f"Submitted {len(this_target_poses)} steps of actions.")
                        # Use a list comprehension to format each number
                        formatted_timestamps = [f"{timestamp:.7f}" for timestamp in action_timestamps]

                        # Join the formatted numbers into a single string
                        formatted_timestamps_str = ' '.join(formatted_timestamps)

                        # Print the result
                        print(f"Timesteps for the predicted actions are [{formatted_timestamps_str}].")
                        """
                        if control_cfg.eval_launch_options.tool_init_pos is not None:
                            target_tool_pose = this_target_poses[-1][6]
                        # visualize
                        episode_id = env.replay_buffer.n_episodes
                        vis_img = obs[f'camera_{vis_camera_idx}'][-1]
                        text = 'Episode: {}, Time: {:.1f}'.format(
                            episode_id, time.monotonic() - t_start
                        )
                        cv2.putText(
                            vis_img,
                            text,
                            (10,20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255,255,255)
                        )
                        cv2.imshow('default', vis_img[...,::-1])

                        key_stroke = cv2.pollKey()
                        if (key_stroke != -1) and chr(key_stroke) in control_cfg.keybinds:
                            command=control_cfg.keybinds[chr(key_stroke)] #access dict value of bind
                        else: 
                            command = None

                        if command == "stop_recording":
                            # Stop episode
                            # Hand control back to human
                            env.end_episode()
                            print('Stopped.')
                            break

                        # auto termination
                        terminate = False
                        #TODO add termination conditions to cfg

                        if time.monotonic() - t_start > max_duration:
                            terminate = True
                            print('Terminated by the timeout!')
                            if control_cfg.real_env == "pusht":
                                assert env.go_up(0.100)

                        term_dist = control_cfg.eval_termination_condition.dist
                        term_time = control_cfg.eval_termination_condition.time
                        term_pos = control_cfg.eval_termination_condition.pos
                        if (not None in [term_pos, term_dist, term_time]):
                            term_pos = np.array(term_pos)
                            if len(term_pos == 3):
                                curr_pos = obs['robot_eef_pose'][-1][:3]
                                dist = np.linalg.norm((curr_pos - term_pos)[:2], axis=-1)
                                if dist < term_dist:
                                    # in termination area
                                    curr_timestamp = obs['timestamp'][-1]
                                    if term_area_start_timestamp > curr_timestamp:
                                        term_area_start_timestamp = curr_timestamp
                                    else:
                                        term_area_time = curr_timestamp - term_area_start_timestamp
                                        if term_area_time > term_time:
                                            terminate = True
                                            print('Terminated by reaching the end zone!')
                                else:
                                    # out of the area
                                    term_area_start_timestamp = float('inf')

                        if terminate:
                            env.end_episode()
                            state = env.get_robot_state()
                            target_pose = state['TargetTCPPose']
                            path = env.video_dir.joinpath("inference_latencies.txt")
                            file_exists = os.path.isfile(path)

                            with open(path, 'a') as file:
                                if file_exists:
                                    file.write('\n')
                                
                                file.write(f"Episode: {env.replay_buffer.n_episodes}, avg. latency: {np.mean(inference_latencies)}")

                            break

                        # wait for execution
                        # print("whole loop:",time.time()-s0)
                        # print("remaining time:", (t_cycle_end - frame_latency)-time.monotonic())
                        precise_wait(t_cycle_end - frame_latency)

                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()
                    state = env.get_robot_state()
                    target_pose = state['TargetTCPPose']
                
                print("Stopped.")

# 
if __name__ == '__main__':
    main()

# %%
