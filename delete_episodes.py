import zarr
import numpy as np



def delete_eps_from_data(data_path, ep_list):
    data = zarr.open(data_path)
    ep_ends = np.array(data.meta.episode_ends)
    ep_to_del_ends = ep_ends[ep_list]
    ep_starts = [ep - 1 for ep in ep_list]
    ep_to_del_starts = ep_ends[ep_starts]
    


    action_numpy = np.array(data.data.action)
    robot_eef_pose_numpy = np.array(data.data.robot_eef_pose)
    timestamps_numpy = np.array(data.data.timestamp)
    stage_numpy = np.array(data.data.stage)
    ft_ee_wrench_numpy = np.array(data.data.ft_ee_wrench)
    gripper_encoder_pos_numpy = np.array(data.data.gripper_encoder_pos)
    gripper_motor_current_numpy = np.array(data.data.gripper_motor_current)
    gripper_obj_grab_status_numpy = np.array(data.data.gripper_obj_grab_status)
    gripper_pos_req_echo_numpy = np.array(data.data.gripper_pos_req_echo)
    robot_eef_pose_vel_numpy = np.array(data.data.robot_eef_pose_vel)
    robot_joint_numpy = np.array(data.data.robot_joint)
    robot_joint_vel_numpy = np.array(data.data.robot_joint_vel)

    for ep_idx in ep_list:
        ep_end_idx = ep_ends[ep_idx]
        ep_start_idx = ep_ends[ep_idx-1]
        ep_length = ep_end_idx - ep_start_idx
        first_act_next_ep_before = action_numpy[ep_end_idx]
        action_numpy = np.concatenate((action_numpy[:ep_start_idx],action_numpy[ep_end_idx:]), axis=0)
        first_act_next_ep_after = action_numpy[ep_start_idx]
        print(first_act_next_ep_before - first_act_next_ep_after)
        robot_eef_pose_numpy = np.concatenate((robot_eef_pose_numpy[:ep_start_idx],robot_eef_pose_numpy[ep_end_idx:]), axis=0)
        timestamps_numpy = np.concatenate((timestamps_numpy[:ep_start_idx],timestamps_numpy[ep_end_idx:]), axis=0)
        stage_numpy = np.concatenate((stage_numpy[:ep_start_idx],stage_numpy[ep_end_idx:]), axis=0)
        ft_ee_wrench_numpy = np.concatenate((ft_ee_wrench_numpy[:ep_start_idx],ft_ee_wrench_numpy[ep_end_idx:]), axis=0)
        gripper_encoder_pos_numpy = np.concatenate((gripper_encoder_pos_numpy[:ep_start_idx],gripper_encoder_pos_numpy[ep_end_idx:]), axis=0)
        gripper_motor_current_numpy = np.concatenate((gripper_motor_current_numpy[:ep_start_idx],gripper_motor_current_numpy[ep_end_idx:]), axis=0)
        gripper_obj_grab_status_numpy = np.concatenate((gripper_obj_grab_status_numpy[:ep_start_idx],gripper_obj_grab_status_numpy[ep_end_idx:]), axis=0)
        gripper_pos_req_echo_numpy = np.concatenate((gripper_pos_req_echo_numpy[:ep_start_idx],gripper_pos_req_echo_numpy[ep_end_idx:]), axis=0)
        robot_eef_pose_vel_numpy = np.concatenate((robot_eef_pose_vel_numpy[:ep_start_idx],robot_eef_pose_vel_numpy[ep_end_idx:]), axis=0)
        robot_joint_numpy = np.concatenate((robot_joint_numpy[:ep_start_idx],robot_joint_numpy[ep_end_idx:]), axis=0)
        robot_joint_vel_numpy = np.concatenate((robot_joint_vel_numpy[:ep_start_idx],robot_joint_vel_numpy[ep_end_idx:]), axis=0)
        ep_ends[ep_idx+1:] -= ep_length
        ep_ends = np.concatenate((ep_ends[:ep_idx],ep_ends[ep_idx+1:]))
    
    data.data.create_dataset(name='action', data=action_numpy, overwrite=True)
    data.data.create_dataset(name='robot_eef_pose', data=robot_eef_pose_numpy, overwrite=True)
    data.data.create_dataset(name='timestamp', data=timestamps_numpy, overwrite=True)
    data.meta.create_dataset(name='episode_ends', data=ep_ends, overwrite=True)
    data.data.create_dataset(name='stage', data=stage_numpy, overwrite=True)
    data.data.create_dataset(name='ft_ee_wrench', data=ft_ee_wrench_numpy, overwrite=True)
    data.data.create_dataset(name='gripper_encoder_pos', data=gripper_encoder_pos_numpy, overwrite=True)
    data.data.create_dataset(name='gripper_motor_current', data=gripper_motor_current_numpy, overwrite=True)
    data.data.create_dataset(name='gripper_obj_grab_status', data=gripper_obj_grab_status_numpy, overwrite=True)
    data.data.create_dataset(name='gripper_pos_req_echo', data=gripper_pos_req_echo_numpy, overwrite=True)
    data.data.create_dataset(name='robot_eef_pose_vel', data=robot_eef_pose_vel_numpy, overwrite=True)
    data.data.create_dataset(name='robot_joint', data=robot_joint_numpy, overwrite=True)
    data.data.create_dataset(name='robot_joint_vel', data=robot_joint_vel_numpy, overwrite=True)


delete_eps_from_data("file_path", [None])