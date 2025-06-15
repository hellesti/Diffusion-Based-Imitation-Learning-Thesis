if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)

import os
import click
import av
import cv2
import collections
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
import threadpoolctl
from matplotlib import pyplot as plt
import json
from diffusion_policy.real_world.single_realsense import SingleRealsense


# usage: 
    # put end-effector in evaluation image position
    # choose (exposure, gain) values in realsense init 
    # choose hsv filter values below
    # run script
    # observe detection
    # tune hsv values if needed

def get_t_mask(img, hsv_ranges=None):
    if hsv_ranges is None:
        hsv_ranges = [
            [0,255],
            [150,255],
            [140,255]
        ]
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = np.ones(img.shape[:2], dtype=bool)
    for c in range(len(hsv_ranges)):
        l, h = hsv_ranges[c]
        mask &= (l <= hsv_img[...,c])
        mask &= (hsv_img[...,c] <= h)
    return mask

def get_mask_metrics(target_mask, mask):
    total = np.sum(target_mask)
    i = np.sum(target_mask & mask)
    u = np.sum(target_mask | mask)
    iou = i / u
    coverage = i / total
    result = {
        'iou': iou,
        'coverage': coverage
    }
    return result


import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import cv2
import json
import time
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.real_world.single_realsense import SingleRealsense

def test():
    
    serials = SingleRealsense.get_connected_devices_serial()
    # import pdb; pdb.set_trace()
    serial = serials[0]
    config = json.load(open('/home/niiloemil/Documents/GitHub/diffusion_policy_emil/diffusion_policy/real_world/realsense_config/435_high_accuracy_mode.json', 'r'))

    def transform(data):
        color = data['color']
        h,w,_ = color.shape
        factor = 2
        color = cv2.resize(color, (w//factor,h//factor), interpolation=cv2.INTER_AREA)
        # color = color[:,140:500]
        data['color'] = color
        return data

    # at 960x540 with //3, 60fps and 30fps are indistinguishable

    with SharedMemoryManager() as shm_manager:
        with SingleRealsense(
            shm_manager=shm_manager,
            serial_number=serial,
            resolution=(1280,720),
            # resolution=(960,540),
            # resolution=(640,480),
            capture_fps=30,
            enable_color=True,
            # enable_depth=True,
            # enable_infrared=True,
            # advanced_mode_config=config,
            # transform=transform,
            # recording_transform=transform
            # verbose=True
            ) as realsense:
            cv2.setNumThreads(1) 
            realsense.set_exposure(exposure=80, gain=0)
            realsense.set_white_balance(white_balance=5900)

            intr = realsense.get_intrinsics()
            print(intr)

            current_dir = os.path.abspath(__file__)
            #video_path = os.path.join(os.path.dirname(current_dir), "recordings/test_rs.mp4")
            #video_path = 'data_local/test.mp4'
            #rec_start_time = time.time() + 2
            #realsense.start_recording(video_path, start_time=rec_start_time)

            data = None
            while True:
                data = realsense.get(out=data)
                t = time.time()
                # print('capture_latency', data['receive_timestamp']-data['capture_timestamp'], 'receive_latency', t - data['receive_timestamp'])
                # print('receive', t - data['receive_timestamp'])

                # dt = time.time() - data['timestamp']
                # print(dt)
                # print(data['capture_timestamp'] - rec_start_time)

                img = data['color']

                mask = get_t_mask(img)
                    ### save masked images for debug
                color=[0,255,0]
                alpha=0.4
                vis = img.copy()
                img_layer = img.copy()
                img_layer[mask] = color
                vis = cv2.addWeighted(img_layer, alpha, vis, 1 - alpha, 0, vis)
                # print(bgr.shape)
                cv2.imshow('default', vis)
                key = cv2.pollKey()
                # if key == ord('q'):
                #     break
                # elif key == ord('r'):
                #     video_path = 'data_local/test.mp4'
                #     realsense.start_recording(video_path)
                # elif key == ord('s'):
                #     realsense.stop_recording()
                
                time.sleep(1/60)
                #if time.time() > (rec_start_time + 20.0):
                    #break


if __name__ == "__main__":
    test()
