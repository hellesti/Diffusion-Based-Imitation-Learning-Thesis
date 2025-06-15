if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import os
import click
import av
import cv2
import collections
import multiprocessing as mp
import numpy as np
import threadpoolctl
from matplotlib import pyplot as plt
import json
from ffprobe import FFProbe

def get_t_mask(img, hsv_ranges=None, path=None):
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
   
    # save image with mask visualization for debug purposes
    color=[0,255,0]
    alpha=0.4
    vis = img.copy()
    img_layer = img.copy()
    img_layer[mask] = color
    vis = cv2.addWeighted(img_layer, alpha, vis, 1 - alpha, 0, vis)
    if path is not None: 
        path = str(path)
        img_idx = os.path.basename(os.path.dirname(path))
        directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(path))),"eval_img_vis")
        if not os.path.exists(directory):
            os.mkdir(directory)
        write_path = os.path.join(directory,str(img_idx)+".png")
        #print(write_path)
        cv2.imwrite(write_path, vis)
    return mask

def get_mask_metrics(target_mask, mask, task_time, task_timeout):
    # We can not assume that every pixel of T is correctly detected in "mask". This is due to frame size and detection validity in certain areas outside of target zone
    # Therefore, infer union from intersection by assuming T always has the same number of pixels
    total = np.sum(target_mask)
    i = np.sum(target_mask & mask)
    u = total+(total-i)
    #u = np.sum(target_mask | mask)
    reached_end_zone=True
    if task_time>=task_timeout:
        reached_end_zone=False
    iou = i / u
    coverage = i / total
    result = {
        'iou': iou,
        'coverage': coverage,
        'duration': task_time,
        'reached_end_zone': reached_end_zone
    }
    return result

def get_image_metrics(eval_image_path, target_mask, task_timeout):
    threadpoolctl.threadpool_limits(1)
    cv2.setNumThreads(1)
    metrics = collections.defaultdict(list)
    eval_image = cv2.imread(eval_image_path)
    img_shape = np.shape(eval_image)
    x = img_shape[1]
    eval_image = eval_image[:,:(x//2)]
    path = eval_image_path

    video_dir = os.path.dirname(eval_image_path)
    video_path = os.path.join(video_dir,"0.mp4")
    probe = FFProbe(video_path)
    task_timestring = probe.metadata["Duration"]
    h, m, s = task_timestring.split(':')
    task_time = 3600*float(h)+60*float(m)+float(s)
    mask = get_t_mask(eval_image, path=path)
    metric = get_mask_metrics(
        target_mask=target_mask, mask=mask, task_time = task_time, task_timeout=task_timeout)
    for k, v in metric.items():
        metrics[k].append(v)
    return metrics

def worker(x):
    return get_image_metrics(*x)

@click.command()
@click.option(
    '--reference', '-r', default="eval/new_pusht/pusht10/perfect_eval.png", 
    help="Reference video whose last frame will define goal.")
@click.option(
    '--input', '-i', default = "eval/new_pusht/pusht10_ft",
    help='Dataset path to evaluate.')
@click.option('--ep_timeout', '-to', default=60.0, type=float)
@click.option('--n_workers', '-n', default=10, type=int)


def main(reference, input, ep_timeout, n_workers):
    # read last frame of the reference video to get target mask
    eval_array = cv2.imread(reference)
    array_shape = np.shape(eval_array)
    x = array_shape[1]
    eval_array = eval_array[:,:(x//2)]
    target_mask = get_t_mask(eval_array)

    # get metrics for each episode
    episode_video_path_map = dict() 
    input_dir = pathlib.Path(input)
    input_video_dir = input_dir.joinpath('videos')
    for vid_dir in input_video_dir.glob("*/"):
        #print(vid_dir)
        episode_idx = int(vid_dir.stem)
        eval_img_path = vid_dir.joinpath('eval.png')
        if eval_img_path.exists():
            episode_video_path_map[episode_idx] = str(eval_img_path.absolute())

    episode_idxs = sorted(episode_video_path_map.keys())
    print(f"Found video for following episodes: {episode_idxs}")

    # run
    with mp.Pool(n_workers) as pool:
        args = list()
        for idx in episode_idxs:
            args.append((episode_video_path_map[idx], target_mask, ep_timeout))
        results = pool.map(worker, args)
    episode_metric_map = dict()
    for idx, result in zip(episode_idxs, results):
        episode_metric_map[idx] = result

    # aggregate metrics
    agg_map = collections.defaultdict(list)
    for idx, metric in episode_metric_map.items():
        for key, value in metric.items():
            agg_map['max/'+key].append(np.max(value))
            agg_map['last/'+key].append(value[-1])

    final_metric = dict()
    for key, value in agg_map.items():
        final_metric[key] = np.mean(value)

    # save metrics
    print('Saving metrics!')
    with input_dir.joinpath('metrics_agg.json').open('w') as f:
        json.dump(final_metric, f, sort_keys=True, indent=2)
    
    with input_dir.joinpath('metrics_raw.json').open('w') as f:
        json.dump(episode_metric_map, f, sort_keys=True, indent=2)
    print('Done!')

if __name__ == '__main__':
    main()