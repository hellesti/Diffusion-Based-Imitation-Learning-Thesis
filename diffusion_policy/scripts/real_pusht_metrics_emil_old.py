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
from tqdm import tqdm
import threadpoolctl
from matplotlib import pyplot as pltq
import json
import time

def get_t_mask(img, hsv_ranges=None, path=None):

    y0=0
    y1=-200 #use negative idx or this will break
    x0=200
    x1=-200 #use negative idx or this will break

    cropped_img = img[y0:y1, x0:x1]
    red,green,blue = cv2.split(cropped_img) 
    #gray = cv2.cvtColor(eval_image, cv2.COLOR_BGR2GRAY) 
    edged = cv2.Canny(red, 30, 200) #30,200
    kernel = np.ones((9,9),np.uint8)
    edged = cv2.dilate(edged, kernel, iterations = 1)
    #edged = cv2.erode(edged, kernel, iterations= 1)
    contours , hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
    mask_rgb = np.zeros_like(cropped_img)


    #print("Number of Contours found = " + str(len(contours))) 
    idx = len(contours)-1

    cv2.drawContours(mask_rgb, contours, idx, (0, 255, 0), 5) #draw edge
    cv2.drawContours(mask_rgb, contours, idx, (0,255,0), -1) #draw fill

    #cv2.drawContours(cropped_img, contours, idx, (0, 255, 0), 5) 
    #cv2.drawContours(cropped_img, contours, idx, (255,255,255), -1)

    mask_rgb = cv2.copyMakeBorder(mask_rgb, y0, -y1, x0, -x1, cv2.BORDER_CONSTANT,np.zeros(3)) #assuming use of negative idx for x1 y1 earlier

    #cv2.imshow("my image", mask_rgb)

    #k = cv2.waitKey(0) & 0xFF
    #if k == 27:  # close on ESC key
    #    cv2.destroyAllWindows()

    if hsv_ranges is None:
        hsv_ranges = [
            [0,255],   #0, 255
            [0,255], #130 216
            [1,255]  #150 230
        ]
    hsv_img = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2HSV)
    mask = np.ones(img.shape[:2], dtype=bool)
    for c in range(len(hsv_ranges)):
        l, h = hsv_ranges[c]
        mask &= (l <= hsv_img[...,c])
        mask &= (hsv_img[...,c] <= h)

    ### save masked images for debug
    color=[0,255,0]
    alpha=0.4
    vis = img.copy()
    img_layer = img.copy()
    img_layer[mask] = color
    vis = cv2.addWeighted(img_layer, alpha, vis, 1 - alpha, 0, vis)
    

    if path is not None: 
        path = str(path)
        img_idx = os.path.basename(os.path.dirname(path))
        write_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(path))),"eval_img_vis",str(img_idx)+".png")
        #print(write_path)
        cv2.imwrite(write_path, vis)

    ###

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

def get_video_metrics(eval_image_path, target_mask, use_tqdm=True):
    threadpoolctl.threadpool_limits(1)
    cv2.setNumThreads(1)
    metrics = collections.defaultdict(list)
    eval_image = cv2.imread(eval_image_path)

    path = eval_image_path

    mask = get_t_mask(eval_image, path=path)
###
###

    metric = get_mask_metrics(
        target_mask=target_mask, mask=mask)
    for k, v in metric.items():
        metrics[k].append(v)
    return metrics

def worker(x):
    return get_video_metrics(*x)

@click.command()
@click.option(
    '--reference', '-r', required=True, 
    help="Reference video whose last frame will define goal.")
@click.option(
    '--input', '-i', required=True,
    help='Dataset path to evaluate.')
@click.option('--n_workers', '-n', default=20, type=int)
def main(reference, input, n_workers):
    # read last frame of the reference video to get target mask
    eval_array = cv2.imread(reference)
    #with av.open(reference) as container:#
    #    stream = container.streams.video[0]
    #    for frame in tqdm(
    #            container.decode(stream), 
    #            total=stream.frames):
    #        eval_frame = frame

    target_mask = get_t_mask(eval_array)

    # path = '/home/ubuntu/dev/diffusion_policy/data/pusht_real/eval_20230109/diffusion_hybrid_ep136/videos/4/0.mp4'
    # last_frame = None
    # with av.open(path) as container:
    #     stream = container.streams.video[0]
    #     for frame in tqdm(
    #             container.decode(stream), 
    #             total=stream.frames):
    #         last_frame = frame
    # img = last_frame.to_ndarray(format='rgb24')
    # mask = get_t_mask(img)

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
    print(f"Found eval image for following episodes: {episode_idxs}")

    # run
    with mp.Pool(n_workers) as pool:
        args = list()
        for idx in episode_idxs:
            args.append((episode_video_path_map[idx], target_mask))
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
