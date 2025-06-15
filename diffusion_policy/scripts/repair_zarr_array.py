if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

from diffusion_policy.common.replay_buffer import ReplayBuffer
import numpy as np

def repair_zarr_array(dir):
    rb = ReplayBuffer.copy_from_path(zarr_path=dir)
    rb 
    for key in rb.data:
        if np.shape(rb.data[key][0])==():
            print("Detected incorrect format. Fixing.")
            wrong_arr =  rb.data[key]
            corr_arr = np.reshape(wrong_arr, (wrong_arr.shape[0], 1))
            rb.data[key] = corr_arr
    rb.save_to_path(zarr_path=dir)

if __name__ == "__main__":
    repair_zarr_array(dir="data/demo_cylinder_peg_regrasp/replay_buffer.zarr")