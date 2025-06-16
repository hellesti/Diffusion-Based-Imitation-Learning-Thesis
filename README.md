# Diffusion-Based Imitation Learning in Real-World Robotics

This repository contains the code used for the experiments presented in the master's thesis *Evaluating Diffusion-Based Imitation Learning Under Dynamic Disturbances*. It builds upon the codebase and README developed by Isosomppi [Diffusion_Policy_Thesis_mtplab](https://github.com/niiloemil/diffusion_policy_thesis_mtplab), which itself was adapted from [Stanford's Diffusion Policy](https://github.com/real-stanford/diffusion_policy). In this thesis, Bidirectional Decoding (BID) was integrated into the system using modules from the official [bid_diffusion repository](https://github.com/YuejiangLIU/bid_diffusion), and Consistency Policy (CP) was implemented from the [Consistency-Policy repository](https://github.com/Aaditya-Prasad/Consistency-Policy).

The goal of this work was to evaluate the robustness and reactivity of diffusion-based imitation learning methods in dynamic, real-world environments. Two reproducible test cases were developed and deployed on a physical UR10 robot platform. The first case assessed reactivity under fast and unstable pendulum dynamics. The second simulated slow and persistent environmental disturbances such as actuation noise. Models evaluated include:

- **Diffusion Policy (DP)**: Baseline method used to test performance in static and dynamic conditions.
- **Bidirectional Decoding (BID)**: Extends DP with closed-loop resampling to improve reactivity.
- **Consistency Policy (CP)**: A latency-aware policy distilled from DP, integrated into the BID pipeline to reduce inference time.

Over 1,200 physical evaluations were conducted across the two test cases, with detailed comparisons across methods and inference horizons. All code for data collection, training, inference, and benchmarking is included in this repository and can be used to replicate or extend the experiments.

The demonstration data and checkpoints can be made avaleble upon request.

<img src="Media/setup_w_cups.png" alt="setup_w_cups.png" width="40%"/>
<img src="Media/full_setup_image.png" alt="full_setup_image.png" width="40%"/>

## 💾 Installation

To reproduce our environment, install the given conda environment on a Linux machine with an Nvidia RTX 3090 GPU. 

The recommended setup uses Mambaforge for faster dependency resolution. While <span style="color:ff5733
"> Mambaforge </span> is discouraged by its authors as of September 2023, it is what has been tested in the lab. For future users of the setup, it may be worthwhile upgrading to <span style="color:ff5733
"> Miniforge </span>.

[Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) installation: 

It is assumed that you have already agreed to the licence when running with the flag -b. In terminal:

```console
cd ~
wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
bash Mambaforge-Linux-x86_64.sh -b
~/mambaforge/bin/conda init bash
source ~/.bashrc
```

(Optional): Disable the ASCII banner:
```console
mamba env config vars set MAMBA_NO_BANNER=1
source ~/.bashrc
```


Install required system packages, dependencies for Spacemouse:
```console
sudo apt update
sudo apt install libspnav-dev spacenavd cmake build-essential
sudo systemctl start spacenavd
```

You will need cmake:
```console
sudo apt-get -y install cmake
sudo apt-get install build-essential
```

Upgrade pip:
```console
pip install --upgrade pip
pip install --upgrade setuptools
```

Create and activate the environment (took around 10 minutes to execute):
```console
cd ~/Documents/GitHub/diffusion_policy_mtplab
mamba env create -f conda_environment_real.yaml
conda activate rd
```
## 🔌 Components to connect 

- Two Intel RealSense cameras (USB 3.1), connect the PC to the ethernet switch.
- Key light, turn it to mode I.
- Ethernet switch connected to:
  - PC (192.168.1.1)
  - UR10 CB3 (192.168.1.10)
  - Robotiq gripper (192.168.1.12)
  - Force-torque sensor (192.168.1.13)


Be sure to power the switch, the PSU, and the CB3 for UR10.

If you are going to use the suction cup, ensure that the Arduino Nano is plugged to the computer via USB cable. Additionally, ensure that the 8mm tube connected to the solenoid valve is plugged into the compressed air supply in the wall, the wall valve is open, and the pressure reads 5bar. 

## 💾 Robot setup

Ensure:
- The robot and PC are connected via Ethernet through the switch.
- The PC has a static IP address: 192.168.1.1
- The UR10 teach pendant network config is set to: 192.168.1.10
- Gripper and FT sensor are reachable via their static IPs.

Verify connectivity by pinging each device from the PC.
Set payload on the teach pendant to 2.18kg. Check that no extra IO protocols are active, verify that in `program robot` -> `installation`:
- MODBUS: OFF
- Ethernet/IP: OFF
- PROFINET: OFF

<span style="color:ff5733"> Warning: </span>\
Joint limits for the robot are <span style="color:ff5733"> not </span> configured in the teach pendant, but in Python. Futhermore, these currently only exist when using teleop or when running model inference. Exercise caution if programming new hard-coded movements. Especially, beware of generating linear movements using moveL around the wrist singularity (when joint 5 is a multiple of ±180°). Doing this will likely break the end-effector.


## 🦾 Demo, Training and Eval on a Real Robot
Initialize and start the robot from the teach pendant (emergency stop button within reach at all time), your cameras plugged in to your workstation (tested with `test_multi_realsense.py`) and your SpaceMouse connected with the `spacenavd` daemon running (verify with `systemctl status spacenavd`).

Create a folder for your data:
```console
cd ~/Documents/GitHub/diffusion_policy_mtplab
mkdir data
```

Demonstration datasets are avaleble upon request or on the computer. Contact the author if you'd like access to these files. All experiments are based on a pendulum placement task with two test cases:

1. Fast system dynamics (Case 1)

2. Persistent environmental disturbance (Case 2)

To collect demonstrations, train policies, and run evaluations:

```console
cd ~/Documents/GitHub/diffusion_policy_mtplab
conda activate rd
python demo_real_robot_from_config.py --config gripper_3dof.yaml
```

Press "C" to start recording. Use SpaceMouse to move the robot. Press "S" to stop recording. If you are unhappy with an episode, finish the episode. Then press backspace and confirm with "y" in the terminal to remove the previous episode. Press "Q" to correctly exit the program. Press "D" to activate the disturbance for Case 2. For These keybinds (with the exception of backspace) can be modified in the config file, which is in diffusion_policy/config/control.

This should result in a demonstration dataset in `data/demo_pendulum` with in the same structure as the pendulum dataset.

To train a Diffusion Policy, launch training with config:
```console
time python train.py --config gripper_3dof.yaml train_diffusion_unet_real_image_workspace training.seed=42 task=pendulum_image task.dataset_path=data/demo_pendulum;
```

If you are working with another task than pendulum, change task=pendulum_image to another config in `diffusion_policy/config/task`. Also change the dataset path to the correct dataset path. To change the random seed, change training.seed 

Edit [`diffusion_policy/config/task/pendulum_image.yaml`](./diffusion_policy/config/task/pendulum_image.yaml) if your camera setup is different. If you are using a different end-effector tool, choose the corresponding control config when making demonstrations. Use the same config during inference. Remember to specify the correct task when training. Find different task configs in diffusion_policy/config/task

Assuming the training has finished and you have a checkpoint at `data/outputs/blah/checkpoints/latest.ckpt`, launch the evaluation script with:
For Case 1, DP then BID (BID-CP is run just like BID, but with different checkpoints):
```console
python eval_real_robot_from_config.py --config pusht10.yaml -i data/outputs/blah/checkpoints/latest.ckpt

python eval_real_robot_from_config_BID.py --config pusht10.yaml -i data/outputs/blah/checkpoints/latest.ckpt -bc data/outputs/blah/checkpoints/bad.ckpt
```

For Case 2, DP then BID (BID-CP is run just like BID, but with different checkpoints):
```console
python eval_disturbance_real_robot_from_config_experimental.py --config pusht10.yaml -i data/outputs/blah/checkpoints/latest_delta.ckpt

python eval_disturbance_real_robot_from_config_BID_exp.py --config pusht10.yaml -i data/outputs/blah/checkpoints/latest.ckpt -bc data/outputs/blah/checkpoints/bad_delta.ckpt
```

Press "C" to start evaluation (handing control over to the policy).\
Press "D" to activate the disturbance for Case 2.\
Press "S" to stop the current episode.\
Press "Q" to correctly exit the program.
