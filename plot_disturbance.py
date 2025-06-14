import matplotlib.pyplot as plt
import numpy as np
from diffusion_policy.common.env_util import propagate_random_walk

# DISTURBANCE PARAMS #
# Easy Mode: bound: 0.00005, scale_factor: 0.0005
# Medium Mode: bound: 000275 or 0.00015, scale_factor: 00275 or 0.0015
# Hard Mode: bound: 0.0005, scale_factor: 0.005

disturbance_seed = 19
dt = 0.1
if isinstance(disturbance_seed, type(None)):
    drng = None
else:
    drng = np.random.RandomState(disturbance_seed)
disturbance_pos = np.array([0.0,0.0])
disturbance_vel = np.array([0.0,0.0])
disturbance_bounds = np.array([[
    -0.0005, 0.0005
],
[
    -0.0005, 0.0005
]])/2
smoothness = 1.0
bound_strength=100.0
scale_factors = np.array([0.005,0.005])/2

propagate_disturbance = lambda pos,vel,drng: propagate_random_walk(pos, vel, dt, disturbance_bounds, smoothness, scale_factors, generator=drng, bound_strength=bound_strength)

# Create list with absolute disturbance reference values for 1000 steps (100 seconds of running)
all_disturbance_pos = np.zeros((120,2))

for i in range(119):
    disturbance_pos, disturbance_vel,_ = propagate_disturbance(disturbance_pos, disturbance_vel, drng)
    all_disturbance_pos[i+1,:] = disturbance_pos


np.savetxt('disturbance_values_ds_19.txt', all_disturbance_pos)
# plt.scatter(all_dist[:,0], all_dist[:,1])
# fig = plt.gcf()
# fig.savefig("figs/disturbance_traj.png", format='png')

# dists = np.loadtxt('logged_values/train_ctm_unet_hybrid_delta_dist_values_ds_3.txt')
# acts = np.loadtxt('logged_values/train_ctm_unet_hybrid_delta_act_values_ds_3.txt')

# print(dists.shape)
# print(acts.shape)

# plt.scatter(dists[:,0], dists[:,1])
# plt.scatter(acts[:,0], acts[:,1])
# plt.legend(['dists', 'acts'])
# plt.show()