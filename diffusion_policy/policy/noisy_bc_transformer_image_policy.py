if __name__ == "__main__":
    import sys
    import os
    import pathlib
    import time
    import dill
    import hydra
    import numpy as np

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

    from diffusion_policy.workspace.base_workspace import BaseWorkspace
    from diffusion_policy.dataset.dynamic_push_dataset import DynamicPushLowdimDataset
    from diffusion_policy.model.history_conditioned.action_heads import Linear

from typing import Dict, Tuple
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.history_conditioned.bc_transformer import BCTransformer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.policy.bc_transformer_image_policy import BCTransformerImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.history_conditioned.action_heads import ActionHead


class NoisyBCTransformerImagePolicy(BCTransformerImagePolicy):
    def __init__(self, 
            n_clean_obs_steps,
            max_noise_level,
            add_noise_during_inference = False,
            max_inference_noise_level = None,
            add_noise_to_lowdim_obs = True,
            independent_noise_sample_per_step = True,
            **kwargs):
        super().__init__(**kwargs)

        self.n_clean_obs_steps = n_clean_obs_steps
        self.max_noise_level = max_noise_level
        self.add_noise_during_inference = add_noise_during_inference
        self.independent_noise_sample_per_step = independent_noise_sample_per_step
        assert not (add_noise_during_inference and not independent_noise_sample_per_step), "Adding inference noise requires independent noise sample per step"

        if not add_noise_to_lowdim_obs:
            self.Do_noise = self.obs_feature_dim - 2
        else:
            self.Do_noise = self.obs_feature_dim

        if isinstance(max_inference_noise_level, type(None)):
            self.max_inference_noise_level = max_noise_level
        else:
            self.max_inference_noise_level = max_inference_noise_level

 #   ========= inference  ============

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key and "hla" (high level actions) key and "tau" (timesteps since high-level actions recieved) key 
        result: must include "action" key
        """

        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B = value.shape[0]
        this_nobs = dict_apply(nobs, lambda x: x[:,[-1],...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        # reshape back to B, To, Do
        cond = nobs_features.reshape(B, 1, -1)
        #cond = cond[:,:,:-2] Use only image features
        Da = self.action_dim

        # Propagate observation history
        if self.obs_history is None:
            self.obs_history = cond
        elif self.obs_history.shape[1] < self.history_length:
            self.obs_history = torch.cat((self.obs_history, cond), dim=1)
        else:
            self.obs_history = torch.roll(self.obs_history,-1, dims=1)
            self.obs_history[:,[-1],:] = cond

        # Noise observations
        if (self.obs_history.shape[1] > self.n_clean_obs_steps) and self.add_noise_during_inference:
            B,T,Do = self.obs_history.shape
            Tn = T - self.n_clean_obs_steps
            noisy_history = copy.deepcopy(self.obs_history)
            noise_level = torch.rand(size=(B,Tn,1), dtype=noisy_history.dtype, device=noisy_history.device)*self.max_inference_noise_level
            noisy_history[:,:Tn,:self.Do_noise] = (1-noise_level)*noisy_history[:,:Tn,:self.Do_noise] \
            + noise_level*torch.normal(mean=0.0,std=1.0, size=(B,Tn,self.Do_noise), dtype=cond.dtype, device=cond.device)

            # run model
            npred = self.model(noisy_history,**self.kwargs)
            npred = self.action_head.predict_action(npred)
        else:
            # run model
            npred = self.model(self.obs_history,**self.kwargs)
            npred = self.action_head.predict_action(npred)
        
        # unnormalize prediction
        naction_pred = npred[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        action = action_pred[:,[-1]]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }

        return result
    

    def compute_loss(self, batch):
        # normalize input
        nobs = self.normalizer.normalize(batch['obs'])
        naction = self.normalizer['action'].normalize(batch['action'])
        value = next(iter(nobs.values()))
        B, H = value.shape[:2]
        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        # reshape back to B, To, Do
        cond = nobs_features.reshape(B, H, -1)
        #cond = cond[:,:,:-2] Use only image features
        Do = cond.shape[-1]
        Tn = H - self.n_clean_obs_steps

        noise_level = torch.rand(size=(B,Tn,1), dtype=cond.dtype, device=cond.device)*self.max_noise_level
        if self.independent_noise_sample_per_step:
            noise = torch.normal(mean=0.0,std=1.0, size=(B,Tn,self.Do_noise), dtype=cond.dtype, device=cond.device)
        else:
            noise = torch.normal(mean=0.0,std=1.0, size=(B,1,self.Do_noise), dtype=cond.dtype, device=cond.device).expand(-1,Tn,-1)
        cond[:,:Tn,:self.Do_noise] = (1-noise_level)*cond[:,:Tn,:self.Do_noise] + noise_level*noise

        Da = self.action_dim

        
        # Predict low-level actions
        pred = self.model(cond)
        target = naction

        loss = self.action_head.get_loss(pred, target)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss



def test():

    h_transformer = BCTransformer(
        action_horizon= 1,
        history_length= 5,
        cond_dim= 18,
        n_layer= 8,
        n_head= 4,
        n_emb= 256,
        p_drop_emb= 0.0,
        p_drop_attn = 0.3,
        causal_attn=True)
    
    action_head = Linear(256, 2)

    policy = NoisyBCTransformerImagePolicy(
        n_clean_obs_steps=2,
        max_noise_level=1.0,
        model= h_transformer,
        history_length = 5,
        obs_dim = 18,
        n_action_steps = 1
        )
    

    # We need a dataset object to set up normalizer
    zarr_path = 'data/demos/pusht/demo_pusht_abs.zarr'
    dataset = DynamicPushLowdimDataset(zarr_path=zarr_path)
    normalizer = dataset.get_normalizer()
    policy.set_normalizer(normalizer)

    device = torch.device("cuda:0")
    policy.to(device)
    policy.eval()

    cond = torch.zeros((4,1,18))
    cond0 = torch.zeros(1,1,18)
    cond1 = torch.ones(1,1,18)
    cond = torch.cat((cond0, cond1, cond0, cond1), dim=0)
    obs_dict = {'obs': cond}

    all_times = []
    t = time.time()
    example_out = policy.predict_action(obs_dict)
    pred_time = time.time() - t
    all_times.append(pred_time)

    for i in range(10):
        t = time.time()
        with torch.no_grad():
            example_out = policy.predict_action(obs_dict)['action']
        pred_time = time.time() - t
        all_times.append(pred_time)
    
    example_out = example_out#.cpu().numpy()
    #assert (torch.allclose(example_out[0], example_out[2])) and (torch.allclose(example_out[1], example_out[3])), str(example_out[0]) + "\n should be equal to:\n" + str(example_out[2])
    
    
    all_times = np.array(all_times)
    print('mean: ', np.mean(all_times))
    print('All: ', all_times)
    print('out: ', example_out)


if __name__ == "__main__":
    test()