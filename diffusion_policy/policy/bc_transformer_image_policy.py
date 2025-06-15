if __name__ == "__main__":
    import sys
    import os
    import pathlib
    import time
    import dill
    import hydra
    import numpy as np
    import functools
    import zarr

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

    from diffusion_policy.workspace.base_workspace import BaseWorkspace
    from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset

from typing import Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.history_conditioned.bc_transformer import BCTransformer
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.model.history_conditioned.action_heads import ActionHead, GaussianMixture


class BCTransformerImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            model_partial,
            action_head: ActionHead,
            history_length,
            action_dim,
            n_action_steps,
            # image
            crop_shape=(76, 76),
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,

            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # create transformer model
        obs_feature_dim = obs_encoder.output_shape()[0]
        self.model = model_partial(cond_dim = obs_feature_dim)


        self.obs_encoder = obs_encoder

        self.action_head = action_head
        self.action_head.init_weights()

        self.normalizer = LinearNormalizer()
        self.history_length = history_length
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.obs_history:torch.Tensor = None
        self.obs_feature_dim = obs_feature_dim
        self.kwargs = kwargs
    
    # ========= inference  ============

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

    def batch_inference(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        obs_dict: must include "obs" key and "hla" (high level actions) key and "tau" (timesteps since high-level actions recieved) key 
        result: must include "action" key
        """

        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, H = value.shape[:2]
        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        # reshape back to B, To, Do
        cond = nobs_features.reshape(B, H, -1)
        #cond = cond[:,:,:-2] Use only image features

        Da = self.action_dim

        # run model
        npred = self.model(cond,**self.kwargs)
        npred = self.action_head.predict_action(npred)

        # unnormalize prediction
        naction_pred = npred[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        return action_pred
    
    def reset_history(self):
        self.obs_history = None


    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, transformer_weight_decay: float, obs_encoder_weight_decay: float, learning_rate: float, betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:

        optim_groups = self.model.get_optim_groups(weight_decay=transformer_weight_decay)
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        optim_groups.extend(self.action_head.get_optim_groups(transformer_weight_decay))
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

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
        
        Da = self.action_dim

        
        # Predict low-level actions
        pred = self.model(cond)

        target = naction

        loss = self.action_head.get_loss(pred, target)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss


def test():

    shape_meta = {'obs': {
        'image': {
            'shape': [3,96,96],
            'type': 'rgb'
        },
        'agent_pos': {
            'shape': [2],
            'type': 'low_dim'
        }
    },
    'action': {
        'shape': [2]
    }
    }

    h_transformer_partial = functools.partial(BCTransformer,
        action_horizon= 1,
        history_length= 16,
        n_layer= 8,
        n_head= 4,
        n_emb= 256,
        p_drop_emb= 0.0,
        p_drop_attn = 0.3,
        causal_attn=True)
    
    action_head = GaussianMixture(3, 256, 2)

    policy = BCTransformerImagePolicy(
        shape_meta=shape_meta,
        model_partial= h_transformer_partial,
        action_head=action_head,
        history_length = 16,
        obs_dim = 18, 
        action_dim = 2, 
        n_action_steps = 1
        )
    

    # We need a dataset object to set up normalizer
    zarr_path = 'data/demos/pusht/pusht_moving_goal/demo.zarr'
    dataset = PushTImageDataset(zarr_path=zarr_path)
    normalizer = dataset.get_normalizer()
    policy.set_normalizer(normalizer)
    root = zarr.open('data/demos/pusht/pusht_moving_goal/demo.zarr')
    root_numpy = {'img': np.array(root.data.img, dtype=np.float32),
                  'agent_pos': np.array(root.data.state, dtype=np.float32),
                  'action': np.array(root.data.action, dtype=np.float32)}
    
    root_numpy2 = {'obs': {
                'image': np.array(root.data.img, dtype=np.float32),
                'agent_pos': np.array(root.data.state[:,:2], dtype=np.float32)
            },
            'action': np.array(root.data.action, dtype=np.float32)
            }
    device = torch.device("cuda:0")
    #root_gpu = dict_apply(root_numpy, lambda x: torch.from_numpy(x).to(device, dtype=torch.float32))
    policy.to(device=device, dtype=torch.float32)
    policy.eval()

    image = torch.zeros(256,16,3,96,96).to(device=device, dtype=torch.float32)
    agpos = torch.ones(256,16,2).to(device=device, dtype=torch.float32)
    action = torch.zeros(256,16,2).to(device=device, dtype=torch.float32)
    batch = {'obs': {
        'image': image,
        'agent_pos': agpos
    },
    'action': action
    }

    with torch.no_grad():
        pred = policy.predict_action(batch['obs'])
        print(pred['action'].shape)
        batch_pred = policy.batch_inference(batch['obs'])
        print(batch_pred.shape)
        loss = policy.compute_loss(batch)

if __name__ == "__main__":
    test()