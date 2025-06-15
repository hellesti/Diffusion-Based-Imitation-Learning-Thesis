import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F



class ActionHead(nn.Module):
    def __init__(self):
        super().__init__()
    
    def predict_action(self, input):
        raise NotImplementedError
    
    def get_loss(self, input, target):
        raise NotImplementedError
    
    def init_weights(self):
        raise NotImplementedError
    
    def get_optim_groups(self, weight_decay):
        # Add weight decay only to weights, not biases
        no_decay = []
        decay = []

        for pn,p in self.named_parameters():
            if "bias" in pn:
                no_decay.append(p)
            else:
                decay.append(p)

        optim_groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0}
        ]
        return optim_groups

# -------  STANDARD BC LINEAR LAYER ACTION HEAD --------#
class Linear(ActionHead):
    def __init__(self, in_dim, ac_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, ac_dim)
    
    def predict_action(self, input):
        return self.proj(input)
    
    def get_loss(self, input, target):
        pred = self.proj(input)
        return F.mse_loss(pred, target, reduction='none')
    
    def init_weights(self):
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj.bias)
    

# -------- GAUSSIAN MIXTURE MODEL ACTION HEAD -------#

class _MaskedIndependent(D.Independent):
    def masked_log_prob(self, value, mask):
        log_prob = self.base_dist.log_prob(value)
        return (log_prob * mask).sum(-1)


class _MixtureHelper(D.MixtureSameFamily):
    def masked_log_prob(self, x, mask):
        if self._validate_args:
            self._validate_sample(x)
        x, mask = self._pad(x), mask[:, None]
        log_prob_x = self.component_distribution.masked_log_prob(x, mask)  # [S, B, k]
        log_mix_prob = torch.log_softmax(
            self.mixture_distribution.logits, dim=-1
        )  # [B, k]
        return torch.logsumexp(log_prob_x + log_mix_prob, dim=-1)  # [S, B]


class GaussianMixture(ActionHead):
    def __init__(
        self, num_modes, in_dim, ac_dim, min_std=1e-4, tanh_mean=False, deterministic_sampling=False
    ):
        super().__init__()
        self._min_std, self._tanh_mean = min_std, tanh_mean
        self._num_modes = num_modes
        self.ac_dim = ac_dim
        self.deterministic_sampling = deterministic_sampling

        self._mean_net = nn.Linear(in_dim, num_modes * ac_dim)
        self._scale_net = nn.Linear(in_dim, num_modes * ac_dim)
        self._logit_net = nn.Linear(in_dim, num_modes)

    def forward(self, in_repr, zero_std=False):
        B, T = in_repr.shape[:2]
        mean = self._mean_net(in_repr).reshape(B, T, self._num_modes, self.ac_dim)
        scale = self._scale_net(in_repr).reshape(B, T, self._num_modes, self.ac_dim)
        logits = self._logit_net(in_repr).reshape((B, T, self._num_modes))

        # bound the action means and convert scale to std
        if self._tanh_mean:
            mean = torch.tanh(mean)
        std = (
            torch.ones_like(scale) * self._min_std
            if zero_std
            else F.softplus(scale) + self._min_std
        )

        # create num_modes independent action distributions
        ac_dist = D.Normal(loc=mean, scale=std)
        ac_dist = _MaskedIndependent(ac_dist, 1)

        # parameterize the mixing distribution and the final GMM
        mix_dist = D.Categorical(logits=logits)
        gmm_dist = _MixtureHelper(
            mixture_distribution=mix_dist, component_distribution=ac_dist
        )
        return gmm_dist
    
    def predict_action(self, input):
        gmm_dist = self(input)
        if self.deterministic_sampling:
            idx_likeliest_mode = torch.argmax(gmm_dist.mixture_distribution.logits, dim=-1, keepdim=True)
            idx_likeliest_mode = idx_likeliest_mode.expand(-1,-1,self.ac_dim)
            means = gmm_dist.component_distribution.mean
            most_likely_action = torch.gather(means, dim=2, index=idx_likeliest_mode.unsqueeze(2))
            most_likely_action = most_likely_action.squeeze(2)
            return most_likely_action
        else:
            return gmm_dist.sample()
    
    def get_loss(self, input, target):
        gmm_dist = self(input)
        log_prob = gmm_dist.log_prob(target)
        return -log_prob
    
    def get_gmm_dist(self, input):
        return self(input)
    
    def init_weights(self):
        torch.nn.init.normal_(self._logit_net.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self._logit_net.bias)
        torch.nn.init.normal_(self._mean_net.weight, mean=0.0, std=0.02)
        torch.nn.init.uniform_(self._mean_net.bias, a=-0.8, b=0.8)
        torch.nn.init.constant_(self._scale_net.bias, -1.0)
        torch.nn.init.normal_(self._scale_net.weight, mean=0.0, std=0.02)