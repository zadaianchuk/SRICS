import torch

from rlkit.torch.gnn.dynnet import DynNet
from rlkit.envs.gt_wrapper import GTWrappedEnv


class GTWrappedGNNEnv(GTWrappedEnv):
    def __init__(self, env, dynnet_path, dynnet_params, *args, **kwargs):
        self.dynnet = DynNet(**dynnet_params)
        self.dynnet.load(dynnet_path)
        super().__init__(env, *args, **kwargs)

    def reset(self):
        super().reset()
        self.dynnet.reset()

    def step(self, action):
        old_obs = self._prev_obs.copy()
        z_where = old_obs['state_observation'].reshape(-1, 2)
        z_where_t, action_t = torch.Tensor(z_where[None]).cuda(), torch.Tensor(action[None]).cuda()
        weights, _ = self.dynnet.encode(z_where_t, action_t)
        obs, reward, done, info = super().step(action)
        obs['weights'] = weights
        return obs, reward, done, info
