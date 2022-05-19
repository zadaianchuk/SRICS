import numpy as np
import torch
from torch import nn
from einops import repeat
from rlkit.policies.base import ExplorationPolicy, Policy
from rlkit.torch.core import eval_np
from rlkit.torch.distributions import TanhNormal
from rlkit.torch.networks import Mlp
from rlkit.torch.modules import Attention, preprocess_attention_input
from rlkit.torch.modules import SetGoalAttention, preprocess_set_goal_attention_input, reshape_subgoals


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class TanhGaussianPolicy(Mlp, ExplorationPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            last_layer_init_w=None,
            last_layer_init_b=None,
            initial_log_std_offset=0,
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            last_layer_init_w=last_layer_init_w,
            last_layer_init_b=last_layer_init_b,
            **kwargs
        )
        self.log_std = None
        self.std = std
        self.initial_log_std_offset = initial_log_std_offset
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)

            if last_layer_init_w is None:
                self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            else:
                last_layer_init_w(self.last_fc_log_std.weight)

            if last_layer_init_b is None:
                self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
            else:
                last_layer_init_b(self.last_fc_log_std.bias)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs_np, deterministic=False):
        with torch.no_grad():
            actions = self.get_actions(obs_np[None],
                                       deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        if self.normalizer is not None:
            obs = self.normalizer.normalize(obs)

        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h) + self.initial_log_std_offset
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, pre_tanh_value,
        )


class MakeDeterministic(nn.Module, Policy):
    def __init__(self, stochastic_policy):
        super().__init__()
        self.stochastic_policy = stochastic_policy

    def get_action(self, observation):
        return self.stochastic_policy.get_action(observation,
                                                 deterministic=True)

    def to(self, device):
        super().to(device)
        self.stochastic_policy.to(device)


class AttentionTanhGaussianPolicy(TanhGaussianPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """
    def __init__(self,
                 hidden_sizes,
                 embed_dim,
                 z_size,
                 max_objects,
                 z_goal_size,
                 action_dim,
                 std=None,
                 init_w=1e-3,
                 n_frames=None,
                 attention_kwargs=None,
                 **kwargs):
        if attention_kwargs is None:
            attention_kwargs = {}
        attention = Attention(embed_dim, z_goal_size, z_size,
                              **attention_kwargs)

        super().__init__(hidden_sizes,
                         attention.output_dim + attention.embed_dim,
                         action_dim,
                         std,
                         init_w,
                         **kwargs)

        self.z_goal_size = z_goal_size
        self.z_size = z_size
        self.n_frames = n_frames

        self.attention = attention

    def forward(self, obs,
                reparameterize=True,
                deterministic=False,
                return_log_prob=False):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        x, g, n_objects = preprocess_attention_input(obs,
                                                     self.z_size,
                                                     self.z_goal_size,
                                                     self.n_frames)                         
        goal_embedding = self.attention.embed(g)
        state_embedding = self.attention.embed(x)

        h = self.attention.forward(state_embedding, goal_embedding, n_objects)

        return super().forward(torch.cat((h, goal_embedding.squeeze(1)), dim=1),
                               reparameterize=reparameterize,
                               deterministic=deterministic,
                               return_log_prob=return_log_prob)

    def to(self, device):
        super().to(device)
        self.attention.to(device)


class SetGoalAttentionTanhGaussianPolicy(TanhGaussianPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """
    def __init__(self,
                 hidden_sizes,
                 embed_dim,
                 goal_dims,
                 z_size,
                 max_objects,
                 n_objects, 
                 z_goal_size,
                 action_dim,
                 std=None,
                 init_w=1e-3,
                 n_frames=None,
                 attention_kwargs=None,
                 **kwargs):
        if attention_kwargs is None:
            attention_kwargs = {}
        attention = SetGoalAttention(embed_dim, z_goal_size, z_size, n_queries=n_objects,
                              **attention_kwargs)
        self.n_objects = n_objects
        self.max_objects = max_objects
        self.goal_dims = goal_dims
        inp_dim = attention.output_dim + attention.embed_dim
        super().__init__(hidden_sizes,
                         inp_dim,
                         action_dim,
                         std,
                         init_w,
                         **kwargs)

        self.z_goal_size = z_goal_size
        self.full_z_goal_size = sum(self.goal_dims)
        self.z_size = z_size
        self.n_frames = n_frames

        self.attention = attention

    def forward(self, obs,
                reparameterize=True,
                deterministic=False,
                return_log_prob=False):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        x, g, n_objects = preprocess_set_goal_attention_input(obs, 
                                                              self.goal_dims,
                                                              self.z_size,
                                                              self.full_z_goal_size,
                                                              self.n_frames)
        g, _, mask_subgoal = reshape_subgoals(g, 
                                   self.goal_dims, 
                                   self.n_objects, 
                                   self.max_objects, 
                                   self.z_goal_size, 
                                   self.z_size
                                   )
        goal_embedding = self.attention.goal_embed(g)
        bs = goal_embedding.shape[0]
        state_embedding = self.attention.embed(x)
        h = self.attention.forward(state_embedding, goal_embedding, n_objects, mask_subgoal)
        mask_subgoal = repeat(mask_subgoal, 'b n d -> b n (repeat d)', repeat=self.attention.embed_dim)
        goal_embedding = goal_embedding[~mask_subgoal].reshape(bs, -1)
        assert goal_embedding.shape[1] == self.attention.embed_dim, "Several subgoals are not implemented"
        return super().forward(torch.cat((h, goal_embedding), dim=1),
                               reparameterize=reparameterize,
                               deterministic=deterministic,
                               return_log_prob=return_log_prob)

    def to(self, device):
        super().to(device)
        self.attention.to(device)


class DeepSetTanhGaussianPolicy(TanhGaussianPolicy):
    def __init__(self,
                 hidden_sizes,
                 key_query_size,  # unused
                 z_size,
                 max_objects,
                 z_goal_size,
                 value_size,  # unused
                 action_dim,
                 embed_dim,
                 aggregation_dim,
                 std=None,
                 init_w=1e-3,
                 n_frames=None,
                 **kwargs):
        super().__init__(hidden_sizes,
                         aggregation_dim + embed_dim,
                         action_dim,
                         std,
                         init_w,
                         **kwargs)
        self.z_goal_size = z_goal_size
        self.z_size = z_size
        self.n_frames = n_frames

        self.embedding = nn.Linear(z_size, embed_dim)
        self.pre_aggregation = nn.Sequential(nn.Linear(embed_dim,
                                                       aggregation_dim),
                                             nn.ReLU(),
                                             nn.Linear(aggregation_dim,
                                                       aggregation_dim))
        for layer in (self.embedding,
                      self.pre_aggregation[0],
                      self.pre_aggregation[2]):
            if 'hidden_init' in kwargs:
                kwargs['hidden_init'](layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, obs,
                reparameterize=True,
                deterministic=False,
                return_log_prob=False):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        x, g, n_objects = preprocess_attention_input(obs,
                                                     self.z_size,
                                                     self.z_goal_size,
                                                     self.n_frames)
        goal_embedding = self.embedding(g)
        state_embedding = self.embedding(x)

        h = self.pre_aggregation(state_embedding).sum(dim=1)

        return super().forward(torch.cat((h, goal_embedding.squeeze(1)), dim=1),
                               reparameterize=reparameterize,
                               deterministic=deterministic,
                               return_log_prob=return_log_prob)

    def to(self, device):
        super().to(device)
