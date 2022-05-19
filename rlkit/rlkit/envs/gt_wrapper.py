import copy
from typing import List, Dict, Tuple
import numpy as np
from gym.spaces import Box
import gym.spaces 
import rlkit.torch.pytorch_util as ptu
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict
from rlkit.envs.wrappers import ProxyEnv
from rlkit.envs.util import get_subgoals, get_parents
from collections import namedtuple

import networkx as nx

DictObs = Dict[str, np.ndarray]
DictGoal = Dict[str, np.ndarray]
DictInfo = Dict[str, float]


class GTWrappedEnv(ProxyEnv, MultitaskEnv):
    """This class wraps environment with a GT stuctured representations.
    This class adheres to the "Silent Multitask Env" semantics: on reset,
    it resamples a goal.
    """
    def __init__(self,
                 wrapped_env,
                 z_where_dim,
                 z_depth_dim, 
                 max_n_objects,
                 sub_task_horizon,
                 solved_goal_threshold=0.05,
                 sample_from_true_prior=True,
                 z_goal_dim=None,
                 reward_params=None,
                 goal_sampling_mode="z_where_prior",
                 norm_order=2,
                 done_on_success=False,
                 track_success_rates=True,
                 goal_prioritization=False,
                 success_rate_coeff=0.95):
        if reward_params is None:
            reward_params = dict()
        super().__init__(wrapped_env)
        self.device = ptu.device
        self.max_n_objects = max_n_objects # used in representation shaping
        self.n_objects_max = self.wrapped_env.n_objects_max + 1 # from env
        self.z_what_dim = self.n_objects_max
        self.z_where_dim = z_where_dim
        self.z_depth_dim = z_depth_dim
        self.sub_task_horizon = sub_task_horizon
        self.sample_from_true_prior = sample_from_true_prior
        self.reward_params = reward_params
        self.eps = self.reward_params.get("eps", 10**(-5)) 
        self.reward_type = self.reward_params.get("type", 'object_distance')
        self.norm_order = self.reward_params.get("norm_order", norm_order)
        if self.reward_type in ('sparse', 'pos_sparse'):
            self.success_threshold = self.reward_params.get('threshold')
            solved_goal_threshold = self.success_threshold
            assert self.success_threshold is not None
        self.solved_goal_threshold = solved_goal_threshold
        self.reward_min_variance = self.reward_params.get("min_variance", 0)
        self.z_dim = (self.z_what_dim + self.z_where_dim + self.z_depth_dim)*self.max_n_objects + 1
        if z_goal_dim is None:
            self.z_goal_dim = self.z_what_dim + self.z_where_dim + 1
        else: 
            self.z_goal_dim = z_goal_dim
        z_what_space = Box(
            -10 * np.ones(self.z_what_dim),
            10 * np.ones(self.z_what_dim),
            dtype=np.float32,
        )
        z_where_space = Box(
            -1 * np.ones(self.z_where_dim),
            1 * np.ones(self.z_where_dim),
            dtype=np.float32,
        )
        z_depth_space = Box(
            -1 * np.ones(self.z_depth_dim),
            1 * np.ones(self.z_depth_dim),
            dtype=np.float32,
        )
        z_space = Box(
            -10 * np.ones(self.z_dim),
            10 * np.ones(self.z_dim),
            dtype=np.float32,
        )
        z_goal_space = Box(
            -10 * np.ones(self.z_goal_dim),
            10 * np.ones(self.z_goal_dim),
            dtype=np.float32,
        )

        spaces = copy.copy(self.wrapped_env.observation_space.spaces)
        spaces['z_what'] = z_what_space
        spaces['z_where'] = z_where_space
        spaces['z_depth'] = z_depth_space
        spaces['latent_obs_vector'] = z_space
        spaces['goal_vector'] = z_goal_space
        spaces['desired_goal'] = z_where_space
        spaces['achieved_goal'] = z_where_space
        self.observation_space = gym.spaces.Dict(spaces)
        self.observation_space.flat_dim = (self.z_what_dim + self.z_where_dim + self.z_depth_dim)*self.max_n_objects + self.z_what_dim + self.z_where_dim + 2
        self.match_thresh = 0.01
        self.current_obs = None
        self.desired_goal = None
        self._initial_obs_vector = None
        self._custom_goal_sampler = None
        self._goal_sampling_mode = goal_sampling_mode
        self._done_on_success = done_on_success
        self._track_success_rates = track_success_rates
        if self._track_success_rates:
            self._success_rate_coeff = success_rate_coeff
            self._successes = {}
            self._attempts = {}
        self._goal_prioritization = goal_prioritization
        if self._goal_prioritization:
            assert self._track_success_rates

        self.t = 0
        self.reset_count = 0
        self.k_in = None
        self.env_params = dict(z_where_dim=z_where_dim,
                               z_depth_dim=z_depth_dim,
                               max_n_objects=max_n_objects,
                               sub_task_horizon=sub_task_horizon,
                               sample_from_true_prior=sample_from_true_prior,
                               reward_params=reward_params,
                               goal_sampling_mode=goal_sampling_mode,
                               norm_order=norm_order,
                               solved_goal_threshold=solved_goal_threshold)

    def reset(self):
        self.reset_count += 1
        self.z_what = self._get_z_what()
        self.wrapped_env.reset()
        zero_action = np.zeros((2,))
        obs, _, _, _ = self.wrapped_env.step(zero_action)
        self.n_objects = self.wrapped_env.n_objects + 1
        self.obj_idx = np.random.choice(self.n_objects_max, self.n_objects, replace=False)
        self._initial_obs_vector = self._get_representation(obs)['latent_obs_vector']
        goal, obs = self.sample_goal(obs)
        self.set_goal(goal)
        obs = self._update_obs(obs)
        self.set_current_obs(obs)
        self.t = 0
        self.k_in = None
        
        return obs

    def _get_z_what(self):
        z_what = np.eye(self.z_what_dim)
        return z_what

    def get_goal_from_gt(self, states, mode="z_where_prior"):
        states = states.reshape(-1, 2)[:self.n_objects]
        z_what = self.z_what
        if mode == "z_where_prior":
            z_where = self._sample_z_where_prior(self.n_objects)
        else:
            z_where = states

        goal_vectors = np.concatenate((np.arange(self.n_objects)[:, None],
                                       z_what, z_where), axis=1)
        if (mode == "z_where_prior" and self._goal_prioritization
                and self.reset_count > 10):
            probs = self._get_sampling_probs(z_what)
            k = np.random.choice(self.n_objects, p=probs)
        else:
            k = np.random.randint(self.n_objects)
        goal_vector = goal_vectors[k]
        z_what_k = z_what[k]
        z_where_k = z_where[k]
        goal = {"goal_vector": goal_vector,
                "goal_vectors": goal_vectors,
                "z_what_goals": z_what,
                "z_where_goals": z_where,
                "z_what_goal": z_what_k,
                "z_where_goal": z_where_k,
                "idx_goal": k}
        return goal

    def sample_goal(self, obs):
        if self._goal_sampling_mode == 'z_where_prior':
            latent_goal = self.get_goal_from_gt(obs['state_observation'])
        elif self._goal_sampling_mode == 'reset_of_env':
            goal_dict = self.wrapped_env.get_goal()
            states = goal_dict["state_desired_goal"]
            latent_goal = self.get_goal_from_gt(states, mode=self._goal_sampling_mode) # random_goal from goal state
        elif self._goal_sampling_mode == 'current_state':
            latent_goal = self.get_goal_from_gt(obs['state_observation'], mode=self._goal_sampling_mode)
        else:
            raise RuntimeError("Invalid: {}".format(self._goal_sampling_mode))
        return latent_goal, obs

    def sample_goals(self, batch_size, initial_goals):
        z_where_prior = self._sample_z_where_prior(batch_size)
        initial_goals[:, -self.z_where_dim:] = z_where_prior

        return initial_goals

    def _sample_z_where_prior(self, batch_size):
        space = self.wrapped_env.observation_space.spaces['desired_goal']
        n = space.sample().reshape(-1, 2).shape[0]
        #this assumes that all the objects are the same, or that we don't have conditional distribution. 
        k = np.random.randint(1, n) #1 if we want to use only objects coordinates. 
        z_wheres = np.stack([space.sample().reshape(-1, 2)[k]
                             for _ in range(batch_size)])
        return z_wheres
    def step(self, action):
        prev_obs = copy.copy(self.current_obs)
        obs, reward, done, info = self.wrapped_env.step(action)
        self.t += 1
        new_obs = self._update_obs(obs)
        self._update_info(info, new_obs)
        reward = self.compute_reward(
            {'latent_obs_vector': prev_obs['latent_obs_vector']},
            {'latent_obs_vector': new_obs['latent_obs_vector'],
             'goal_vector': new_obs['goal_vector']}
        )
        if (self._track_success_rates and self.t % self.sub_task_horizon == 0):
            self._update_success_rates(new_obs, reward)

        if self._goal_sampling_mode == 'reset_of_env':
            # meta policy part
            if (self.t % self.sub_task_horizon == 0) and (self.t != 0):
                k_init = new_obs["idx_goal"]
                self.update_goal()
                new_obs = self._update_obs(obs, action)
                new_reward = self.compute_reward(
                    {'latent_obs_vector': prev_obs['latent_obs_vector']},
                    {'latent_obs_vector': new_obs['latent_obs_vector'],
                     'goal_vector': new_obs['goal_vector']}
                )
                while self._compute_success_from_reward(new_reward):
                    self.update_goal()
                    new_obs = self._update_obs(obs, action)
                    new_reward = self.compute_reward(
                        {'latent_obs_vector': prev_obs['latent_obs_vector']}, 
                        {'latent_obs_vector': new_obs['latent_obs_vector'],
                         'goal_vector': new_obs['goal_vector']}
                    )
                    if new_obs["idx_goal"] == k_init:
                        done = True
                        break
        else:
            done = self.compute_done(done, reward)
        self.set_current_obs(new_obs)
        #TODO: why old reward is used here, not updated one?
        return new_obs, reward, done, info

    def _compute_success_from_reward(self, reward):
        if self.reward_type == 'object_distance':
            return (np.abs(reward) < self.solved_goal_threshold)

    def compute_done(self, done, reward):
        return done

    def update_goal(self):
        goal = self.desired_goal
        goal_vectors = goal["goal_vectors"]
        z_where = goal["z_where_goals"]
        z_what = goal["z_what_goals"]
        n_objects = goal_vectors.shape[0]
        k = goal["idx_goal"]
        k = (k+1) % n_objects
        goal_vector = goal_vectors[k]
        z_what_k = z_what[k]
        z_where_k = z_where[k]
        goal = {"goal_vectors": goal_vectors,
                "z_what_goals": z_what,
                "z_where_goals": z_where,
                "goal_vector": goal_vector,
                "z_what_goal": z_what_k,
                "z_where_goal": z_where_k,
                "idx_goal": k}
        self.desired_goal = goal

    def _get_representation(self, obs):
        z_where = obs['state_observation'].reshape(-1, 2)
        z_where = z_where[:self.n_objects, :]
        z_what = self.z_what
        z_depth = np.zeros((self.n_objects, 1))
        representation = dict(z_what=z_what, z_where=z_where, z_depth=z_depth)
        obs_vector = dict2vector(representation, self.max_n_objects)
        representation["latent_obs_vector"] = obs_vector

        return representation

    def _update_obs(self, obs, action=None):
        representation = self._get_representation(obs)
        obs = {**obs, **representation, **self.desired_goal}
        return obs

    def _update_info(self, info, obs):
        k = obs["idx_goal"]
        z_where = obs["z_where"][k]
        z_where_goal = obs["z_where_goal"]
        dist = z_where - z_where_goal
        info["z_where_dist"] = np.linalg.norm(dist, ord=self.norm_order)

    def _update_moving_average(self, key, success, coeff):
        successes = self._successes.get(key, 0)
        self._successes[key] = success + coeff * successes
        attempts = self._attempts.get(key, 0)
        self._attempts[key] = 1 + coeff * attempts

    def _update_success_rates(self, obs, reward):
        success = self._compute_success_from_reward(reward)
        z_what_goal = np.argmax(self.desired_goal["z_what_goal"])
        self._update_moving_average(z_what_goal,
                                    1.0 if success else 0.0,
                                    self._success_rate_coeff)

    def _get_sampling_probs(self, z_whats):
        rates = []
        for z_what in z_whats:
            i = np.argmax(z_what)
            if i not in self._successes:
                rates.append(0)
            else:
                rates.append(self._successes[i] / self._attempts[i])

        fail_rates = 1 - np.array(rates)
        probs = (fail_rates + 0.05) / np.sum(fail_rates + 0.05)

        return probs

    """
    Multitask functions
    """

    def get_goal(self):
        return self.desired_goal

    def compute_reward(self, obs, next_obs):
        obs = {
            k: v[None] for k, v in obs.items()
        }
        next_obs = {
            k: v[None] for k, v in next_obs.items()
        }
        return self.compute_rewards(obs, next_obs)[0]

    def compute_rewards(self, obs, next_obs):
        latent_obs_vector = next_obs['latent_obs_vector']
        goal_vector = next_obs['goal_vector']
        k = goal_vector[:, 0]
        desired_goals = goal_vector[:, -self.z_where_dim:]
        achieved_goals = get_z_where_from_obs(latent_obs_vector,
                                              k.astype(np.int),
                                              self.max_n_objects,
                                              self.z_what_dim,
                                              self.z_where_dim)
        dist = np.linalg.norm(desired_goals - achieved_goals,
                              ord=self.norm_order, axis=1)
        if self.reward_type == 'object_distance':
            return -dist
        elif self.reward_type == 'sparse':
            return -1.0 * (dist >= self.success_threshold)
        elif self.reward_type == 'pos_sparse':
            return (dist < self.success_threshold).astype(np.float32)
        else:
            raise NotImplementedError('reward_type {}'
                                      .format(self.reward_type))

    def match_goals(self, latent_obs, z_goal):
        match_idx = z_goal[:, 0].astype(np.int)
        if len(match_idx) == 1 and len(latent_obs) != 1:
            match_idx = np.tile(match_idx, (len(latent_obs), 1))

        match = np.ones_like(match_idx, dtype=np.bool)

        return match, match_idx

    def extract_achieved_goals(self, latent_obs_vector, future_goal_goals, obj_indices):
        bs = len(latent_obs_vector)
        latent_obs = latent_obs_vector[:, 1:]

        zs = latent_obs.reshape((bs, self.max_n_objects, -1))

        goal_objects = zs[np.arange(bs), obj_indices, :self.z_what_dim + self.z_where_dim]

        goals = np.concatenate((obj_indices[:, None], goal_objects), axis=1)

        return goals

    @property
    def goal_dim(self):
        return self.z_where_dim

    def set_goal(self, goal):
        """
        Assume goal contains both image_desired_goal and any goals required for wrapped envs

        :param goal:
        :return:
        """
        self.desired_goal = goal
    
    def set_current_obs(self, obs):
        """
        Assume obs contains dic with all the info required for wrapped envs

        :param goal:
        :return:
        """
        self.current_obs = obs

    def get_diagnostics(self, paths, **kwargs):
        statistics = self.wrapped_env.get_diagnostics(paths, **kwargs)
        for stat_name_in_paths in ["z_where_dist"]:
            stats = get_stat_in_paths(paths, 'env_infos', stat_name_in_paths)
            statistics.update(create_stats_ordered_dict(
                stat_name_in_paths,
                stats,
                always_show_all_stats=True,
            ))
            final_stats = [s[-1] for s in stats]
            statistics.update(create_stats_ordered_dict(
                "Final " + stat_name_in_paths,
                final_stats,
                always_show_all_stats=True,
            ))

        if self._track_success_rates:
            for i in range(self.n_objects_max):
                attempts = self._attempts.get(i, 0)
                success_rate = self._successes.get(i, 0) / max(attempts, 1)
                statistics["success_rate_{}".format(i)] = success_rate

            sampling_probs = self._get_sampling_probs(np.eye(self.z_what_dim))
            for i, prob in enumerate(sampling_probs):
                statistics["goal_sampling_prob_{}".format(i)] = prob

        return statistics

    """
    Other functions
    """
    @property
    def goal_sampling_mode(self):
        return self._goal_sampling_mode

    @goal_sampling_mode.setter
    def goal_sampling_mode(self, mode):
        assert mode in [
            'z_where_prior',
            'env',
            'reset_of_env',
            'current_state'
        ], "Invalid env mode"
        self._goal_sampling_mode = mode


GoalVectorDims = namedtuple('GoalVectorDims', ['goal_index', 
                                               'goal_index_onehot', 
                                               'parents_index_onehot', 
                                               'goal_vector', 
                                               'init_obs_vector'])


class GTRelationalWrappedEnv(GTWrappedEnv):

    def __init__(self, env, env_id: str, 
                            n_meta: int, 
                            ordered_evaluation: bool,
                            z_where_dim: int, 
                            max_n_objects: int, 
                            alpha_sel: float, 
                            z_depth_dim: int, 
                            graph_path: str,
                            *args, **kwargs) -> None:
        n_objects = env.n_objects + 1 #n push objects + arm
        self.ordered_evaluation = ordered_evaluation
        self.use_prev = True
        self.use_current_goals = True
        self.weights = np.load(graph_path)
        self.subgoals = get_subgoals(self.weights)
        
        print(self.subgoals)
        self.parents = get_parents(self.subgoals, N=n_objects)
        self.alpha_sel = alpha_sel 
        z_what_dim = n_objects
        self.n_objects = n_objects
        self.z_subgoal_dim = z_what_dim + z_where_dim
        self.z_dim = (z_what_dim + z_where_dim + z_depth_dim) * max_n_objects + 1
        self.goal_dims = GoalVectorDims(goal_index=1, 
                                        goal_index_onehot=n_objects, 
                                        parents_index_onehot=n_objects, 
                                        goal_vector=self.z_subgoal_dim * n_objects,
                                        init_obs_vector=self.z_dim)
        self.goal_idx_slices = get_idx(self.goal_dims)
        self.n_meta = n_meta
        self.n_meta_done = 0 
        print(f"Eval attempts: {self.n_meta}" )
        super().__init__(wrapped_env=env, 
                         max_n_objects=max_n_objects,
                         z_depth_dim=z_depth_dim,
                         z_where_dim=z_where_dim,
                         z_goal_dim=sum(self.goal_dims), *args, **kwargs)
        self.env_params['env_id'] = env_id
        self.env_params['n_meta'] = n_meta
        self.env_params['alpha_sel'] = alpha_sel
        self.env_params['ordered_evaluation'] = ordered_evaluation
        print(f"{self.n_subgoals} subgoals was found")



    @property
    def z_object_dim(self) -> int: 
        return self.z_what_dim + self.z_where_dim + self.z_depth_dim # for depth dim  
    
    @property
    def n_subgoals(self) -> int:  
        return len(self.subgoals)


    def reset(self):
        self.n_meta_done = 0
        return super().reset()

    def step(self, action: np.ndarray) -> Tuple[DictObs, float, bool, DictInfo]:
        prev_obs = copy.copy(self.current_obs)
        obs, reward, done, info = self.wrapped_env.step(action)
        self.t += 1
        new_obs = self._update_obs(obs)
        self._update_info(info, prev_obs, new_obs)
        reward, dist_reward = self.compute_reward(
            {'latent_obs_vector': prev_obs['latent_obs_vector']},
            {'latent_obs_vector': new_obs['latent_obs_vector'],
             'goal_vector': new_obs['goal_vector']}
        )
        if (self._track_success_rates and self.t % self.sub_task_horizon == 0):
            self._update_success_rates(new_obs, dist_reward)
        if self._goal_sampling_mode == 'reset_of_env':
            # meta policy part
            if (self.t % self.sub_task_horizon == 0) and (self.t != 0):
                self.n_meta_done += 1
                k_init = new_obs["idx_goal"]
                self.update_goal(init_obs=new_obs['latent_obs_vector'])
                new_obs = self._update_obs(obs, action)
                _, new_dist_reward = self.compute_reward(
                        {'latent_obs_vector': prev_obs['latent_obs_vector']}, 
                        {'latent_obs_vector': new_obs['latent_obs_vector'],
                        'goal_vector': new_obs['goal_vector']}
                )
                if (self.n_meta - self.n_subgoals)  > self.n_meta_done:
                    while self._compute_success_from_reward(new_dist_reward):
                        self.update_goal(init_obs=new_obs['latent_obs_vector'])
                        new_obs = self._update_obs(obs, action)
                        _, new_dist_reward = self.compute_reward(
                            {'latent_obs_vector': prev_obs['latent_obs_vector']}, 
                            {'latent_obs_vector': new_obs['latent_obs_vector'],
                            'goal_vector': new_obs['goal_vector']}
                        )
                        if new_obs["idx_goal"] == k_init:
                            done = True
                            break
                else:
                    while self._compute_success_from_reward(new_dist_reward):
                        if self.t < self.n_meta * self.sub_task_horizon:
                            self.t += self.sub_task_horizon #update time also, as we try once
                            self.n_meta_done += 1
                        self.update_goal(init_obs=new_obs['latent_obs_vector'])
                        new_obs = self._update_obs(obs, action)
                        _, new_dist_reward = self.compute_reward(
                            {'latent_obs_vector': prev_obs['latent_obs_vector']}, 
                            {'latent_obs_vector': new_obs['latent_obs_vector'],
                            'goal_vector': new_obs['goal_vector']}
                        )
                        if new_obs["idx_goal"] == k_init:
                            done = True
                            break
                if self.t >= self.n_meta * self.sub_task_horizon:
                    done = True


        else:
            done = self.compute_done(done, reward)
        self.set_current_obs(new_obs)
        return new_obs, reward, done, info

    def get_subgoal_from_gt(self, states: np.ndarray, mode: str ="z_where_prior") -> DictGoal:
        states = states.reshape(-1, 2)[:self.n_objects]
        z_what = self.z_what
        if mode == "z_where_prior":
            z_where = self._sample_z_where_prior(self.n_objects)
        else:
            z_where = states

        goal_vectors = np.concatenate([z_what, z_where], axis=1)
        if (mode == "z_where_prior" and self._goal_prioritization
                and self.reset_count > 10):
            probs = self._get_sampling_probs()
            k = np.random.choice(self.n_subgoals, p=probs)
        elif mode == "z_where_prior":
            goals = self.subgoals
            k = np.random.choice(len(goals))
        else:
            goals = self.subgoals
            k = np.random.choice(len(goals))
        subgoal_idx = np.array(self.subgoals[k])[None]
        subgoal_idx_fixed = self._to_fixed_vector(subgoal_idx)
        
        subgoal_vectors = np.concatenate([subgoal_idx_fixed, goal_vectors.reshape(1,-1)], axis=1).squeeze()
        full_goal = np.concatenate([subgoal_vectors, self._initial_obs_vector])
        goal = {"goal_vector": full_goal,
                "idx_goal": k}
        return goal

    def sample_goal(self, obs: DictObs) -> Tuple[DictGoal, DictObs]:
        if self._goal_sampling_mode == 'z_where_prior':
            latent_goal = self.get_subgoal_from_gt(obs['state_observation'])
        elif self._goal_sampling_mode == 'reset_of_env':
            goal_dict = self.wrapped_env.get_goal()
            states = goal_dict["state_desired_goal"]
            latent_goal = self.get_subgoal_from_gt(states, mode=self._goal_sampling_mode) # random_goal from goal state
        elif self._goal_sampling_mode == 'current_state':
            latent_goal = self.get_subgoal_from_gt(obs['state_observation'], mode=self._goal_sampling_mode)
        else:
            raise RuntimeError("Invalid: {}".format(self._goal_sampling_mode))
        return latent_goal, obs

    def sample_goals(self, batch_size: int, initial_goals: np.ndarray) -> np.ndarray:
        z_where_prior = self._sample_z_where_prior(batch_size)
        # we need to add goals to the right positions in the goal vector
        subgoal_index = initial_goals[:, self.goal_idx_slices.goal_index.start]
        start =  self.goal_idx_slices.goal_vector.start + subgoal_index * self.z_subgoal_dim + self.z_what_dim
        end = self.goal_idx_slices.goal_vector.start + (subgoal_index + 1) * self.z_subgoal_dim
        idx = np.array(range(initial_goals.shape[1]))[None,:]
        start = start[:, None]
        end = end[:, None]
        mask = (idx>=start) & (idx<end)
        initial_goals[mask] = z_where_prior.reshape(-1)

        post = initial_goals[0, int(start[0]):int(end[0])]
        pr = z_where_prior[0]
        assert (post == pr).all(), f"{pr} {post}"

        return initial_goals
    
    def update_goal(self, init_obs: np.ndarray) -> None:
        k = self.desired_goal["idx_goal"]
        if self.ordered_evaluation and (self.n_meta - self.n_subgoals) == self.n_meta_done:
            k = 0 
        else:
            k = (k+1) % self.n_subgoals
        subgoal_idx = np.array(self.subgoals[k])[None]
        subgoal_idx_fixed = self._to_fixed_vector(subgoal_idx)
        self.desired_goal["goal_vector"][self.goal_idx_slices.goal_index.start:self.goal_idx_slices.parents_index_onehot.stop] = subgoal_idx_fixed
        self.desired_goal["goal_vector"][self.goal_idx_slices.init_obs_vector] = init_obs
        self.desired_goal["idx_goal"] = k
    
    def _compute_success_from_reward(self, reward):
        continious_reward_types = ['object_distance', 'selectivity', 'frac_sel', 'frac_sel_goal']
        if self.reward_type in continious_reward_types:
            return (np.abs(reward) < self.solved_goal_threshold)
        else:
            raise NotImplementedError(f"reward_type: {self.reward_type}")


    def _update_success_rates(self, obs, reward):
        success = self._compute_success_from_reward(reward)
        z_what_goal = int(self.desired_goal["goal_vector"][self.goal_idx_slices.goal_index])
        self._update_moving_average(z_what_goal,
                                    1.0 if success else 0.0,
                                    self._success_rate_coeff)

    def match_goals(self, latent_obs: np.ndarray, z_goal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        match_idx = z_goal[:, self.goal_idx_slices.goal_index.start].astype(np.int)
        if len(match_idx) == 1 and len(latent_obs) != 1:
            match_idx = np.tile(match_idx, (len(latent_obs), 1))

        match = np.ones_like(match_idx, dtype=np.bool)

        return match, match_idx

    def _to_fixed_vector(self, subgoal_idx: List[int]) -> np.ndarray:
        bs, n  = subgoal_idx.shape
        assert bs == 1, f"{bs}, this method should be used for one subgoal"

        subgoal_idx_last = subgoal_idx[:, (n-1):]
        subgoal_idx_parents= subgoal_idx[:, :(n-1)]
        subgoal_last_idx = np.zeros((1, self.n_objects))
        subgoal_parents_idx= np.zeros((1, self.n_objects))
        subgoal_last_idx[:, subgoal_idx_last] = 1
        subgoal_parents_idx[:, subgoal_idx_parents] = 1
        subgoal_idx_fixed = np.concatenate([subgoal_idx_last, subgoal_last_idx, subgoal_parents_idx], axis=1)
        return subgoal_idx_fixed

    def get_z_wheres_from_goal(self, goal_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Returns have shape Nx2 
        '''
        assert len(goal_vector.shape) == 1
        goal_vector_subgoal = goal_vector[self.goal_idx_slices.goal_vector]
        goal_vector_subgoal = goal_vector_subgoal.reshape((-1, self.z_subgoal_dim))
        goal_z_wheres = goal_vector_subgoal[:, self.z_what_dim:]

        init_obs = goal_vector[self.goal_idx_slices.init_obs_vector][1:]
        init_obs = init_obs.reshape((-1, self.z_object_dim))[:self.n_objects]
        init_obs_z_wheres = init_obs[:, self.z_what_dim:(self.z_what_dim + self.z_where_dim)]
        return goal_z_wheres, init_obs_z_wheres

    def _update_info(self, info: DictInfo, prev_obs: DictObs, obs: DictObs) -> None:
        k = obs["idx_goal"]
        subgoal_idx = self.subgoals[k][-1:]
        non_parents_idx = list(set(range(self.n_objects)) - set(self.subgoals[k]))
        
        z_where = obs["z_where"]
        z_where_part = z_where[subgoal_idx]
        z_where_goal, z_where_init_obs = self.get_z_wheres_from_goal(obs["goal_vector"])
        if not self.use_prev:
            z_where_prev = z_where_init_obs
        else:
            z_where_prev = prev_obs["z_where"]
        z_where_goal = z_where_goal[subgoal_idx]
        dist = z_where_part - z_where_goal
        z_where_dist = np.linalg.norm(dist, ord=self.norm_order)
        info["z_where_dist"] = np.linalg.norm(dist, ord=self.norm_order)
        z_where_prev_subgoal = z_where_prev[subgoal_idx]
        z_where_prev_non_parents = z_where_prev[non_parents_idx]
        z_where_part_non_parents = z_where[non_parents_idx]



        if non_parents_idx:
            dist_non_parents = z_where_part_non_parents - z_where_prev_non_parents
            dist_non_parents = np.linalg.norm(dist_non_parents, ord=self.norm_order, axis=1).mean()
            info["z_where_non_parents_dist"] = dist_non_parents

            prev_dist = np.linalg.norm(z_where_part - z_where_prev_subgoal, ord=self.norm_order, axis=1).mean()
            
            eps = 10**(-5)
            if (prev_dist + dist_non_parents) > eps:
                sel = prev_dist / (prev_dist + dist_non_parents)
            else:
                sel = 0.0 * np.ones_like(prev_dist)
            if z_where_dist > self.solved_goal_threshold:
                full_sel = sel
            else:
                full_sel = 1.0 - dist_non_parents

                
            info["selectivity"] = sel
            info["full_selectivity"] = full_sel
        else: 
            info["z_where_non_parents_dist"] = 0.0
            info["selectivity"] = 0
            info["full_selectivity"] = 0
        

    def compute_reward(self, obs, next_obs):
        obs = {
            k: v[None] for k, v in obs.items()
        }
        next_obs = {
            k: v[None] for k, v in next_obs.items()
        }
        r, dist_r = self._compute_rewards(obs, next_obs)
        return r[0], dist_r[0]
    
    def _compute_rewards(self, obs: DictObs, next_obs: DictObs) -> Tuple[np.ndarray, np.ndarray]:
        prev_obs_vector = obs["latent_obs_vector"]
        latent_obs_vector = next_obs["latent_obs_vector"]
        subgoal_vector = next_obs["goal_vector"]
        
        subgoal_idx = subgoal_vector[:, self.goal_idx_slices.goal_index_onehot][..., None]
        subgoal_parents_idx = subgoal_vector[:, self.goal_idx_slices.parents_index_onehot][..., None]
        independent_subgoals_idx = np.ones(subgoal_idx.shape) - subgoal_idx - subgoal_parents_idx
        
        subgoal_val = subgoal_vector[:, self.goal_idx_slices.goal_vector]
        initial_obs_vector = subgoal_vector[:, self.goal_idx_slices.init_obs_vector]
        if not self.use_prev: 
            prev_obs_vector = initial_obs_vector

        z_wheres_goal = get_z_wheres_from_goals(subgoal_val,
                                                self.n_objects,
                                                self.z_what_dim,
                                                self.z_where_dim)
        
        z_wheres = get_z_wheres_from_obs(latent_obs_vector,
                                         self.max_n_objects,
                                         self.z_what_dim,
                                         self.z_where_dim)
        z_wheres = z_wheres[:, :self.n_objects]

        z_wheres_prev = get_z_wheres_from_obs(prev_obs_vector,
                                         self.max_n_objects,
                                         self.z_what_dim,
                                         self.z_where_dim)
        z_wheres_prev = z_wheres_prev[:, :self.n_objects]
        diff = (z_wheres_goal - z_wheres) * subgoal_idx
        diff_ind = (z_wheres_prev - z_wheres) * independent_subgoals_idx
        sel_pos_diff = (z_wheres_prev - z_wheres) * subgoal_idx

        # dist_1 = np.linalg.norm(diff, ord=self.norm_order, axis=(1, 2))
        n_subgoals = subgoal_idx.sum(axis=1)[:, 0] 
        dist = np.linalg.norm(diff, ord=self.norm_order, axis=2).sum(1)/n_subgoals
        sel_pos_dist = np.linalg.norm(sel_pos_diff, ord=self.norm_order, axis=2).sum(1)/n_subgoals

        if self.reward_type == 'object_distance':
            return -dist, -dist
        elif self.reward_type == 'selectivity':
            n_non_parents = independent_subgoals_idx.sum(axis=1)[:, 0]
            dist_norm = np.linalg.norm(diff_ind, ord=self.norm_order, axis=(2)).sum(1)
            dist_ind =  np.divide(dist_norm, n_non_parents, 
                                  out=np.zeros_like(dist_norm), 
                                  where=n_non_parents!=0)
            return -(dist + self.alpha_sel * dist_ind), -dist
        elif self.reward_type == 'frac_sel':
            n_non_parents = independent_subgoals_idx.sum(axis=1)[:, 0]
            dist_norm = np.linalg.norm(diff_ind, ord=self.norm_order, axis=(2)).sum(1)
            dist_ind =  np.divide(dist_norm, n_non_parents, 
                                  out=np.zeros_like(dist_norm), 
                                  where=n_non_parents!=0)
            sel_reward =  -1 + np.divide(sel_pos_dist, (sel_pos_dist + dist_ind), 
                                         out=np.zeros_like(dist_norm), 
                                         where=(sel_pos_dist + dist_ind) > self.eps)
            return sel_reward, -dist
        elif self.reward_type == 'frac_sel_goal':
            n_non_parents = independent_subgoals_idx.sum(axis=1)[:, 0]
            dist_norm = np.linalg.norm(diff_ind, ord=self.norm_order, axis=(2)).sum(1)
            dist_ind =  np.divide(dist_norm, n_non_parents, 
                                  out=np.zeros_like(dist_norm), 
                                  where=n_non_parents!=0)
            sel_reward =  -1 + np.divide(sel_pos_dist, (sel_pos_dist + dist_ind), 
                                         out=np.zeros_like(dist_norm), 
                                         where=(sel_pos_dist + dist_ind) > self.eps)
            goal_sel_reward = np.where(self._compute_success_from_reward(-dist), - dist_ind , self.alpha_sel * sel_reward)
            reward = - dist + goal_sel_reward
            return reward, -dist
        else:
            raise NotImplementedError('reward_type {}'
                                      .format(self.reward_type))
    
    def compute_rewards(self, obs: DictObs, next_obs: DictObs) -> np.ndarray:
        return self._compute_rewards(obs, next_obs)[0]


    def extract_achieved_goals(self, latent_obs_vector: np.ndarray, 
                                     future_goal_goals: np.ndarray, 
                                     obj_indices: np.ndarray) -> np.ndarray:
        bs, _ = latent_obs_vector.shape

        initial_obs = future_goal_goals[:, self.goal_idx_slices.init_obs_vector]
        latent_obs = latent_obs_vector[:, 1:]
        zs = latent_obs.reshape((bs, self.max_n_objects, -1))
        zs = zs[:, :self.n_objects]

        one_hot_object_idx = np.zeros((bs, self.n_objects))
        one_hot_object_idx[np.arange(bs), obj_indices] = 1
        subgoal_parents_idx = self.find_parents(obj_indices)
        goal_objects = zs[:, :, :self.z_subgoal_dim].reshape(-1, self.n_objects * self.z_subgoal_dim)
        goals = np.concatenate((obj_indices[:, None], one_hot_object_idx, subgoal_parents_idx, goal_objects, initial_obs), axis=1)

        return goals
    
    def find_parents(self, obj_index: np.ndarray) -> np.ndarray:
        u, inv = np.unique(obj_index, return_inverse = True)
        subgoal_parents_idx = np.array([self.parents[x] for x in u])[inv].reshape(-1, self.n_objects)
        return subgoal_parents_idx   

    def get_diagnostics(self, paths, **kwargs):
        statistics = self.wrapped_env.get_diagnostics(paths, **kwargs)
        for stat_name_in_paths in ["z_where_dist", "z_where_non_parents_dist"]:
            stats = get_stat_in_paths(paths, 'env_infos', stat_name_in_paths)
            statistics.update(create_stats_ordered_dict(
                stat_name_in_paths,
                stats,
                always_show_all_stats=True,
            ))
            final_stats = [s[-1] for s in stats]
            statistics.update(create_stats_ordered_dict(
                "Final " + stat_name_in_paths,
                final_stats,
                always_show_all_stats=True,
            ))

        if self._track_success_rates:
            for i in range(self.n_objects_max):
                attempts = self._attempts.get(i, 0)
                success_rate = self._successes.get(i, 0) / max(attempts, 1)
                statistics["success_rate_{}".format(i)] = success_rate

            # sampling_probs = self._get_sampling_probs(np.eye(self.z_what_dim))
            # for i, prob in enumerate(sampling_probs):
            #     statistics["goal_sampling_prob_{}".format(i)] = prob

        return statistics


def dict2vector(representation: Dict[str, np.ndarray], 
                max_n_objects: int) -> np.ndarray:
    n_objects = representation["z_where"].shape[-2]
    z_n = np.concatenate((representation["z_what"],
                          representation["z_where"],
                          representation["z_depth"]), axis=1)
    z = np.zeros((max_n_objects, z_n.shape[1]))
    z[:n_objects, :] = z_n
    z_vector = np.concatenate([np.array([n_objects]), z.flatten()])

    return z_vector

def get_z_where_from_obs(latent_obs: np.ndarray, 
                         ks: np.ndarray, max_n_objects, 
                         z_what_dim: int, 
                         z_where_dim: int) -> np.ndarray:
    '''
    Return k object z_where part of latent_obs.
    Return has shape Bx2 
    '''   
    bs = len(latent_obs)

    zs = latent_obs[:, 1:].reshape((bs, max_n_objects, -1))
    z_wheres = zs[np.arange(bs), ks, z_what_dim:z_what_dim + z_where_dim]
    return z_wheres

def get_z_wheres_from_obs(latent_obs: np.ndarray, 
                          max_n_objects: int, 
                          z_what_dim: int, 
                          z_where_dim: int) -> np.ndarray:
    '''
    Return all z_wheres of latent_obs.
    Return has shape BxMAX_Nx2 
    '''
    bs = len(latent_obs)
    zs = latent_obs[:, 1:].reshape((bs, max_n_objects, -1))
    z_wheres = zs[:, :, z_what_dim:z_what_dim + z_where_dim]

    return z_wheres

def get_z_wheres_from_goals(goal_val: np.ndarray, 
                            n_objects: int, 
                            z_what_dim: int, 
                            z_where_dim: int) -> np.ndarray:
    '''
    Returns all z_wheres of goal_vector.
    Returns have shape BxNx2 
    '''
    bs = len(goal_val)
    zs = goal_val.reshape((bs, n_objects, -1))
    z_wheres = zs[:, :, z_what_dim:z_what_dim + z_where_dim]

    return z_wheres

def get_idx(goal_dims: GoalVectorDims) -> GoalVectorDims:
    goal_dims = list(goal_dims)
    idxs_end  = np.cumsum(list(goal_dims))
    goal_dims.insert(0, 0)
    idxs_start = np.cumsum(goal_dims)[:-1]
    idx = [slice(idx_start, idx_end, 1) for (idx_start, idx_end) in zip(idxs_start, idxs_end)]
    return GoalVectorDims(*idx)

def get_subgoals(weights):
    weights = np.transpose(weights)
    G = nx.from_numpy_array(weights, create_using=nx.DiGraph)
    all_pathes = [list(nx.all_simple_paths(G, source=0, target=i)) for i in range(1, len(G)) if list(nx.all_simple_paths(G, source=0, target=i))]
    all_pathes += [[[0]]]
    all_pathes

    all_final = []
    for pathes in all_pathes:
        first_element = pathes[0][0]
        other_elements = set()
        for path in pathes:
            other_elements.update(set(path[1:]))
        final = [first_element] + list(other_elements)
        all_final.append(final)
    return all_final

if __name__ == '__main__':
    
    goal_dims = GoalVectorDims(goal_index=1, 
                               goal_index_onehot=5, 
                               parents_index_onehot=5, 
                               goal_vector=35,
                               init_obs_vector=49)

    print(get_idx(goal_dims))