import pathlib
import pickle
import numpy as np
from gym.spaces import Dict, Discrete
from typing import Union
from rlkit.data_management.replay_buffer import ReplayBuffer


class ObsDictRelabelingBuffer(ReplayBuffer):
    """
    Replay buffer for environments whose observations are dictionaries, such as
        - OpenAI Gym GoalEnv environments. https://blog.openai.com/ingredients-for-robotics-research/
        - multiworld MultitaskEnv. https://github.com/vitchyr/multiworld/

    Implementation details:
     - Only add_path is implemented.
     - Image observations are presumed to start with the 'image_' prefix
     - Every sample from [0, self._size] will be valid.
     - Observation and next observation are saved separately. It's a memory
       inefficient to save the observations twice, but it makes the code
       *much* easier since you no longer have to worry about termination
       conditions.
    """

    def __init__(
            self,
            max_size,
            env,
            fraction_goals_rollout_goals=1.0,
            fraction_goals_env_goals=0.0,
            internal_keys=None,
            goal_keys=None,
            observation_key='observation',
            desired_goal_key='desired_goal',
            achieved_goal_key='achieved_goal',
            terminate_episode_on_success=False,
            success_reward=1.0
    ):
        if internal_keys is None:
            internal_keys = []
        self.internal_keys = internal_keys
        if goal_keys is None:
            goal_keys = []
        if desired_goal_key not in goal_keys:
            goal_keys.append(desired_goal_key)
        self.goal_keys = goal_keys
        assert isinstance(env.observation_space, Dict)
        assert 0 <= fraction_goals_rollout_goals
        assert 0 <= fraction_goals_env_goals
        assert 0 <= fraction_goals_rollout_goals + fraction_goals_env_goals
        assert fraction_goals_rollout_goals + fraction_goals_env_goals <= 1
        self.max_size = max_size
        self.env = env
        self.fraction_goals_rollout_goals = fraction_goals_rollout_goals
        self.fraction_goals_env_goals = fraction_goals_env_goals
        self.ob_keys_to_save = [
            observation_key,
            desired_goal_key,
            achieved_goal_key,
        ]
        self.observation_key = observation_key
        self.desired_goal_key = desired_goal_key
        self.achieved_goal_key = achieved_goal_key
        self.terminate_episode_on_success = terminate_episode_on_success
        self.success_reward = success_reward
        if isinstance(self.env.action_space, Discrete):
            self._action_dim = env.action_space.n
        else:
            self._action_dim = env.action_space.low.size

        self._actions = np.zeros((max_size, self._action_dim))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_size, 1), dtype='uint8')
        # self._obs[key][i] is the value of observation[key] at time i
        self._obs = {}
        self._next_obs = {}
        self.ob_spaces = self.env.observation_space.spaces
        self.internal_keys = internal_keys
        for key in self.ob_keys_to_save + internal_keys:
            assert key in self.ob_spaces, \
                "Key not found in the observation space: %s" % key
            type = np.float64
            if key.startswith('image'):
                type = np.uint8
            self._obs[key] = np.zeros(
                (max_size, self.ob_spaces[key].low.size), dtype=type)
            self._next_obs[key] = np.zeros(
                (max_size, self.ob_spaces[key].low.size), dtype=type)

        self._top = 0
        self._size = 0

        # Let j be any index in self._idx_to_future_obs_idx[i]
        # Then self._next_obs[j] is a valid next observation for observation i
        self._idx_to_future_obs_idx = [None] * max_size

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        raise NotImplementedError("Only use add_path")

    def terminate_episode(self):
        pass

    def num_steps_can_sample(self):
        return self._size

    def add_path(self, path):
        obs = path["observations"]
        actions = path["actions"]
        rewards = path["rewards"]
        next_obs = path["next_observations"]
        terminals = path["terminals"]
        path_len = len(rewards)

        actions = flatten_n(actions)
        if isinstance(self.env.action_space, Discrete):
            actions = np.eye(self._action_dim)[actions].reshape((-1, self._action_dim))
        obs = flatten_dict(obs, self.ob_keys_to_save + self.internal_keys)
        next_obs = flatten_dict(next_obs, self.ob_keys_to_save + self.internal_keys)
        obs = preprocess_obs_dict(obs)
        next_obs = preprocess_obs_dict(next_obs)

        if self._top + path_len >= self.max_size:
            """
            All of this logic is to handle wrapping the pointer when the
            replay buffer gets full.
            """
            num_pre_wrap_steps = self.max_size - self._top
            # numpy slice
            pre_wrap_buffer_slice = np.s_[
                                    self._top:self._top + num_pre_wrap_steps, :
                                    ]
            pre_wrap_path_slice = np.s_[0:num_pre_wrap_steps, :]

            num_post_wrap_steps = path_len - num_pre_wrap_steps
            post_wrap_buffer_slice = slice(0, num_post_wrap_steps)
            post_wrap_path_slice = slice(num_pre_wrap_steps, path_len)
            for buffer_slice, path_slice in [
                (pre_wrap_buffer_slice, pre_wrap_path_slice),
                (post_wrap_buffer_slice, post_wrap_path_slice),
            ]:
                self._actions[buffer_slice] = actions[path_slice]
                self._terminals[buffer_slice] = terminals[path_slice]
                for key in self.ob_keys_to_save + self.internal_keys:
                    self._obs[key][buffer_slice] = obs[key][path_slice]
                    self._next_obs[key][buffer_slice] = next_obs[key][path_slice]
            # Pointers from before the wrap
            for i in range(self._top, self.max_size):
                self._idx_to_future_obs_idx[i] = np.hstack((
                    # Pre-wrap indices
                    np.arange(i, self.max_size),
                    # Post-wrap indices
                    np.arange(0, num_post_wrap_steps)
                ))
            # Pointers after the wrap
            for i in range(0, num_post_wrap_steps):
                self._idx_to_future_obs_idx[i] = np.arange(
                    i,
                    num_post_wrap_steps,
                )
        else:
            slc = np.s_[self._top:self._top + path_len, :]
            self._actions[slc] = actions
            self._terminals[slc] = terminals
            for key in self.ob_keys_to_save + self.internal_keys:
                self._obs[key][slc] = obs[key]
                self._next_obs[key][slc] = next_obs[key]
            for i in range(self._top, self._top + path_len):
                self._idx_to_future_obs_idx[i] = np.arange(
                    i, self._top + path_len
                )
        self._top = (self._top + path_len) % self.max_size
        self._size = min(self._size + path_len, self.max_size)


    def _sample_indices(self, batch_size):
        return np.random.randint(0, self._size, batch_size)

    def random_batch(self, batch_size):
        indices = self._sample_indices(batch_size)
        resampled_goals = self._next_obs[self.desired_goal_key][indices]

        num_env_goals = int(batch_size * self.fraction_goals_env_goals)
        num_rollout_goals = int(batch_size * self.fraction_goals_rollout_goals)
        num_future_goals = batch_size - (num_env_goals + num_rollout_goals)
        new_obs_dict = self._batch_obs_dict(indices)
        new_next_obs_dict = self._batch_next_obs_dict(indices)

        if num_env_goals > 0:
            env_goals = self.env.sample_goals(num_env_goals)
            env_goals = preprocess_obs_dict(env_goals)
            last_env_goal_idx = num_rollout_goals + num_env_goals
            resampled_goals[num_rollout_goals:last_env_goal_idx] = (
                env_goals[self.desired_goal_key]
            )
            for goal_key in self.goal_keys:
                new_obs_dict[goal_key][num_rollout_goals:last_env_goal_idx] = \
                    env_goals[goal_key]
                new_next_obs_dict[goal_key][
                num_rollout_goals:last_env_goal_idx] = \
                    env_goals[goal_key]
        if num_future_goals > 0:
            future_obs_idxs = []
            for i in indices[-num_future_goals:]:
                possible_future_obs_idxs = self._idx_to_future_obs_idx[i]
                # This is generally faster than random.choice. Makes you wonder what
                # random.choice is doing
                num_options = len(possible_future_obs_idxs)
                next_obs_i = int(np.random.randint(0, num_options))
                future_obs_idxs.append(possible_future_obs_idxs[next_obs_i])
            future_obs_idxs = np.array(future_obs_idxs)
            resampled_goals[-num_future_goals:] = self._next_obs[
                self.achieved_goal_key
            ][future_obs_idxs]
            for goal_key in self.goal_keys:
                new_obs_dict[goal_key][-num_future_goals:] = \
                    self._next_obs[goal_key][future_obs_idxs]
                new_next_obs_dict[goal_key][-num_future_goals:] = \
                    self._next_obs[goal_key][future_obs_idxs]

        new_obs_dict[self.desired_goal_key] = resampled_goals
        new_next_obs_dict[self.desired_goal_key] = resampled_goals
        new_obs_dict = postprocess_obs_dict(new_obs_dict)
        new_next_obs_dict = postprocess_obs_dict(new_next_obs_dict)
        # resampled_goals must be postprocessed as well
        resampled_goals = new_next_obs_dict[self.desired_goal_key]

        new_actions = self._actions[indices]
        """
        For example, the environments in this repo have batch-wise
        implementations of computing rewards:

        https://github.com/vitchyr/multiworld
        """

        if hasattr(self.env, 'compute_rewards'):
            new_rewards = self.env.compute_rewards(
                new_obs_dict,
                new_next_obs_dict,
            )
        else:  # Assuming it's a (possibly wrapped) gym GoalEnv
            new_rewards = np.ones((batch_size, 1))
            for i in range(batch_size):
                new_rewards[i] = self.env.compute_reward(
                    new_next_obs_dict[self.achieved_goal_key][i],
                    new_next_obs_dict[self.desired_goal_key][i],
                    None
                )
        new_terminals = self._terminals[indices].copy()
        if self.terminate_episode_on_success:
            new_terminals[new_rewards == self.success_reward] = 1.0

        new_rewards = new_rewards.reshape(-1, 1)

        new_obs = new_obs_dict[self.observation_key]
        new_next_obs = new_next_obs_dict[self.observation_key]
        batch = {
            'observations': new_obs,
            'actions': new_actions,
            'rewards': new_rewards,
            'terminals': new_terminals,
            'next_observations': new_next_obs,
            'resampled_goals': resampled_goals,
            'indices': np.array(indices).reshape(-1, 1),
        }
        return batch

    def _batch_obs_dict(self, indices):
        return {
            key: self._obs[key][indices]
            for key in self.ob_keys_to_save
        }

    def _batch_next_obs_dict(self, indices):
        return {
            key: self._next_obs[key][indices]
            for key in self.ob_keys_to_save
        }

    def sample_buffer_goals(self, batch_size):
        """
        Samples goals from weighted replay buffer for relabeling or exploration.
        Returns None if replay buffer is empty.

        Just uses uniform distribution over all buffer entries
        """
        if self._size == 0:
            return None

        idxs = self._sample_indices(batch_size)
        achieved_goal = self._next_obs[self.achieved_goal_key][idxs]

        return {
            self.desired_goal_key: achieved_goal
        }

    def get_state(self):
        state = dict(obs=self._obs,
                next_obs=self._next_obs,
                idx_to_future_obs_idx = self._idx_to_future_obs_idx,
                actions=self._actions,
                terminals=self._terminals,
                top=self._top,
                size=self._size
                )
        return state

    def set_state(self, data):
        self._obs = data["obs"]
        self._next_obs = data["next_obs"]
        self._idx_to_future_obs_idx = data["idx_to_future_obs_idx"]
        self._actions = data["actions"]
        self._terminals = data["terminals"]
        self._top = data["top"]
        self._size = data["size"]

    def save(self, save_dir: Union[pathlib.Path, str]):
        """Saves the data in the replay buffer to a given directory.
        Args:
            save_dir (str): the directory to save the data to. File name will be
                replay_buffer.npz.
        """
        path = pathlib.Path(save_dir) / "replay_buffer.pkl"
        state = self.get_state()
        with open(path,'wb') as file_:
            pickle.dump(state, file_)

    def load(self, load_dir: Union[pathlib.Path, str]) -> None:
        """Loads transition data from a given directory.
        Args:
            load_dir (str): the directory where the buffer is stored.
        """
        path = pathlib.Path(load_dir) / "replay_buffer.pkl"
        with open(path,'rb') as file_:
            data = pickle.load(file_)
            self.set_state(data)


def flatten_n(xs):
    xs = np.asarray(xs)
    return xs.reshape((xs.shape[0], -1))


def flatten_dict(dicts, keys):
    """
    Turns list of dicts into dict of np arrays
    """
    return {
        key: flatten_n([d[key] for d in dicts])
        for key in keys
    }


def preprocess_obs_dict(obs_dict):
    """
    Apply internal replay buffer representation changes: save images as bytes
    """
    for obs_key, obs in obs_dict.items():
        if 'image' in obs_key and obs is not None:
            obs_dict[obs_key] = unnormalize_image(obs)
    return obs_dict


def postprocess_obs_dict(obs_dict):
    """
    Undo internal replay buffer representation changes: save images as bytes
    """
    for obs_key, obs in obs_dict.items():
        if 'image' in obs_key and obs is not None:
            obs_dict[obs_key] = normalize_image(obs)
    return obs_dict


def normalize_image(image):
    assert image.dtype == np.uint8
    return np.float64(image) / 255.0


def unnormalize_image(image):
    assert image.dtype != np.uint8
    return np.uint8(image * 255.0)
