import time
import warnings
import cv2
import numpy as np
import os.path as osp
import gym

import multiworld
from multiworld.core.image_env import ImageEnv, unormalize_image

from rlkit.core import logger
from rlkit.torch.gnn.dynnet import DynNet


def generate_dataset(variant, force_cashed=False):
    env_kwargs = variant.get('env_kwargs', None)
    env_id = variant.get('env_id', None)
    N = variant.get('N', 1000)
    only_states = variant.get('only_states', False)
    rollout_length = variant.get('rollout_length', 50)
    use_cached = variant.get('use_cached', True)
    imsize = variant.get('imsize', 64)
    num_channels = variant.get('num_channels', 3)
    show = variant.get('show', False)
    init_camera = variant.get('init_camera', None)
    save_file_prefix = variant.get('save_file_prefix', None)
    tag = variant.get('tag', '')

    if env_kwargs is None:
        env_kwargs = {}
    if save_file_prefix is None:
        save_file_prefix = env_id
    filename = "./data/tmp/{}_N{}_rollout_length{}_imsize{}_{}{}.npz".format(
        save_file_prefix,
        str(N),
        str(rollout_length),
        init_camera.__name__ if init_camera else '',
        imsize,
        tag,
    )
    print(filename)
    import os
    if not osp.exists('./data/tmp/'):
        os.makedirs('./data/tmp/')
    info = {}
    import os
    if not os.path.exists("./data/tmp/"):
        os.makedirs("./data/tmp/")
    if use_cached and osp.isfile(filename):
        dataset = np.load(filename)
        print("loaded data from saved file", filename)
    else:
        if force_cashed:
            raise FileNotFoundError("No saved dataset found. Please make sure that you have dataset generated during training in ./data/tmp dir")
        if use_cached:
            warnings.warn("use_cached=True, but no file found, so new collection data started")
        now = time.time()
        multiworld.register_all_envs()
        env = gym.make(env_id)
        if hasattr(env, "detect_contact"):
            detect_contact = env.detect_contact
        else:
            detect_contact = False

        if not isinstance(env, ImageEnv):
            env = ImageEnv(
                env,
                imsize,
                init_camera=init_camera,
                transpose=True,
                normalize=True,
                non_presampled_goal_img_is_garbage=True,
            )
        init_obs = env.reset()
        act_dim = env.action_space.low.size
        state_dim = init_obs['state_observation'].shape[0]
        num_objects = int(init_obs['num_objects']) + 1
        info['env'] = env
        if not only_states:
            imgs = np.zeros((N, rollout_length, imsize * imsize * num_channels), dtype=np.uint8)
        states = np.zeros((N, rollout_length, state_dim))
        actions = np.zeros((N, rollout_length, act_dim))
        if detect_contact:
            detected_contacts = np.zeros((N, rollout_length, num_objects, num_objects))
        for i in range(N):
            env.reset()
            for j in range(rollout_length):
                action = env.action_space.sample()
                obs = env.step(action)[0]
                state = obs['state_observation']
                if detect_contact:
                    detected_contacts[i, j, :, :] = obs['detected_contacts']
                if not only_states:
                    img = obs['image_observation']
                    imgs[i, j, :] = unormalize_image(img)
                actions[i, j, :] = action
                states[i, j, :] = state
                if not only_states and show:
                    img = img.reshape(3, imsize, imsize).transpose()
                    img = img[::-1, :, ::-1]
                    cv2.imshow('img', img)
                    cv2.waitKey(1)
        print("done making training data", filename, time.time() - now)
        if not only_states:
            dataset = {"imgs": imgs, "states": states, "actions": actions}
        else:
            dataset = {"states": states, "actions": actions}
        if detect_contact:
            dataset["detected_contacts"] = detected_contacts
        np.savez(filename, **dataset)

    return dataset, info

def gt_training(variant):
    dynnet_params = variant.get("dynnet_params", dict())
    dynnet_params["logdir"] = logger.get_snapshot_dir()
    dynnet = DynNet(**dynnet_params)
    data, _ = generate_dataset(variant['generate_dataset_kwargs'])
    states, actions = data["states"], data["actions"]

    # number of episodes for test
    k = int(0.1 * variant['generate_dataset_kwargs']["N"])
    states_test = states[:k]
    actions_test = actions[:k]

    states_train = states[k:]
    actions_train = actions[k:]
    dynnet.train(states=states_train, actions=actions_train, evaluate_data=(states_test, actions_test))
