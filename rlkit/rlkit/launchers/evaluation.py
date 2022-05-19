import time
from rlkit.core import logger
import torch
import cv2
import numpy as np
import os
import pandas as pd
from rlkit.samplers.data_collector.srics_env import WrappedEnvPathCollector
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.core import logger, eval_util
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
from rlkit.launchers.scalor_mourl_experiment import get_envs
from rlkit.envs.render_wrapper import get_render_wrapper


def evaluation(variant):
    import rlkit.torch.pytorch_util as ptu
    observation_key = variant.get('observation_key')
    desired_goal_key = variant.get('desired_goal_key')
    achieved_goal_key = variant.get('achieved_goal_key')
    policy_folder_path = variant.get('policy_folder_path')
    policy_folder_name = os.path.split(policy_folder_path)[-1]
    results_path = variant.get('results_path')

    itr = variant.get('itr')
    policy_path = os.path.join(policy_folder_path, f"itr_{itr}.pkl")
    data = torch.load(policy_path, map_location=ptu.device)
    policy = data['evaluation/policy']
    policy.to(ptu.device)
    env = get_envs(variant)
    num_steps = variant["num_eval_steps"]
    max_path_length = variant["max_path_length"]
    n_meta = variant["n_meta"]
    eval_path_collector = WrappedEnvPathCollector(
        variant['evaluation_goal_sampling_mode'],
        env,
        policy,
        n_meta * max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    paths = eval_path_collector.collect_new_paths(
                                max_path_length=n_meta * max_path_length,
                                num_steps=num_steps,
                                discard_incomplete_paths=True)
    statistics = eval_util.get_generic_path_information(paths)
    statistics = {f"evaluation/{key}": value for key, value in statistics.items()}
    results_folder_path = os.path.join(results_path, policy_folder_name)
    if not os.path.exists(results_folder_path):
        os.makedirs(results_folder_path)
    results_path = os.path.join(results_folder_path, f"itr_{itr}_eval_results.csv")
    pd.DataFrame(statistics, index=[0]).to_csv(results_path)
    print(statistics["evaluation/env_infos/final/avg_object_distance Mean"])