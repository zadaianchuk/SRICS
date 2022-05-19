import sys
import os
import json
from glob import glob
from contextlib import contextmanager
import numpy as np
import torch

from rlkit.launchers.gnn_training import generate_dataset
from rlkit.torch.gnn.dynnet import DynNet
from rlkit.envs.gt_wrapper import get_subgoals


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def get_checkpoint_pathes(dir_path, checkpoint):
    """Get all checkpoint pathes from dir with many runs
    """
    file_path = os.path.join(dir_path, "*/*/"+f"ckpt_epoch_{checkpoint}.pth")
    return glob(file_path)

def get_dynnet(path):

    dir_path = os.path.dirname(os.path.dirname(path))
    with open(os.path.join(dir_path, "variant.json")) as f:
        variant = json.load(f)
    dynnet_params = variant.get("dynnet_params", dict())
    dynnet_params["logdir"] = "./data/log"
    dynnet = DynNet(**dynnet_params)
    dynnet.load(path)
    return dynnet

def get_data(path):
    with suppress_stdout():
        dir_path = os.path.dirname(os.path.dirname(path))
        with open(os.path.join(dir_path, "variant.json")) as f:
            variant = json.load(f)

    variant["env_id"] = variant["generate_dataset_kwargs"]["env_id"]
    variant["dynnet_path"] = path

    from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
    variant['generate_dataset_kwargs']['init_camera'] = sawyer_init_camera_zoomed_in
    # this should only load existing dataset generated during traning GNN stage
    data, _ = generate_dataset(variant['generate_dataset_kwargs'], force_cashed=True)
    return data

def get_predictions(dynnet: DynNet, data):
    states, actions = data["states"], data["actions"]
    states_t = torch.Tensor(states).to(dynnet.device)
    actions_t = torch.Tensor(actions).to(dynnet.device)

    weights, pred_z_wheres_all, loss = dynnet.evaluate(states_t, actions_t, 0)
    pred_z_wheres_all = pred_z_wheres_all.detach().cpu().numpy()

    return states[:, 1:, ...], pred_z_wheres_all, weights[:,:,:,:, 0].detach().cpu().numpy(), loss

def find_best(pathes):
    """Picks the run with smallest loss"""
    losses = []
    for path_rnn in pathes:
        data = get_data(path_rnn)
        dynnet_rnn = get_dynnet(path_rnn)
        _, _, _, loss = get_predictions(dynnet_rnn, data)
        losses.append(loss)
    return pathes[np.argmin(losses)]

def estimate_weights(experiment_dir, checkpoint):
    """Estimages glpobal graph weights 
    averaging predictions over all trajectories"""
    pathes = get_checkpoint_pathes(experiment_dir, checkpoint)
    best_run = find_best(pathes)
    dynnet = get_dynnet(best_run)
    data = get_data(best_run)
    _, _, graph_weights, _ = get_predictions(dynnet, data)
    graph_weights_mean = (graph_weights.mean((0,1))).astype(np.float)
    return graph_weights_mean

if __name__ == '__main__':
    #set your experiments pathes with exp names as dict
    # {exp_name : exp_path, ...}
    PROJECT_PATH = "./data"
    checkpoints_path = os.path.join(PROJECT_PATH, "./checkpoints")
    experiment_checkpoint = {"3-obj" : 100000,
                             "4-obj" : 100000,
                             "4-obj-complex": 100000
                             }
    experiments_pathes = {exp: os.path.join(checkpoints_path, exp) for exp in experiment_checkpoint}

    #set threshold for global graph estimation
    TR = 0.06

    #set your results path
    results_path = os.path.join(PROJECT_PATH, "./graphs")

    os.makedirs(results_path, exist_ok=True)
    for env, checkpoint in experiment_checkpoint.items():
        experiment_dir = experiments_pathes[env]
        weights = estimate_weights(experiment_dir, checkpoint)
        weights_path = os.path.join(results_path, f"{env}.npy")
        nb_weights_path = os.path.join(results_path, f"nb_{env}.npy")
        np.save(nb_weights_path, weights)
        binary_weights = (weights > TR).astype(np.int32)
        np.save(weights_path, binary_weights)
        print(f"Predicted subgoals are: {get_subgoals(binary_weights)}")