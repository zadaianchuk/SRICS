from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.srics import gt_experiment

from path_length_settings import get_path_settings

path_settings = get_path_settings('MOURL', 'Rearrange', n_objects=4)
PATH_LENGTH = path_settings.subtask_length
N_META = path_settings.attempts
PATH_LENGTH_EVAL = path_settings.eval_path_length
experiment = gt_experiment
variant=dict(
    algorithm='GT_MOURL',
    relational=True,
    double_algo=False,
    imsize=64,
    init_camera=sawyer_init_camera_zoomed_in,
    env_id='SawyerMultiobjectRearrangeEnv-FourObjFirstStableCon2to3-v0',
    graph_path="./data/graphs/4-obj-complex.npy", # set out graph path for training with new relational graph
    save_video=False,
    qf_kwargs=dict(
        hidden_sizes=[400, 300],
    ),
    policy_kwargs=dict(
        hidden_sizes=[400, 300],
    ),
    vf_kwargs=dict(
        hidden_sizes=[400, 300],
    ),
    z_where_dim=2,
    z_depth_dim=1,
    max_n_objects=6,
    max_path_length=PATH_LENGTH,
    n_meta=N_META,
    alpha_sel=0.25,
    start_epoch=0,
    algo_kwargs=dict(
        batch_size=2048,
        num_epochs=25,
        num_eval_steps_per_epoch=50 * PATH_LENGTH_EVAL,
        num_expl_steps_per_train_loop=200,
        num_trains_per_train_loop=200,
        min_num_steps_before_training=4800,
    ),
    twin_sac_trainer_kwargs=dict(
        discount=0.95,
        reward_scale=1,
        soft_target_tau=0.05,
        target_update_period=1,  # 1
        use_automatic_entropy_tuning=True,
    ),
    replay_buffer_kwargs=dict(
        max_size=250000,
        fraction_goals_rollout_goals=0.1,
        fraction_goals_env_goals=0.5,
        max_n_objects=6
    ),
    attention_key_query_size=16,
    exploration_goal_sampling_mode='z_where_prior',
    evaluation_goal_sampling_mode='reset_of_env',
    normalize=False,
    render=False,
    exploration_noise=0.0,
    exploration_type='ou',
    training_mode='train',
    testing_mode='test',
    reward_params=dict(
        type='frac_sel_goal',
    ),
    observation_key='latent_obs_vector',
    desired_goal_key='goal_vector',
    achieved_goal_key='latent_obs_vector',

    hp_options={"activation": "relu",
                "initialization": "xavier_uniform", 
                "policy_last_layer_scale": 1, 
                "policy_initial_std": 0.2,
                "qf_depth": 3,
                "qf_width": 128,
                "policy_depth": 2,
                "policy_width": 128, 
                "lr": 0.001,
                "num_attention_heads": 5,
                "attention_output_aggregation": True}
)

if __name__ == "__main__":

    mode = 'local'
    exp_prefix = 'rlkit-mourl-gt-rearrange-4-object'

    run_experiment(
                gt_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True)
