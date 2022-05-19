from rlkit.core import logger
import mujoco_py

from rlkit.samplers.data_collector.srics_env import WrappedEnvPathCollector
from rlkit.torch.her.her import HERTrainer
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.launchers.launcher_util import apply_hp_options

def get_envs(variant):
    from rlkit.envs.gt_wrapper import GTWrappedEnv, GTRelationalWrappedEnv
    from rlkit.envs.gt_gnn_wrapper import GTWrappedGNNEnv
    reward_params = variant.get("reward_params", dict())
    z_where_dim = variant.get("z_where_dim", 2)
    z_depth_dim = variant.get("z_depth_dim", 1)
    max_n_objects = variant.get("max_n_objects", 5)
    done_on_success = variant.get("done_on_success", False)

    if 'env_id' in variant:
        import gym
        import multiworld
        multiworld.register_all_envs()
        mujoco_py.ignore_mujoco_warnings().__enter__()
        env = gym.make(variant['env_id'])
    else:
        env = variant["env_class"](**variant['env_kwargs'])
    variant["z_what_dim"] = env.n_objects_max + 1
    dynnet_path = variant.get("dynnet_path", None)
    if  dynnet_path is not None:
        gt_env = GTWrappedGNNEnv( 
        env,
        dynnet_path=dynnet_path,
        dynnet_params=variant["dynnet_params"],        
        z_where_dim=z_where_dim,
        z_depth_dim=z_depth_dim,
        max_n_objects=max_n_objects,
        sub_task_horizon=variant['max_path_length'],
        reward_params=reward_params,
        done_on_success=done_on_success,
        **variant.get('wrapped_env_kwargs', {})
        )
    else:
        relational = variant.get('relational', False)
        if relational:
            alpha_sel =  variant.get('alpha_sel')
            solved_goal_threshold = variant.get('solved_goal_threshold', 0.05)
            ordered_evaluation = variant.get("ordered_evaluation", True)
            graph_path = variant.get("graph_path")
            
            gt_env = GTRelationalWrappedEnv(
            env,
            env_id=variant['env_id'],
            z_where_dim=z_where_dim,
            z_depth_dim=z_depth_dim,
            alpha_sel=alpha_sel,
            max_n_objects=max_n_objects,
            sub_task_horizon=variant['max_path_length'],
            n_meta=variant['n_meta'],
            solved_goal_threshold=solved_goal_threshold,
            ordered_evaluation=ordered_evaluation,
            graph_path=graph_path,
            reward_params=reward_params,
            done_on_success=done_on_success,
            **variant.get('wrapped_env_kwargs', {}))
        else:
            gt_env = GTWrappedEnv(
            env,
            z_where_dim=z_where_dim,
            z_depth_dim=z_depth_dim,
            max_n_objects=max_n_objects,
            sub_task_horizon=variant['max_path_length'],
            reward_params=reward_params,
            done_on_success=done_on_success,
            **variant.get('wrapped_env_kwargs', {})

    )


    return gt_env

def gt_experiment(variant):
    import rlkit.torch.pytorch_util as ptu
    from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
    from rlkit.data_management.shared_obs_dict_replay_buffer import SharedObsDictRelabelingMultiObjectBuffer
    # from rlkit.data_management.obs_dict_multiobject_replay_buffer import ObsDictRelabelingMultiObjectBuffer
    from rlkit.torch.networks import SetGoalAttentionMlp
    from rlkit.torch.sac.policies import SetGoalAttentionTanhGaussianPolicy

    apply_hp_options(variant)
    # mourl_preprocess_variant(variant)  
    train_env = get_envs(variant)
    eval_env = get_envs(variant)

    assert hasattr(train_env, "goal_dims"), "Environment is not relational"
    goal_dims = train_env.goal_dims
    observation_key = variant.get('observation_key', 'latent_obs_vector')
    desired_goal_key = variant.get('desired_goal_key', 'goal_vector')
    achieved_goal_key = variant.get('achieved_goal_key', 'latent_obs_vector')
    action_dim = train_env.action_space.low.size
    z_what_dim = variant.get("z_what_dim")
    z_where_dim = variant.get("z_where_dim", 2)
    z_depth_dim = variant.get("z_depth_dim", 1)
    max_n_objects = variant.get("max_n_objects")
    n_objects = train_env.n_objects
    assert n_objects <= max_n_objects
    z_goal_dim = z_what_dim + z_where_dim
    z_where_dim = variant.get("z_where_dim", 2)
    z_object_dim = z_what_dim + z_where_dim + z_depth_dim
    n_meta = variant.get("n_meta")

    if 'embedding_dim' in variant:
        embed_dim = variant["embedding_dim"]
    else:
        embed_dim = variant["attention_key_query_size"]

    qf_class = variant.get("qf_class", SetGoalAttentionMlp)

    print(f"z_goal_dim: {z_goal_dim}, z_dim: {z_object_dim}")
    qf1 = qf_class(
                        embed_dim=embed_dim,
                        goal_dims=goal_dims,
                        z_goal_size=z_goal_dim,
                        z_size=z_object_dim,
                        action_size=action_dim,
                        max_objects=max_n_objects,
                        n_objects=n_objects,
                        output_size=1,
                        **variant.get('qf_kwargs', {})
    )

    qf2 = qf_class(
                        embed_dim=embed_dim,
                        goal_dims=goal_dims,
                        z_goal_size=z_goal_dim,
                        z_size=z_object_dim,
                        action_size=action_dim,
                        max_objects=max_n_objects,
                        n_objects=n_objects,
                        output_size=1,
                        **variant.get('qf_kwargs', {})
    )

    target_qf1 = qf_class(
                        embed_dim=embed_dim,
                        goal_dims=goal_dims,
                        z_goal_size=z_goal_dim,
                        z_size=z_object_dim,
                        action_size=action_dim,
                        max_objects=max_n_objects,
                        n_objects=n_objects,
                        output_size=1,
                        **variant.get('qf_kwargs', {})
    )

    target_qf2 = qf_class(
                        embed_dim=embed_dim,
                        goal_dims=goal_dims,
                        z_goal_size=z_goal_dim,
                        z_size=z_object_dim,
                        action_size=action_dim,
                        max_objects=max_n_objects,
                        n_objects=n_objects,
                        output_size=1,
                        **variant.get('qf_kwargs', {})
    )

    policy_class = variant.get("policy_class", SetGoalAttentionTanhGaussianPolicy)
    policy = policy_class(
                        embed_dim=embed_dim,
                        z_size=z_object_dim,
                        goal_dims=goal_dims,
                        z_goal_size=z_goal_dim,
                        max_objects=max_n_objects,
                        n_objects=n_objects,
                        action_dim=action_dim,
                        **variant.get('policy_kwargs', {})
    )

    replay_buffer = SharedObsDictRelabelingMultiObjectBuffer(
        env=train_env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )

    max_path_length = variant['max_path_length']

    trainer = SACTrainer(
        env=train_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['twin_sac_trainer_kwargs']
    )
    trainer = HERTrainer(trainer)
    eval_path_collector = WrappedEnvPathCollector(
        variant['evaluation_goal_sampling_mode'],
        eval_env,
        MakeDeterministic(policy),
        n_meta * max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    expl_path_collector = WrappedEnvPathCollector(
        variant['exploration_goal_sampling_mode'],
        train_env,
        policy,
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=train_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        max_path_length=max_path_length,
        max_path_length_eval=max_path_length * n_meta,
        **variant['algo_kwargs']
    )
    if variant["start_epoch"] == 0:
        snapshot = algorithm._get_snapshot()
        logger.save_itr_params(-1, snapshot)
    else:
        logger.resume = True

    algorithm.to(ptu.device)
    run_n_epochs = variant.get("run_n_epochs", 5)
    algorithm.train(start_epoch=variant["start_epoch"], run_n_epochs=run_n_epochs)
