import rlkit.util.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.gnn_training import gt_training
import argparse


variant = dict(
    algorithm='GNN dynamcis',

    generate_dataset_kwargs=dict(
        init_camera=sawyer_init_camera_zoomed_in,
        env_id='SawyerMultiobjectRearrangeEnv-FourObjFirstStableCon2to3-v0',
        N=1000,
        rollout_length=50,
        test_p=.9,
        use_cached=True,
        only_states=True,
        show=False,
    ),
    dynnet_params=dict(
        n_itr=300001,
        lr=0.0005,
        batch_size=100,
        recurent_dyn=True,
        seq_size=20,
        force_teach_each=1,
        self_mask=True,
        no_interaction=False,
        data_dims={
            "action_dim": 2,
            "z_where_dim": 2,
            "z_what_dim": 5,
        }
    ),
    save_period=25,
)
if __name__ == "__main__":
    
    search_space = {}
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'gt_training_4_obj_1_stable_2to3connected_rearrang_descrete_z/rnn'

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', default=False,
                    help='test run with be stored in test folder.')
    args = parser.parse_args()
    if args.test:
        exp_prefix = 'gt_training_4_obj_1_stable_2to3connected_rearrang_test'


    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                gt_training,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                )
