from experiments.util import generate_base_command, generate_run_commands, hash_dict, sample_flag, RESULT_DIR

import yaml
import argparse
import numpy as np
import copy
import os
import itertools

applicable_configs = {
    'general': ['use_wandb'],
    'env': ['time_limit', 'n_envs', 'time_limit_eval'],
    'optimizer': ['optimizer_type', 'num_samples', 'num_elites', 'num_steps', 'horizon', 'alpha'],
    'agent': ['discount', 'n_particles', 'reset_model', 'batch_size', 'train_steps', 'num_epochs', 'max_train_steps'],
    'dynamics_model': ['num_ensembles', 'hidden_layers', 'num_neurons', 'pred_diff', 'deterministic'],
    'trainer': ['eval_freq', 'total_train_steps', 'buffer_size',
                'exploration_steps', 'eval_episodes', 'train_freq', 'rollout_steps',
                'validate', 'normalize', 'action_normalize', 'record_test_video', 'validation_buffer_size',
                'validation_batch_size'],
}

# default_configs = {
#     'use_wandb': True,
#     'time_limit': 200,
#     'time_limit_eval': 200,
#     'n_envs': 5,
#     'num_samples': 500,
#     'num_elites': 50,
#     'num_steps': 10,
#     'horizon': 20,
#     'discount': 1.0,
#     'n_particles': 10,
#     'reset_model': True,
#     'num_ensembles': 5,
#     'hidden_layers': 2,
#     'num_neurons': 128,
#     'pred_diff': True,
#     'batch_size': 256,
#     'eval_freq': 1,
#     'max_train_steps': 10000,
#     'buffer_size': 1000000,
#     'exploration_steps': 0,
#     'eval_episodes': 1,
#     'train_freq': 1,
#     'train_steps': 5000,
#     'rollout_steps': 200,
#     'validate': True,
#     'normalize': True,
#     'action_normalize': True,
#     'record_test_video': True,
#     'validation_buffer_size': 100000,
#     'validation_batch_size': 4096,
# }

search_ranges = {
}


def main(args):
    env_name = args.env_name
    file_path = os.path.dirname(os.path.abspath(__file__))
    assert env_name in ['Pendulum', 'Cheetah', 'MountainCar', 'Reacher', 'Swimmer', 'GpPend'], \
        "Only cheetah, mountain car, pendulum, reacher, Swimmer and GpPend environment work"
    if env_name in ['Pendulum', 'MountainCar', 'Reacher', 'Swimmer']:
        EXPLORATION_STRATEGY = ['Uniform', 'Optimistic', 'Mean', 'PETS', 'HUCRL']
        if env_name == 'Pendulum':
            import experiments.pendulum_exp.active_exploration_exp_pendulum as active_exploration_exp
            default_configs = yaml.safe_load(open(file_path + '/pendulum_exp/hyperparams.yaml', 'r'))
        elif env_name == 'Reacher':
            import experiments.reacher_exp.active_exploration_reacher as active_exploration_exp
            default_configs = yaml.safe_load(open(file_path + '/reacher_exp/hyperparams.yaml', 'r'))
        elif env_name == 'Swimmer':
            import experiments.swimmer_exp.active_exploration_swimmer as active_exploration_exp
            default_configs = yaml.safe_load(open(file_path + '/swimmer_exp/hyperparams.yaml', 'r'))
        else:
            import experiments.mountain_car_exp.active_exploration_exp_mountain_car as active_exploration_exp
            default_configs = yaml.safe_load(open(file_path + '/mountain_car_exp/hyperparams.yaml', 'r'))
    elif env_name in ['Cheetah']:
        EXPLORATION_STRATEGY = ['Uniform', 'Optimistic', 'Mean', 'PETS', 'HUCRL']
        import experiments.half_cheetah_exp.active_exploration_cheetah as active_exploration_exp
        default_configs = yaml.safe_load(open(file_path + '/half_cheetah_exp/hyperparams.yaml', 'r'))
    else:
        EXPLORATION_STRATEGY = ['Uniform', 'Optimistic', 'Mean', 'PETS', 'HUCRL']
        import experiments.pendulum_exp.active_exploration_exp_pendulum_gp as active_exploration_exp
        default_configs = yaml.safe_load(open(file_path + '/pendulum_exp/hyperparams_gp.yaml', 'r'))


    # check consistency of configuration dicts
    assert set(itertools.chain(*list(applicable_configs.values()))) == {*default_configs.keys(), *search_ranges.keys()}
    rds = np.random.RandomState(args.seed)
    assert args.num_seeds_per_hparam < 100
    init_seeds = list(rds.randint(0, 10 ** 6, size=(100,)))

    # determine name of experiment
    exp_base_path = os.path.join(RESULT_DIR, args.exp_name)
    exp_path = os.path.join(exp_base_path, f'{args.exp_name}')

    command_list = []
    for _ in range(args.num_hparam_samples):
        # transfer flags from the args
        flags = copy.deepcopy(args.__dict__)
        logs_dir = './'
        [flags.pop(key) for key in ['seed', 'num_hparam_samples', 'num_seeds_per_hparam', 'num_cpus',
                                    'num_gpus', 'launch_mode', 'env_name', 'long_run', 'num_hours']]

        # randomly sample flags
        for flag in default_configs:
            if flag in search_ranges:
                flags[flag] = sample_flag(sample_spec=search_ranges[flag], rds=rds)
            else:
                flags[flag] = default_configs[flag]
        flags['logs_dir'] = logs_dir
        for exploration_strategy in EXPLORATION_STRATEGY:
            # determine subdir which holds the repetitions of the exp
            flags_hash = hash_dict(flags)
            flags['exploration_strategy'] = exploration_strategy
            flags['exp_result_folder'] = os.path.join(exp_path, flags_hash)

            for j in range(args.num_seeds_per_hparam):
                seed = init_seeds[j]
                cmd = generate_base_command(active_exploration_exp, flags=dict(**flags, **{'seed': seed}))
                command_list.append(cmd)

    # submit jobs
    generate_run_commands(command_list, mode=args.launch_mode,
                          promt=True,
                          mem=16000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=834, help='random number generator seed')
    parser.add_argument('--exp_name', type=str, required=True, default='Pendulum-ActiveExploration')
    parser.add_argument('--env_name', type=str, default='Pendulum', help='Name of the environment')
    parser.add_argument('--num_cpus', type=int, default=8, help='number of cpus to use')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--launch_mode', type=str, default='local', help='how to launch the experiments')
    parser.add_argument('--num_hparam_samples', type=int, default=1)
    parser.add_argument('--num_seeds_per_hparam', type=int, default=3)
    parser.add_argument('--use_log', default=False, action="store_true")
    parser.add_argument('--use_al', default=False, action="store_true")
    parser.add_argument('--beta', type=float, default=2.0)
    parser.add_argument('--long_run', default=False, action="store_true")
    parser.add_argument('--num_hours', type=int, default=None)

    args = parser.parse_args()
    main(args)
