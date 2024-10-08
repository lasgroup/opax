from experiments.util_neorl import generate_run_commands, generate_base_command, dict_permutations
from experiments.neorl.pendulum import exp
import argparse

PROJECT_NAME = 'OMBRL-PendUp23ApriilFinalRun'
_applicable_configs = {
    'use_wandb': [1],
    'exp_name': [PROJECT_NAME],
    'seed': list(range(10)),
    'time_limit': [200],
    'n_envs': [1],
    'num_samples': [500],
    'num_elites': [50],
    'num_steps': [10],
    'horizon': [20],
    'n_particles': [5],
    'reset_model': [0],
    'num_ensembles': [10],
    'hidden_layers': [2],
    'num_neurons': [256],
    'deterministic': [0],
    'max_train_steps': [5_000],
    'pred_diff': [1],
    'batch_size': [64],
    'num_epochs': [50],
    'eval_freq': [5],
    'total_train_steps': [1_000],
    'buffer_size': [1_000_000],
    'exploration_steps': [0],
    'eval_episodes': [1],
    'train_freq': [1],
    'rollout_steps': [10],
    'normalize': [1],
    'action_normalize': [1],
    'validate': [1],
    'record_test_video': [1],
    'validation_buffer_size': [100_000],
    'validation_batch_size': [10_000],
    'action_cost': [0.0, 0.1, 0.2, 0.5],
    'time_limit_eval': [200],
    'action_repeat': [1],
    'lr': [1e-3],
    'colored_noise_exponent': [0.25],
    'calibrate_model': [1],
}

_applicable_configs_hucrl = {'exploration_strategy': ['HUCRL'], 'beta': [2.0]} | _applicable_configs
_applicable_configs_ombrl = {'exploration_strategy': ['OMBRL'], 'beta': [2.0]} | _applicable_configs
_applicable_configs_pets = {'exploration_strategy': ['PETS'], 'beta': [0.0]} | _applicable_configs
_applicable_configs_mean = {'exploration_strategy': ['Mean'], 'beta': [0.0]} | _applicable_configs
_applicable_configs_thompson = {'exploration_strategy': ['Thompson'], 'beta': [0.0]} | _applicable_configs

all_flags_combinations = dict_permutations(_applicable_configs_hucrl) + dict_permutations(_applicable_configs_pets) \
                         + dict_permutations(_applicable_configs_mean) + dict_permutations(_applicable_configs_thompson) \
                         + dict_permutations(_applicable_configs_ombrl)


def main(args):
    command_list = []
    logs_dir = './'
    if args.mode == 'euler':
        logs_dir = '/cluster/scratch/'
        logs_dir += 'sukhijab' + '/' + PROJECT_NAME + '/'

    for flags in all_flags_combinations:
        flags['logs_dir'] = logs_dir
        cmd = generate_base_command(exp, flags=flags)
        command_list.append(cmd)

    # submit jobs
    num_hours = 23 if args.long_run else 3
    generate_run_commands(command_list, num_cpus=args.num_cpus, num_gpus=args.num_gpus,
                          mode=args.mode, duration=f'{num_hours}:59:00', prompt=True, mem=16000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_cpus', type=int, default=1, help='number of cpus to use')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--mode', type=str, default='euler', help='how to launch the experiments')
    parser.add_argument('--long_run', default=False, action="store_true")

    args = parser.parse_args()
    main(args)
