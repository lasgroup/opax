import glob
import hashlib
import itertools
import json
import multiprocessing
import os
import sys
from typing import Dict, Optional, Any, List

import jax.numpy as jnp
import numpy as np
import pandas as pd

""" Relevant Directories """

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULT_DIR = os.path.join(BASE_DIR, 'results')

""" Data Utilities """


def load_csv_recordings(recordings_dir: str) -> List[pd.DataFrame]:
    """ Load all csv files in a directory into a list of pandas dataframes."""
    dfs = []
    for path in glob.glob(os.path.join(recordings_dir, '*sampled.csv')):
        df = pd.read_csv(path)
        df.columns = [c[1:] for c in df.columns]
        dfs.append(df)
    return dfs


def get_trajectory_windows(arr: jnp.array, window_size: int = 10) -> jnp.array:
    """Sliding window over an array along the first axis."""
    arr_strided = jnp.stack([arr[i:(-window_size + i)] for i in range(window_size)], axis=-2)
    assert arr_strided.shape == (arr.shape[0] - window_size, window_size, arr.shape[-1])
    return jnp.array(arr_strided)


""" Custom Logger """


class Logger:
    """ Trivial light-weight logger for writing output to the console and a log file.
        Not intended as full Logger with verbosity capabilities.
    """

    def __init__(self, filename, stream=sys.stdout):
        self.stream = stream
        self.file = open(filename, 'a')

    def write(self, message):
        self.stream.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()


""" Async executor """


class AsyncExecutor:

    def __init__(self, n_jobs=1):
        self.num_workers = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        self._pool = []
        self._populate_pool()

    def run(self, target, *args_iter, verbose=False):
        workers_idle = [False] * self.num_workers
        tasks = list(zip(*args_iter))
        n_tasks = len(tasks)

        while not all(workers_idle):
            for i in range(self.num_workers):
                if not self._pool[i].is_alive():
                    self._pool[i].terminate()
                    if len(tasks) > 0:
                        if verbose:
                            print(n_tasks - len(tasks))
                        next_task = tasks.pop(0)
                        self._pool[i] = _start_process(target, next_task)
                    else:
                        workers_idle[i] = True

    def _populate_pool(self):
        self._pool = [_start_process(_dummy_fun) for _ in range(self.num_workers)]


def _start_process(target, args=None):
    if args:
        p = multiprocessing.Process(target=target, args=args)
    else:
        p = multiprocessing.Process(target=target)
    p.start()
    return p


def _dummy_fun():
    pass


""" Command generators """


def generate_base_command(module, flags: Optional[Dict[str, Any]] = None, unbuffered: bool = True) -> str:
    """ Generates the command to execute python module with provided flags

    Args:
        module: python module / file to run
        flags: dictionary of flag names and the values to assign to them.
               assumes that boolean flags are encoded as store_true flags with False as default.
        unbuffered: whether to invoke an unbuffered python output stream

    Returns: (str) command which can be executed via bash

    """

    """ Module is a python file to execute """
    interpreter_script = sys.executable
    base_exp_script = os.path.abspath(module.__file__)
    if unbuffered:
        base_cmd = interpreter_script + ' -u ' + base_exp_script
    else:
        base_cmd = interpreter_script + ' ' + base_exp_script
    if flags is not None:
        assert isinstance(flags, dict), "Flags must be provided as dict"
        for flag, setting in flags.items():
            if type(setting) == bool or type(setting) == np.bool_:
                if setting:
                    base_cmd += f" --{flag}"
            else:
                base_cmd += f" --{flag}={setting}"
    return base_cmd


def generate_run_commands(command_list: List[str], output_file_list: Optional[List[str]] = None,
                          num_cpus: int = 1, num_gpus: int = 0,
                          dry: bool = False, mem: int = 2 * 1028, duration: str = '3:59:00',
                          mode: str = 'local', prompt: bool = True) -> None:
    if mode == 'euler':
        cluster_cmds = []
        bsub_cmd = 'sbatch ' + \
                   f'--time={duration} ' + \
                   f'--mem-per-cpu={mem} ' + \
                   f'--cpus-per-task {num_cpus} '

        if num_gpus > 0:
            bsub_cmd += f'-G {num_gpus} --gres=gpumem:10240m '

        assert output_file_list is None or len(command_list) == len(output_file_list)
        if output_file_list is None:
            for cmd in command_list:
                cluster_cmds.append(bsub_cmd + f'--wrap="{cmd}"')
        else:
            for cmd, output_file in zip(command_list, output_file_list):
                cluster_cmds.append(bsub_cmd + f'--output={output_file} --wrap="{cmd}"')

        if dry:
            for cmd in cluster_cmds:
                print(cmd)
        else:
            if prompt:
                answer = input(f"about to launch {len(command_list)} jobs with {num_cpus} "
                               f"cores each. proceed? [yes/no]")
            else:
                answer = 'yes'
            if answer == 'yes':
                for cmd in cluster_cmds:
                    os.system(cmd)

    elif mode == 'local':
        if prompt:
            answer = input(f"about to run {len(command_list)} jobs in a loop. proceed? [yes/no]")
        else:
            answer = 'yes'

        if answer == 'yes':
            for cmd in command_list:
                if dry:
                    print(cmd)
                else:
                    os.system(cmd)

    elif mode == 'local_async':
        if prompt:
            answer = input(f"about to launch {len(command_list)} commands in {num_cpus} "
                           f"local processes. proceed? [yes/no]")
        else:
            answer = 'yes'

        if answer == 'yes':
            if dry:
                for cmd in command_list:
                    print(cmd)
            else:
                executor = AsyncExecutor(n_jobs=num_cpus)
                executor.run(lambda command: os.system(command), command_list)
    else:
        raise NotImplementedError


""" Hashing and Encoding dicts to JSON """


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)


def hash_dict(d: Dict) -> str:
    dhash = hashlib.md5()
    dhash.update(json.dumps(d, sort_keys=True, cls=NumpyArrayEncoder, indent=4).encode())
    return dhash.hexdigest()


""" Randomly sampling flags """


def sample_flag(sample_spec, rds=None):
    if rds is None:
        rds = np.random
    assert len(sample_spec) == 2

    sample_type, sample_range = sample_spec
    if sample_type == 'loguniform':
        assert len(sample_range) == 2
        return 10 ** rds.uniform(*sample_range)
    elif sample_type == 'uniform':
        assert len(sample_range) == 2
        return rds.uniform(*sample_range)
    elif sample_type == 'choice':
        return rds.choice(sample_range)
    else:
        raise NotImplementedError


def sample_param_flags(hyperparam_spec: Dict[str, Dict], rds: Optional[np.random.RandomState] = None) -> Dict[str, Any]:
    if rds is None:
        rds = np.random
    flags_dict = {}
    for flag, flag_spec in hyperparam_spec.items():
        if 'value' in flag_spec:
            flags_dict[flag] = flag_spec['value']
        elif 'values' in flag_spec:
            flags_dict[flag] = rds.choice(flag_spec['values'])
        elif 'distribution' in flag_spec:
            if flag_spec['distribution'] == 'uniform':
                flags_dict[flag] = rds.uniform(flag_spec['min'], flag_spec['max'])
            elif flag_spec['distribution'] == 'log_uniform_10':
                flags_dict[flag] = 10 ** rds.uniform(flag_spec['min'], flag_spec['max'])
            elif flag_spec['distribution'] == 'log_uniform':
                flags_dict[flag] = np.exp(rds.uniform(flag_spec['min'], flag_spec['max']))
            else:
                raise ValueError(f"Unknown distribution {flag_spec['distribution']}")
        else:
            raise NotImplementedError(f'Unable to process hyperparam spec {flag_spec}')
    assert flags_dict.keys() == hyperparam_spec.keys()
    return flags_dict


""" Collecting the exp result"""


def collect_exp_results(exp_name: str, dir_tree_depth: int = 3, verbose: bool = True,
                        add_dirname: bool = True):
    exp_dir = os.path.join(RESULT_DIR, exp_name)
    no_results_counter = 0
    success_counter = 0
    exp_dicts = []
    param_names = set()
    search_path = os.path.join(exp_dir, '/'.join(['*' for _ in range(dir_tree_depth)]) + '.json')
    results_jsons = glob.glob(search_path)
    for results_file in results_jsons:

        if os.path.isfile(results_file):
            try:
                with open(results_file, 'r') as f:
                    exp_dict = json.load(f)
                if isinstance(exp_dict, dict):
                    exp_dict_merged = {**exp_dict['evals'], **exp_dict['params']}  # put params and evals in one dict
                    if add_dirname:
                        exp_dict_merged['dirname'] = os.path.dirname(results_file)
                    exp_dicts.append(exp_dict_merged)
                    param_names = param_names.union(set(exp_dict['params'].keys()))
                elif isinstance(exp_dict, list):
                    exp_dicts.extend([{**d['evals'], **d['params']} for d in exp_dict])
                    for d in exp_dict:
                        param_names = param_names.union(set(d['params'].keys()))
                else:
                    raise ValueError
                success_counter += 1
            except json.decoder.JSONDecodeError as e:
                print(f'Failed to load {results_file}', e)
        else:
            no_results_counter += 1

    assert success_counter + no_results_counter == len(results_jsons)
    if verbose:
        print(f'Parsed results in {search_path} - found {success_counter} folders with results'
              f' and {no_results_counter} folders without results')

    param_names.add('dirname')

    return pd.DataFrame(data=exp_dicts), list(param_names)


""" Some aggregation functions """


def ucb(row):
    return np.quantile(row, q=0.95, axis=0)


def lcb(row):
    return np.quantile(row, q=0.05, axis=0)


def median(row):
    return np.quantile(row, q=0.5, axis=0)


def count(row):
    return row.shape[0]


def dict_permutations(d: dict) -> List[dict]:
    keys = d.keys()
    values = d.values()
    perms = []

    # Calculate the Cartesian product of all values in the dictionary
    for value_combo in itertools.product(*values):
        perms.append(dict(zip(keys, value_combo)))

    return perms


if __name__ == '__main__':
    # Example for dict_permutations
    d = {
        "A": [1, 2],
        "B": ["x", "y"],
        "C": ["!", "@"]
    }

    result = dict_permutations(d)
    for r in result:
        print(r)
