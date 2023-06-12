from gym.wrappers import RescaleAction, TimeLimit
from mbse.utils.vec_env.env_util import make_vec_env
from mbse.agents.actor_critic.sac import SACAgent
from mbse.trainer.off_policy.off_policy_trainer import OffPolicyTrainer as Trainer
import numpy as np
import time
import json
import os
import sys
import argparse
from experiments.util import Logger, hash_dict, NumpyArrayEncoder
import wandb
from mbse.models.environment_models.halfcheetah_reward_model import HalfCheetahReward
from mbse.envs.wrappers.action_repeat import ActionRepeat


def experiment(logs_dir: str, use_wandb: bool, time_limit: int, n_envs: int, exp_name: str,
               discount: float, lr_actor: float, lr_critic: float, lr_alpha: float, hidden_layers_actor: int,
               num_neurons_actor: int, hidden_layers_critic: int, num_neurons_critic: int,
               tau: float, scale_reward: float, tune_entropy_coef: bool, init_ent_coef: float,
               batch_size: int, eval_freq: int, total_train_steps: int, buffer_size: int,
               exploration_steps: int, eval_episodes: int, train_freq: int, train_steps: int, rollout_steps: int,
               seed: int, exploration_strategy: str):
    """ Run experiment for a given method and environment. """

    """ Environment """
    action_repeat = 1
    import math
    time_lim = math.ceil(time_limit / action_repeat)
    wrapper_cls = lambda x: RescaleAction(
        TimeLimit(
            ActionRepeat(x, repeat=action_repeat),
            max_episode_steps=time_lim),
        min_action=-1,
        max_action=1,
    )

    if exploration_strategy == 'PetsCheetah':
        reward_model_forward = HalfCheetahReward(forward_velocity_weight=1.0, ctrl_cost_weight=0.1)
        env_kwargs_forward = {
            'reward_model': reward_model_forward,
            'render_mode': 'rgb_array'
        }
        from mbse.envs.pets_halfcheetah import HalfCheetahEnvDM
        env = make_vec_env(env_id=HalfCheetahEnvDM, wrapper_class=wrapper_cls, n_envs=n_envs, seed=seed,
                           env_kwargs=env_kwargs_forward)
    else:
        from dm_control.suite.cheetah import run
        from mbse.envs.dm_control_env import DeepMindBridge
        env_dm = run(time_limit=float('inf'), environment_kwargs={'flat_observation': True})
        env = lambda *kwargs: DeepMindBridge(env=env_dm)
        env = make_vec_env(env_id=env, wrapper_class=wrapper_cls, n_envs=n_envs, seed=seed)
    video_prefix = ""
    actor_features = [num_neurons_actor] * hidden_layers_actor
    critic_features = [num_neurons_critic] * hidden_layers_critic
    agent = SACAgent(
        train_steps=train_steps,
        batch_size=batch_size,
        action_space=env.action_space,
        observation_space=env.observation_space,
        discount=discount,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        lr_alpha=lr_alpha,
        actor_features=actor_features,
        critic_features=critic_features,
        scale_reward=scale_reward,
        tau=tau,
        tune_entropy_coef=tune_entropy_coef,
        init_ent_coef=init_ent_coef
    )

    USE_WANDB = use_wandb
    trainer = Trainer(
        env=env,
        agent=agent,
        buffer_size=buffer_size,
        total_train_steps=total_train_steps,
        exploration_steps=exploration_steps,
        use_wandb=USE_WANDB,
        eval_episodes=eval_episodes,
        eval_freq=eval_freq,
        train_freq=train_freq,
        rollout_steps=rollout_steps,
    )

    group_name = exploration_strategy
    if USE_WANDB:
        wandb.init(
            dir=logs_dir,
            project=exp_name,
            group=group_name,
        )
    trainer.train()

    result_dict = {
    }
    return result_dict


def main(args):
    """"""
    from pprint import pprint
    print(args)
    """ generate experiment hash and set up redirect of output streams """
    exp_hash = hash_dict(args.__dict__)
    if args.exp_result_folder is not None:
        os.makedirs(args.exp_result_folder, exist_ok=True)
        log_file_path = os.path.join(args.exp_result_folder, '%s.log ' % exp_hash)
        logger = Logger(log_file_path)
        sys.stdout = logger
        sys.stderr = logger

    pprint(args.__dict__)
    print('\n ------------------------------------ \n')

    """ Experiment core """
    t_start = time.time()
    np.random.seed(args.seed + 5)

    eval_metrics = experiment(
        logs_dir=args.logs_dir,
        use_wandb=args.use_wandb,
        exp_name=args.exp_name,
        time_limit=args.time_limit,
        n_envs=args.n_envs,
        discount=args.discount,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        lr_alpha=args.lr_alpha,
        hidden_layers_actor=args.hidden_layers_actor,
        num_neurons_actor=args.num_neurons_actor,
        hidden_layers_critic=args.hidden_layers_critic,
        num_neurons_critic=args.num_neurons_critic,
        tau=args.tau,
        scale_reward=args.scale_reward,
        tune_entropy_coef=args.tune_entropy_coef,
        init_ent_coef=args.init_ent_coef,
        batch_size=args.batch_size,
        eval_freq=args.eval_freq,
        total_train_steps=args.total_train_steps,
        buffer_size=args.buffer_size,
        exploration_steps=args.exploration_steps,
        eval_episodes=args.eval_episodes,
        train_freq=args.train_freq,
        train_steps=args.train_steps,
        rollout_steps=args.rollout_steps,
        seed=args.seed,
        exploration_strategy=args.exploration_strategy,
    )

    t_end = time.time()

    """ Save experiment results and configuration """
    results_dict = {
        'evals': eval_metrics,
        'params': args.__dict__,
        'duration_total': t_end - t_start
    }

    if args.exp_result_folder is None:
        from pprint import pprint
        pprint(results_dict)
    else:
        exp_result_file = os.path.join(args.exp_result_folder, '%s.json' % exp_hash)
        with open(exp_result_file, 'w') as f:
            json.dump(results_dict, f, indent=4, cls=NumpyArrayEncoder)
        print('Dumped results to %s' % exp_result_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Active-Exploration-run')

    # general experiment args
    parser.add_argument('--exp_name', type=str, default='active_exploration_sac')
    parser.add_argument('--logs_dir', type=str, default='./')
    parser.add_argument('--use_wandb', default=False, action="store_true")
    # env experiment args
    parser.add_argument('--time_limit', type=int, default=1000)
    parser.add_argument('--n_envs', type=int, default=1)

    # agent experiment args
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--lr_actor', type=float, default=0.001)
    parser.add_argument('--lr_critic', type=float, default=0.001)
    parser.add_argument('--lr_alpha', type=float, default=0.001)
    parser.add_argument('--hidden_layers_actor', type=int, default=2)
    parser.add_argument('--hidden_layers_critic', type=int, default=2)
    parser.add_argument('--num_neurons_actor', type=int, default=256)
    parser.add_argument('--num_neurons_critic', type=int, default=256)
    parser.add_argument('--tau', type=float, default=0.0005)
    parser.add_argument('--scale_reward', type=float, default=1.0)
    parser.add_argument('--tune_entropy_coef', default=True, action="store_true")
    parser.add_argument('--init_ent_coef', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=256)

    # trainer experiment args
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--total_train_steps', type=int, default=10000)
    parser.add_argument('--buffer_size', type=int, default=1000000)
    parser.add_argument('--exploration_steps', type=int, default=0)
    parser.add_argument('--eval_episodes', type=int, default=1)
    parser.add_argument('--train_freq', type=int, default=1)
    parser.add_argument('--train_steps', type=int, default=1500)
    parser.add_argument('--rollout_steps', type=int, default=250)
    parser.add_argument('--exploration_strategy', type=str, default='true')


    # general args
    parser.add_argument('--exp_result_folder', type=str, default=None)
    parser.add_argument('--seed', type=int, default=834, help='random number generator seed')

    args = parser.parse_args()
    main(args)
