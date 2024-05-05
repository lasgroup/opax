from gym.wrappers import RescaleAction, TimeLimit
from opax.utils.vec_env.env_util import make_vec_env
from opax.utils.vec_env.subproc_vec_env import SubprocVecEnv
from opax.models.environment_models.pendulum_swing_up import PendulumReward, CustomPendulumEnv, PendulumDynamicsModel
from opax.models.active_learning_model import ActiveLearningHUCRLModel, ActiveLearningPETSModel
from opax.models.hucrl_model import HUCRLModel
from opax.agents.model_based.model_based_agent import ModelBasedAgent
from opax.trainer.model_based.model_based_trainer import ModelBasedTrainer as Trainer
import numpy as np
import time
import json
import os
import sys
import argparse
from experiments.util import Logger, hash_dict, NumpyArrayEncoder
import wandb
from typing import Optional


def experiment(logs_dir: str, use_wandb: bool, exp_name: str, time_limit: int, n_envs: int,
               optimizer_type: str,
               num_samples: int, num_elites: int, num_steps: int, horizon: int, n_particles: int, reset_model: bool,
               num_ensembles: int, hidden_layers: int, num_neurons: int, beta: float, deterministic: bool,
               max_train_steps: int, pred_diff: bool, batch_size: int, eval_freq: int, total_train_steps: int,
               buffer_size: int, exploration_steps: int, eval_episodes: int, train_freq: int, train_steps: int,
               num_epochs: int, rollout_steps: int, normalize: bool, action_normalize: bool, validate: bool,
               record_test_video: bool, validation_buffer_size: int, validation_batch_size: int,
               seed: int, exploration_strategy: str, use_log: bool, use_al: bool, action_cost: float = 0.0,
               time_limit_eval: Optional[int] = None,
               resume_path: Optional[str] = None):
    """ Run experiment for a given method and environment. """
    import jax.numpy as jnp
    """ Environment """
    # from jax.config import config
    sac_kwargs = {
        'discount': 0.99,
        'init_ent_coef': 1.0,
        'lr_actor': 0.001,
        'weight_decay_actor': 1e-5,
        'lr_critic': 0.001,
        'weight_decay_critic': 1e-5,
        'lr_alpha': 0.0005,
        'weight_decay_alpha': 0.0,
        'actor_features': [64, 64],
        'critic_features': [256, 256],
        'scale_reward': 1,
        'tune_entropy_coef': True,
        'tau': 0.005,
        'batch_size': 256,
        'train_steps': 500,
    }

    optimizer_kwargs = {
        'num_samples': num_samples,
        'num_elites': num_elites,
        'num_steps': num_steps,
        'exponent': 0.25,
        'train_steps_per_model_update': 25,
        'transitions_per_update': 500,
        'sac_kwargs': sac_kwargs,
        'sim_transitions_ratio': 0.0,
        'reset_actor_params': True,
    }
    lr = 5e-4
    # ctrl_cost = 0.1
    # env_kwargs = {'ctrl_cost': ctrl_cost}
    # config.update("jax_log_compiles", 1)
    wrapper_cls = lambda x: RescaleAction(
        TimeLimit(x, max_episode_steps=time_limit),
        min_action=-1,
        max_action=1,
    )
    wrapper_cls_test = wrapper_cls
    if time_limit_eval is not None:
        wrapper_cls_test = lambda x: RescaleAction(
            TimeLimit(x, max_episode_steps=time_limit_eval),
            min_action=-1,
            max_action=1,
        )
    env_kwargs_keep_down = {
        'target_angle': jnp.pi,
    }
    env = make_vec_env(env_id=CustomPendulumEnv, wrapper_class=wrapper_cls, n_envs=n_envs, seed=seed,
                       env_kwargs=env_kwargs_keep_down)
    env_kwargs_swing_up = {
        'target_angle': 0.0,
    }

    test_env_keep_down = make_vec_env(env_id=CustomPendulumEnv, wrapper_class=wrapper_cls, n_envs=1, seed=seed,
                                     env_kwargs=env_kwargs_keep_down)
    test_env_swing_up = make_vec_env(env_id=CustomPendulumEnv, wrapper_class=wrapper_cls, n_envs=1, seed=seed,
                                 env_kwargs=env_kwargs_swing_up)
                       # vec_env_cls=SubprocVecEnv)
    # test_env = make_vec_env(env_id=CustomPendulumEnv, wrapper_class=wrapper_cls_test, n_envs=1, seed=seed, env_kwargs=env_kwargs)
    test_env = [test_env_keep_down, test_env_swing_up]
    features = [num_neurons] * hidden_layers
    reward_model_keep_down = PendulumReward(action_space=env.action_space, target_angle=np.pi)
    reward_model_keep_down.set_bounds(max_action=1.0)
    reward_model_swing_up = PendulumReward(action_space=env.action_space, target_angle=0.0)
    reward_model_swing_up.set_bounds(max_action=1.0)
    video_prefix = ""
    if exploration_strategy == 'Mean':
        beta = 0.0
        video_prefix += 'Mean'

    if exploration_strategy == 'PETS':
        dynamics_model_keep_down = ActiveLearningPETSModel(
            action_space=env.action_space,
            observation_space=env.observation_space,
            num_ensemble=num_ensembles,
            reward_model=reward_model_keep_down,
            features=features,
            pred_diff=pred_diff,
            beta=beta,
            seed=seed,
            use_log_uncertainties=use_log,
            use_al_uncertainties=use_al,
            deterministic=deterministic,
            lr=lr,
            action_cost=action_cost,
        )

        dynamics_model_swing_up = ActiveLearningPETSModel(
            action_space=env.action_space,
            observation_space=env.observation_space,
            num_ensemble=num_ensembles,
            reward_model=reward_model_swing_up,
            features=features,
            pred_diff=pred_diff,
            beta=beta,
            seed=seed,
            use_log_uncertainties=use_log,
            use_al_uncertainties=use_al,
            deterministic=deterministic,
            lr=lr,
            action_cost=action_cost,
        )
        dynamics_model = [dynamics_model_keep_down, dynamics_model_swing_up]
        video_prefix += 'PETS'
    elif exploration_strategy == 'true_model':
        dynamics_model = PendulumDynamicsModel(env=test_env)
        video_prefix += 'true_model'
        validation_buffer_size = 0
        total_train_steps = 1
    elif exploration_strategy == 'HUCRL':
        dynamics_model_keep_down = HUCRLModel(
            action_space=env.action_space,
            observation_space=env.observation_space,
            num_ensemble=num_ensembles,
            reward_model=reward_model_keep_down,
            features=features,
            pred_diff=pred_diff,
            beta=beta,
            seed=seed,
            deterministic=deterministic,
            lr=lr,
        )

        dynamics_model_swing_up = HUCRLModel(
            action_space=env.action_space,
            observation_space=env.observation_space,
            num_ensemble=num_ensembles,
            reward_model=reward_model_swing_up,
            features=features,
            pred_diff=pred_diff,
            beta=beta,
            seed=seed,
            deterministic=deterministic,
            lr=lr,
        )
        dynamics_model = [dynamics_model_keep_down, dynamics_model_swing_up]
        video_prefix += 'HUCRL'

    else:
        dynamics_model_keep_down = ActiveLearningHUCRLModel(
            action_space=env.action_space,
            observation_space=env.observation_space,
            num_ensemble=num_ensembles,
            reward_model=reward_model_keep_down,
            features=features,
            pred_diff=pred_diff,
            beta=beta,
            seed=seed,
            use_log_uncertainties=use_log,
            use_al_uncertainties=use_al,
            deterministic=deterministic,
            lr=lr,
            action_cost=action_cost,
        )

        dynamics_model_swing_up = ActiveLearningHUCRLModel(
            action_space=env.action_space,
            observation_space=env.observation_space,
            num_ensemble=num_ensembles,
            reward_model=reward_model_swing_up,
            features=features,
            pred_diff=pred_diff,
            beta=beta,
            seed=seed,
            use_log_uncertainties=use_log,
            use_al_uncertainties=use_al,
            deterministic=deterministic,
            lr=lr,
            action_cost=action_cost,
        )
        dynamics_model = [dynamics_model_keep_down, dynamics_model_swing_up]

        video_prefix += 'Optimistic'
        if exploration_strategy == 'Uniform':
            video_prefix += 'Uniform'

    agent = ModelBasedAgent(
        train_steps=train_steps,
        batch_size=batch_size,
        max_train_steps=max_train_steps,
        num_epochs=num_epochs,
        action_space=env.action_space,
        observation_space=env.observation_space,
        dynamics_model=dynamics_model,
        n_particles=n_particles,
        reset_model=reset_model,
        policy_optimizer_name=optimizer_type,
        optimizer_kwargs=optimizer_kwargs,
        horizon=horizon,
    )

    USE_WANDB = use_wandb
    uniform_exploration = False
    if exploration_strategy == 'Uniform':
        uniform_exploration = True
    trainer = Trainer(
        agent=agent,
        env=env,
        test_env=test_env,
        buffer_size=buffer_size,
        total_train_steps=total_train_steps,
        exploration_steps=exploration_steps,
        use_wandb=USE_WANDB,
        eval_episodes=eval_episodes,
        eval_freq=eval_freq,
        train_freq=train_freq,
        rollout_steps=rollout_steps,
        normalize=normalize,
        action_normalize=action_normalize,
        learn_deltas=pred_diff,
        validate=validate,
        record_test_video=record_test_video,
        validation_buffer_size=validation_buffer_size,
        validation_batch_size=validation_batch_size,
        seed=seed,
        uniform_exploration=uniform_exploration,
        video_prefix=video_prefix,
        video_folder=logs_dir,
    )
    if resume_path is not None:
        print("Loading training checkpoint from", resume_path)
        trainer.load_agent(resume_path)
        if total_train_steps > trainer.total_train_steps:
            trainer.set_total_trainer_steps(total_train_steps)
    group_name = exploration_strategy
    if use_log:
        group_name += "_log_rewards"
    if use_al:
        group_name += "_aleatoric_rewards"
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
        optimizer_type=args.optimizer_type,
        num_samples=args.num_samples,
        num_elites=args.num_elites,
        num_steps=args.num_steps,
        horizon=args.horizon,
        n_particles=args.n_particles,
        reset_model=args.reset_model,
        beta=args.beta,
        deterministic=args.deterministic,
        num_ensembles=args.num_ensembles,
        pred_diff=args.pred_diff,
        batch_size=args.batch_size,
        max_train_steps=args.max_train_steps,
        eval_freq=args.eval_freq,
        total_train_steps=args.total_train_steps,
        buffer_size=args.buffer_size,
        exploration_steps=args.exploration_steps,
        eval_episodes=args.eval_episodes,
        train_freq=args.train_freq,
        train_steps=args.train_steps,
        num_epochs=args.num_epochs,
        rollout_steps=args.rollout_steps,
        normalize=args.normalize,
        action_normalize=args.action_normalize,
        validate=args.validate,
        record_test_video=args.record_test_video,
        validation_buffer_size=args.validation_buffer_size,
        validation_batch_size=args.validation_batch_size,
        seed=args.seed,
        hidden_layers=args.hidden_layers,
        num_neurons=args.num_neurons,
        exploration_strategy=args.exploration_strategy,
        use_log=args.use_log,
        use_al=args.use_al,
        time_limit_eval=args.time_limit_eval,
        action_cost=args.action_cost,
        resume_path=args.resume
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
    parser.add_argument('--exp_name', type=str, required=True, default='active_exploration')
    parser.add_argument('--logs_dir', type=str, default='./')
    parser.add_argument('--use_wandb', default=False, action="store_true")
    # env experiment args
    parser.add_argument('--time_limit', type=int, default=200)
    parser.add_argument('--n_envs', type=int, default=5)

    # optimizer experiment args
    parser.add_argument('--optimizer_type', type=str, default='iCemTO')
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--num_elites', type=int, default=50)
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--horizon', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=0.0)

    # agent experiment args
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_particles', type=int, default=10)
    parser.add_argument('--reset_model', default=True, action="store_true")

    # dynamics_model experiment args
    parser.add_argument('--num_ensembles', type=int, default=5)
    parser.add_argument('--hidden_layers', type=int, default=2)
    parser.add_argument('--num_neurons', type=int, default=128)
    parser.add_argument('--pred_diff', default=True, action="store_true")
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--deterministic', default=True, action="store_true")

    # trainer experiment args
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_train_steps', type=int, default=5000)
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--total_train_steps', type=int, default=2500)
    parser.add_argument('--buffer_size', type=int, default=1000000)
    parser.add_argument('--exploration_steps', type=int, default=0)
    parser.add_argument('--eval_episodes', type=int, default=1)
    parser.add_argument('--train_freq', type=int, default=1)
    parser.add_argument('--train_steps', type=int, default=5000)
    parser.add_argument('--num_epochs', type=int, default=-1)
    parser.add_argument('--rollout_steps', type=int, default=200)
    parser.add_argument('--normalize', default=True, action="store_true")
    parser.add_argument('--action_normalize', default=True, action="store_true")
    parser.add_argument('--validate', default=True, action="store_true")
    parser.add_argument('--record_test_video', default=False, action="store_true")
    parser.add_argument('--validation_buffer_size', type=int, default=100000)
    parser.add_argument('--validation_batch_size', type=int, default=4096)
    parser.add_argument('--exploration_strategy', type=str, default='Optimistic')
    parser.add_argument('--use_log', default=False, action="store_true")
    parser.add_argument('--use_al', default=False, action="store_true")
    parser.add_argument('--action_cost', type=float, default=0.0)
    parser.add_argument('--time_limit_eval', type=int, default=200)

    # general args
    parser.add_argument('--exp_result_folder', type=str, default=None)
    parser.add_argument('--seed', type=int, default=834, help='random number generator seed')

    parser.add_argument("--resume", type=str, default=None, help="Path to stored checkpoint from which to resume training")

    args = parser.parse_args()
    main(args)
