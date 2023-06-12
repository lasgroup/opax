from argparse_dataclass import ArgumentParser
from typing import Any
import yaml
from mbse.trainer.model_based.model_based_trainer import ModelBasedTrainer as Trainer
from mbse.agents.model_based.model_based_agent import ModelBasedAgent
from dataclasses import dataclass, field
import wandb
from gym.wrappers.time_limit import TimeLimit
from gym.wrappers.rescale_action import RescaleAction
from mbse.models.environment_models.pendulum_swing_up import PendulumSwingUpEnv, PendulumDynamicsModel
from mbse.optimizers.cross_entropy_optimizer import CrossEntropyOptimizer
from mbse.utils.vec_env.env_util import make_vec_env
from mbse.models.hucrl_model import HUCRLModel

OptState = Any
from jax.config import config

config.update("jax_log_compiles", 1)


@dataclass
class Experiment:
    """Definition of Experiment dataclass."""
    config: str = field(
        metadata=dict(help="File with config.")
    )


if __name__ == "__main__":
    parser = ArgumentParser(Experiment)
    args = parser.parse_args()
    with open(args.config, "r") as file:
        kwargs = yaml.safe_load(file)
    wrapper_cls = lambda x: RescaleAction(
        TimeLimit(x, max_episode_steps=kwargs['time_limit']),
        min_action=-1,
        max_action=1,
    )
    n_envs = 4
    horizon = 20
    env = make_vec_env(PendulumSwingUpEnv, wrapper_class=wrapper_cls, n_envs=n_envs)

    reward_model = env.envs[0].reward_model()
    reward_model.set_bounds(max_action=1.0)
    # dynamics_model = PendulumDynamicsModel(env=env.envs[0])
    dynamics_model = HUCRLModel(
        action_space=env.action_space,
        observation_space=env.observation_space,
        num_ensemble=5,
        reward_model=reward_model,
        features=[128, 128],
        pred_diff=True,
    )
    # policy_optimizer = GradientBasedOptimizer(
    #    upper_bound=1,
    #    num_samples=50,
    #    num_steps=10,
    #    action_dim=(horizon, env.action_space.shape[0])
    # )
    optimizer_kwargs = {
        'num_samples': 500,
        'num_elites': 50,
        'num_steps': 10,
    }
    agent = ModelBasedAgent(
        train_steps=kwargs['agent']['train_steps'],
        batch_size=kwargs['agent']['batch_size'],
        action_space=env.action_space,
        observation_space=env.observation_space,
        dynamics_model=dynamics_model,
        n_particles=10,
        policy_optimizer_name="TraJaxTO",
        horizon=horizon,
        optimizer_kwargs=optimizer_kwargs,
        # policy_optimizer=policy_optimizer,
    )

    USE_WANDB = True

    trainer = Trainer(
        agent=agent,
        # model_free_agent=model_free_agent,
        env=env,
        buffer_size=kwargs['trainer']['buffer_size'],
        total_train_steps=kwargs['trainer']['total_train_steps'],
        exploration_steps=kwargs['trainer']['exploration_steps'],
        use_wandb=USE_WANDB,
        eval_episodes=kwargs['trainer']['eval_episodes'],
        eval_freq=kwargs['trainer']['eval_freq'],
        train_freq=kwargs['trainer']['train_freq'],
        rollout_steps=kwargs['trainer']['rollout_steps'],
        normalize=True,
        action_normalize=True,
        learn_deltas=dynamics_model.pred_diff,
        validate=True,
        record_test_video=False,
    )
    if USE_WANDB:
        wandb.init(
            project=kwargs['project_name'],
        )
    trainer.train()
