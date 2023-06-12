"""
    Credits: https://github.com/sfujim/TD3
"""
from argparse_dataclass import ArgumentParser
from typing import Any
import yaml
from mbse.trainer.off_policy.off_policy_trainer import OffPolicyTrainer as Trainer
from mbse.agents.actor_critic.sac import SACAgent
from dataclasses import dataclass, field
import wandb
from gym.wrappers.time_limit import TimeLimit
from gym.wrappers.rescale_action import RescaleAction
from mbse.utils.vec_env.env_util import make_vec_env
from mbse.models.environment_models.pendulum_swing_up import CustomPendulumEnv


OptState = Any
# from jax.config import config
# config.update("jax_log_compiles", 1)

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
    # env = make_vec_env(kwargs['env_id'], wrapper_class=wrapper_cls, n_envs=10)
    env = make_vec_env(CustomPendulumEnv, wrapper_class=wrapper_cls, n_envs=10)

    agent = SACAgent(
                train_steps=kwargs['agent']['train_steps'],
                batch_size=kwargs['agent']['batch_size'],
                action_space=env.action_space,
                observation_space=env.observation_space,
                discount=kwargs['agent']['discount'],
                lr_actor=kwargs['agent']['lr_actor'],
                lr_critic=kwargs['agent']['lr_critic'],
                lr_alpha=kwargs['agent']['lr_alpha'],
                actor_features=kwargs['agent']['actor_features'],
                critic_features=kwargs['agent']['critic_features'],
                scale_reward=kwargs['agent']['scale_reward'],
                tau=kwargs['agent']['tau'],
                tune_entropy_coef=kwargs['agent']['tune_entropy_coef'],
                init_ent_coef=kwargs['agent']['init_ent_coef']
        )

    USE_WANDB = True
    trainer = Trainer(
        env=env,
        agent=agent,
        buffer_size=kwargs['trainer']['buffer_size'],
        total_train_steps=kwargs['trainer']['total_train_steps'],
        exploration_steps=kwargs['trainer']['exploration_steps'],
        use_wandb=USE_WANDB,
        eval_episodes=kwargs['trainer']['eval_episodes'],
        eval_freq=kwargs['trainer']['eval_freq'],
        train_freq=kwargs['trainer']['train_freq'],
        rollout_steps=kwargs['trainer']['rollout_steps'],
    )
    if USE_WANDB:
        wandb.init(
            project=kwargs['project_name'],
        )
    #import jax
    #with jax.disable_jit():
    trainer.train()


