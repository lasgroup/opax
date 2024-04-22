import jax
import math

from opax.models.dynamics_model import DynamicsModel
from opax.optimizers.cem_trajectory_optimizer import CemTO
from opax.optimizers.trajax_trajectory_optimizer import TraJaxTO
from opax.optimizers.sac_based_optimizer import SACOptimizer
from opax.optimizers.icem_trajectory_optimizer import ICemTO
import gym
from opax.utils.replay_buffer import ReplayBuffer, Transition
from opax.utils.utils import get_idx
from opax.agents.dummy_agent import DummyAgent
import jax.numpy as jnp
import wandb
from typing import Union, Optional, Dict, Any
from opax.models.hucrl_model import HUCRLModel
from opax.models.gp_dynamics_model import GPDynamicsModel
from jaxtyping import PyTree

Model_list = list[DynamicsModel]


class ModelBasedAgent(DummyAgent):
    """
    action_space: gym.spaces.box
    observation_space: gym.spaces.box
    dynamics_model: Union[DynamicsModel, Model_list], if multiple rewards a list of dynamics model is passed
    policy_optimizer_name: str, name for the policy optimizer used for control
    horizon: int, planning horizon for optimizer
    n_particles: int, number of particles used in planning
    reset_model: bool, if model should reset after every update
    reset_model_opt_state: bool, if model optimizer state should reset after every update
    calibrate_model: bool, if model calibration should be done
    init_function: bool, if init function for the model should be called
    optimizer_kwargs: Optional[Dict[str, Any]], kwargs for the model based policy optimizer
    reset_optimizer_params_for: int, number of initial training steps for which the policy optimizer is reset
    log_full_training: bool, if model training should be logged
    log_agent_training: bool, if agent training should be logged
    """
    def __init__(
            self,
            action_space: gym.spaces.box,
            observation_space: gym.spaces.box,
            dynamics_model: Union[DynamicsModel, Model_list],
            policy_optimizer_name: str = "CemTO",
            horizon: int = 100,
            n_particles: int = 10,
            reset_model: bool = False,
            reset_model_opt_state: bool = True,
            calibrate_model: bool = True,
            init_function: bool = True,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            reset_optimizer_params_for: int = 5,
            log_full_training: bool = False,
            log_agent_training: bool = False,
            *args,
            **kwargs,
    ):
        super(ModelBasedAgent, self).__init__(*args, **kwargs)
        self.action_space = action_space
        self.observation_space = observation_space
        if isinstance(dynamics_model, DynamicsModel):
            self.dynamics_model_list = [dynamics_model]
            self.num_dynamics_models = 1
        else:
            self.dynamics_model_list = dynamics_model
            self.num_dynamics_models = len(dynamics_model)
        assert policy_optimizer_name in ["CemTO", "TraJaxTO", "SacOpt", "iCemTO"], "Optimizer must be CEM, TraJax, " \
                                                                                   "SAC " \
                                                                                   "or iCem"

        optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
        action_dim = self.action_space.shape
        if isinstance(self.dynamics_model, HUCRLModel) or isinstance(self.dynamics_model, GPDynamicsModel):
            action_dim = (self.action_space.shape[0] + self.observation_space.shape[0], )
        if policy_optimizer_name == "CemTO":
            self.policy_optimizer = CemTO(
                dynamics_model_list=self.dynamics_model_list,
                horizon=horizon,
                action_dim=action_dim,
                n_particles=n_particles,
                **optimizer_kwargs,
            )
        elif policy_optimizer_name == "TraJaxTO":
            self.policy_optimizer = TraJaxTO(
                dynamics_model_list=self.dynamics_model_list,
                horizon=horizon,
                action_dim=action_dim,
                n_particles=n_particles,
                **optimizer_kwargs,
            )
        elif policy_optimizer_name == "SacOpt":
            self.policy_optimizer = SACOptimizer(
                dynamics_model_list=self.dynamics_model_list,
                horizon=horizon,
                action_dim=action_dim,
                **optimizer_kwargs,
            )
        elif policy_optimizer_name == 'iCemTO':
            self.policy_optimizer = ICemTO(
                dynamics_model_list=self.dynamics_model_list,
                horizon=horizon,
                n_particles=n_particles,
                action_dim=action_dim,
                **optimizer_kwargs,
            )
        self.n_particles = n_particles
        self.reset_model = reset_model
        self.reset_model_opt_state = reset_model_opt_state
        self.calibrate_model = calibrate_model
        self.reset_optimizer_params_for = reset_optimizer_params_for
        self.update_steps = 0
        self.log_agent_training = log_agent_training
        self.log_full_training = log_full_training
        if init_function:
            self._init_fn()

    def _init_fn(self):
        """Creates model training function"""
        def step(carry, ins):
            rng = carry[0]
            model_params = carry[2]
            model_opt_state = carry[3]
            tran = ins[0]
            val_tran = ins[1]
            train_rng, rng = jax.random.split(rng, 2)

            (
                new_model_params,
                new_model_opt_state,
                alpha,
                summary,
            ) = \
                self.dynamics_model._train_step(
                    tran=tran,
                    model_params=model_params,
                    model_opt_state=model_opt_state,
                    val=val_tran,
                    rng=train_rng,
                )
            carry = [
                rng,
                alpha,
                new_model_params,
                new_model_opt_state,
            ]
            outs = [summary]
            return carry, outs

        self.step = jax.jit(step)

    def act_in_jax(self, obs: jax.Array, rng: jax.random.PRNGKeyArray, eval: bool = False, eval_idx: int = 0):
        # forward pass through policy if policy optimizer is SAC
        if isinstance(self.policy_optimizer, SACOptimizer):
            if eval:
                action = self.policy_optimizer.get_action_for_eval(obs=obs, rng=rng, agent_idx=eval_idx)
            else:
                action = self.policy_optimizer.get_action_for_exploration(obs=obs, rng=rng)
            action = action[..., :self.action_space.shape[0]]
        else:
            # optimize for actions and take the first best one --> MPC
            if eval:
                dim_state = obs.shape[-1]
                obs = obs.reshape(-1, dim_state)
                n_envs = obs.shape[0]
                rollout_rng, optimizer_rng = jax.random.split(rng, 2)
                rollout_rng = jax.random.split(rollout_rng, n_envs)
                optimizer_rng = jax.random.split(optimizer_rng, n_envs)
                optimize_fn = self.policy_optimizer.optimize_for_eval_fns[eval_idx]
                action_sequence, best_reward = optimize_fn(
                    dynamics_params=self.dynamics_model.model_params,
                    obs=obs,
                    key=rollout_rng,
                    optimizer_key=optimizer_rng,
                    model_props=self.dynamics_model.model_props,
                )
                action = action_sequence[:, 0, ...]
                if action.shape[0] == 1:
                    action = action.squeeze(0)
            else:
                n_envs = obs.shape[0]
                rollout_rng, optimizer_rng = jax.random.split(rng, 2)
                rollout_rng = jax.random.split(rollout_rng, n_envs)
                optimizer_rng = jax.random.split(optimizer_rng, n_envs)
                action_sequence, best_reward = self.policy_optimizer.optimize_for_exploration(
                        dynamics_params=self.dynamics_model.model_params,
                        obs=obs,
                        key=rollout_rng,
                        optimizer_key=optimizer_rng,
                        model_props=self.dynamics_model.model_props,
                    )
                action = action_sequence[:, 0, ...]
            action = action[..., :self.action_space.shape[0]]
        return action

    def train_step(self,
                   rng: jax.random.PRNGKeyArray,
                   buffer: ReplayBuffer,
                   validate: bool = True,
                   log_results: bool = True,
                   ) -> int:
        """
        Training of model based agent
        :param rng: jax.random.PRNGKeyArray, random key
        :param buffer: ReplayBuffer, buffer of collected transition
        :param validate: bool, boolean for validating model
        :param log_results: bool, boolean to indicate logging of training results.
        :return total_train_steps: int, total number of training steps done for the model
        """
        # if its a GP fit the model with all transitions
        if self.is_gp:
            transitions = buffer.get_full_normalized_data()
            train_rng, rng = jax.random.split(rng, 2)
            model_params = self.dynamics_model.init_model_params
            model_opt_state = self.dynamics_model.init_model_opt_state
            alpha = jnp.ones(self.observation_space.shape)
            if self.num_epochs > 0:
                total_train_steps = math.ceil(buffer.size * self.num_epochs)
            else:
                total_train_steps = self.train_steps
            (
                model_params,
                model_opt_state,
                alpha,
                summary,
            ) = \
                self.dynamics_model._train_step(
                    tran=transitions,
                    model_params=model_params,
                    model_opt_state=model_opt_state,
                    rng=train_rng,
                    num_steps=total_train_steps,
                )
        else:
            # perform train steps on the NN model.
            max_train_steps_per_iter = 8000
            if self.num_epochs > 0:
                total_train_steps = math.ceil(buffer.size * self.num_epochs / self.batch_size)
            else:
                total_train_steps = self.train_steps
            self.update_steps += 1
            total_train_steps = min(total_train_steps, self.max_train_steps)
            train_loops = math.ceil(total_train_steps / max_train_steps_per_iter)
            train_steps = min(max_train_steps_per_iter, total_train_steps)
            train_loops = max(train_loops, 1)
            if self.reset_model:
                model_params = self.dynamics_model.init_model_params
                model_opt_state = self.dynamics_model.init_model_opt_state
            else:
                model_params = self.dynamics_model.model_params
                if self.reset_model_opt_state:
                    model_opt_state = self.dynamics_model.init_model_opt_state
                else:
                    model_opt_state = self.dynamics_model.model_opt_state
            alpha = jnp.ones(self.observation_space.shape)
            for i in range(train_loops):
                train_rng, rng = jax.random.split(rng, 2)
                transitions = buffer.sample(train_rng, batch_size=int(self.batch_size * train_steps))
                transitions = transitions.reshape(train_steps, self.batch_size)
                val_transitions = None
                if validate:
                    train_rng, val_rng = jax.random.split(train_rng, 2)
                    val_transitions = buffer.sample(val_rng, batch_size=int(self.batch_size * train_steps))
                    val_transitions = val_transitions.reshape(train_steps, self.batch_size)
                carry = [
                    train_rng,
                    alpha,
                    model_params,
                    model_opt_state,
                ]
                ins = [transitions, val_transitions]
                carry, outs = jax.lax.scan(self.step, carry, ins, length=train_steps)
                model_params = carry[2]
                model_opt_state = carry[3]
                alpha = carry[1]
                summary = outs[0]
                if log_results:
                    if self.log_full_training:
                        for i in range(total_train_steps):
                            wandb.log(get_idx(summary, i).dict())
                    else:
                        for i in range(0, total_train_steps, 10):
                            wandb.log(get_idx(summary, i).dict())
            if self.calibrate_model:
                alpha = carry[1]
        self.update_models(model_params=model_params, model_opt_state=model_opt_state, alpha=alpha)
        # train policy if SAC optimizer is used.
        if isinstance(self.policy_optimizer, SACOptimizer):
            if buffer.size > self.policy_optimizer.transitions_per_update:
                train_rng = carry[0]
                policy_train_rng, train_rng = jax.random.split(train_rng, 2)
                policy_agent_train_summary, policy_train_rewards = self.policy_optimizer.train(
                    rng=policy_train_rng,
                    buffer=buffer,
                    dynamics_params=model_params,
                    model_props=self.dynamics_model.model_props,
                    reset_agent=self.reset_agent_params,
                )
                if log_results and self.log_agent_training:
                    # for j in range(self.policy_optimizer.train_steps_per_model_update):
                    for i in range(len(self.policy_optimizer.agent_list)):
                        summary = policy_agent_train_summary[i]
                        summary_dict = summary.dict()
                        summary_relabeled_dict = {}
                        for key, value in summary_dict.items():
                            summary_relabeled_dict[key + '_agent_' + str(i)] = value
                        wandb.log(
                            summary_relabeled_dict
                        )
                        reward_dict = {}
                        for reward in policy_train_rewards[i]:
                            reward_dict['sac_reward_agent_' + str(i)] = reward.item()
                            wandb.log(
                                reward_dict
                            )
        return total_train_steps

    def set_transforms(self,
                       bias_obs: Union[jnp.ndarray, float] = 0.0,
                       bias_act: Union[jnp.ndarray, float] = 0.0,
                       bias_out: Union[jnp.ndarray, float] = 0.0,
                       scale_obs: Union[jnp.ndarray, float] = 1.0,
                       scale_act: Union[jnp.ndarray, float] = 1.0,
                       scale_out: Union[jnp.ndarray, float] = 1.0,
                       ):
        """Set normalization factors for the dynamics models."""
        for i in range(len(self.dynamics_model_list)):
            self.dynamics_model_list[i].set_transforms(
                bias_obs=bias_obs,
                bias_act=bias_act,
                bias_out=bias_out,
                scale_obs=scale_obs,
                scale_act=scale_act,
                scale_out=scale_out,
            )

    def predict_next_state(self,
                           tran: Transition,
                           ):
        """Predict next state with learned model."""
        return self.dynamics_model.predict_raw(
            parameters=self.dynamics_model.model_params,
            tran=tran,
            model_props=self.dynamics_model.model_props,
        )

    def update_models(self, model_params: PyTree, model_opt_state: PyTree, alpha: Union[float, jax.Array] = 1.0):
        """Update model for params."""
        for i in range(len(self.dynamics_model_list)):
            self.dynamics_model_list[i].update_model(
                model_params=model_params,
                model_opt_state=model_opt_state,
                alpha=alpha,
            )

    @property
    def dynamics_model(self) -> DynamicsModel:
        return self.dynamics_model_list[0]

    @property
    def reset_agent_params(self) -> bool:
        return self.update_steps < self.reset_optimizer_params_for

    def prepare_agent_for_rollout(self):
        self.policy_optimizer.reset()

    def update_posterior(self, buffer: ReplayBuffer):
        for i in range(len(self.dynamics_model_list)):
            self.dynamics_model_list[i].update_model_posterior(buffer=buffer)

    @property
    def is_gp(self) -> bool:
        return isinstance(self.dynamics_model, GPDynamicsModel)
