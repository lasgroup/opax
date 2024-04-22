from typing import Optional, NamedTuple, Callable

import chex

from opax.models.dynamics_model import DynamicsModel, ModelSummary
from opax.utils.type_aliases import ModelProperties
import gym
import numpy as np
from opax.models.reward_model import RewardModel
import jax.numpy as jnp
from opax.utils.models import GPModel, GPModelParameters
from opax.utils.utils import sample_normal_dist
import jax
from opax.utils.replay_buffer import Transition, ReplayBuffer
from jaxtyping import PyTree

EPS = 1e-6


class GPSamplingType(NamedTuple):
    optimistic: bool = True
    mean: bool = False
    random: bool = False


class GPDynamicsModel(DynamicsModel):
    def __init__(self,
                 action_space: gym.spaces.box,
                 observation_space: gym.spaces.box,
                 reward_model: RewardModel,
                 sampling_type: GPSamplingType = GPSamplingType(),
                 lr: float = 1e-2,
                 weight_decay: float = 1e-4,
                 seed: int = 0,
                 is_active_exploration_model: bool = False,
                 beta: float = 1.0,
                 action_cost: float = 0.0,
                 *args,
                 **kwargs
                 ):
        super(GPDynamicsModel, self).__init__(*args, **kwargs)
        self.reward_model = reward_model

        obs_dim = np.prod(observation_space.shape).item()
        act_dim = np.prod(action_space.shape).item()
        self.model = GPModel(
            input_dim=obs_dim + act_dim,
            output_dim=obs_dim,
            seed=seed,
            lr=lr,
            weight_decay=weight_decay,
        )
        self.sampling_type = sampling_type
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.is_active_exploration_model = is_active_exploration_model
        self.beta = beta
        self.action_cost = action_cost

        self._init_fn()

    def _init_fn(self):
        def predict_with_uncertainty(
                parameters,
                obs,
                action,
                rng,
                model_props: ModelProperties = ModelProperties(),
                *args,
                **kwargs,
        ):
            return self._predict_with_uncertainty(
                predict_fn=self.model._predict,
                parameters=parameters,
                act_dim=self.act_dim,
                obs=obs,
                action=action,
                rng=rng,
                sampling_type=self.sampling_type,
                model_props=model_props,
                pred_diff=self.pred_diff,
                beta=self.beta,
            )

        self.predict_with_uncertainty = jax.jit(predict_with_uncertainty)

        def _predict_raw(
                parameters,
                tran: Transition,
                model_props: ModelProperties = ModelProperties(),
        ):
            sampling_type = GPSamplingType(optimistic=False, mean=True)

            mean, uncertainty = self._predict_with_uncertainty(
                predict_fn=self.model._predict,
                parameters=parameters,
                act_dim=self.act_dim,
                obs=tran.obs,
                action=tran.action,
                rng=None,
                sampling_type=sampling_type,
                model_props=model_props,
                pred_diff=self.pred_diff,
                beta=self.beta,
            )

            return mean, uncertainty

        self.predict_raw = jax.jit(_predict_raw)

        def predict(
                parameters,
                obs,
                action,
                rng,
                model_props: ModelProperties = ModelProperties(),
                *args,
                **kwargs,
        ):
            return self._predict(
                predict_with_uncertainty=self.predict_with_uncertainty,
                parameters=parameters,
                obs=obs,
                action=action,
                rng=rng,
                model_props=model_props,
            )

        self.predict = jax.jit(predict)

        def predict_with_mean(
                parameters,
                obs,
                action,
                rng,
                model_props: ModelProperties = ModelProperties(),
                *args,
                **kwargs,
        ):
            sampling_type = GPSamplingType(optimistic=False, mean=True)

            mean, uncertainty = self._predict_with_uncertainty(
                predict_fn=self.model._predict,
                parameters=parameters,
                act_dim=self.act_dim,
                obs=obs,
                action=action,
                rng=rng,
                sampling_type=sampling_type,
                model_props=model_props,
                pred_diff=self.pred_diff,
                beta=self.beta,
            )
            return mean

        self.predict_with_mean = jax.jit(predict_with_mean)

        def evaluate(
                parameters,
                obs,
                action,
                rng,
                model_props: ModelProperties = ModelProperties(),
                *args,
                **kwargs,
        ):
            return self._evaluate(
                pred_fn=self.predict_with_mean,
                reward_fn=self.reward_model.predict,
                parameters=parameters,
                obs=obs,
                action=action,
                rng=rng,
                model_props=model_props,
                act_dim=self.act_dim,
            )

        self.evaluate = jax.jit(evaluate)

        if self.is_active_exploration_model:
            def evaluate_for_exploration(
                    parameters,
                    obs,
                    action,
                    rng,
                    model_props: ModelProperties = ModelProperties(),
                    *args,
                    **kwargs,
            ):
                act, _ = jnp.split(action, axis=-1, indices_or_sections=[self.act_dim])
                next_obs, reward = self._evaluate_for_active_exploration(
                    pred_with_uncertainty=self.predict_with_uncertainty,
                    act_dim=self.act_dim,
                    parameters=parameters,
                    obs=obs,
                    action=action,
                    rng=rng,
                    model_props=model_props,
                )
                action_cost = self.action_cost * jnp.sum(jnp.square(act), axis=-1)
                reward = reward - action_cost
                return next_obs, reward

            self.evaluate_for_exploration = jax.jit(evaluate_for_exploration)
        else:
            def evaluate_for_exploration(
                    parameters,
                    obs,
                    action,
                    rng,
                    model_props: ModelProperties = ModelProperties(),
                    *args,
                    **kwargs,
            ):
                return self._evaluate(
                    pred_fn=self.predict,
                    reward_fn=self.reward_model.predict,
                    parameters=parameters,
                    obs=obs,
                    action=action,
                    rng=rng,
                    model_props=model_props,
                    act_dim=self.act_dim,
                )

            self.evaluate_for_exploration = jax.jit(evaluate_for_exploration)

    @staticmethod
    def _predict_with_uncertainty(predict_fn: Callable,
                                  parameters: PyTree,
                                  act_dim: int,
                                  obs: chex.Array,
                                  action: chex.Array,
                                  rng: jax.random.PRNGKeyArray,
                                  sampling_type: GPSamplingType,
                                  model_props: ModelProperties = ModelProperties(),
                                  pred_diff: bool = 1,
                                  beta: float = 1.0,
                                  *args,
                                  **kwargs,
                                  ):
        """Predict next state and epistemic std using GP model. Optimistic, mean or DS prediction."""
        alpha = model_props.alpha
        bias_obs = model_props.bias_obs
        bias_act = model_props.bias_act
        bias_out = model_props.bias_out
        scale_obs = model_props.scale_obs
        scale_act = model_props.scale_act
        scale_out = model_props.scale_out
        act, eta = jnp.split(action, axis=-1, indices_or_sections=[act_dim])
        transformed_obs = (obs - bias_obs) / scale_obs
        transformed_act = (act - bias_act) / scale_act
        obs_action = jnp.concatenate([transformed_obs, transformed_act], axis=-1)
        next_obs_mean, next_obs_std, y_mean, y_std = predict_fn(obs_action, parameters)
        next_obs_mean = next_obs_mean.reshape(obs.shape)
        next_obs_std = next_obs_std.reshape(obs.shape)
        if sampling_type.optimistic:
            next_obs = next_obs_mean + beta * next_obs_std * eta

        elif sampling_type.mean:
            next_obs = next_obs_mean

        else:
            next_obs = sample_normal_dist(
                next_obs_mean,
                next_obs_std,
                rng,
            )

        next_obs = next_obs * scale_out + bias_out + pred_diff * obs
        return next_obs, next_obs_std * scale_out

    @staticmethod
    def _predict(
            predict_with_uncertainty: Callable,
            parameters: PyTree,
            obs: chex.Array,
            action: chex.Array,
            rng: jax.random.PRNGKeyArray,
            model_props: ModelProperties = ModelProperties(),
            *args,
            **kwargs,
    ):
        """Predict next state only using the GP model."""
        next_obs, _ = predict_with_uncertainty(
            parameters=parameters,
            obs=obs,
            action=action,
            rng=rng,
            model_props=model_props,
        )
        return next_obs

    @staticmethod
    def _evaluate(
            pred_fn: Callable,
            reward_fn: Callable,
            act_dim: int,
            parameters: PyTree,
            obs: chex.Array,
            action: chex.Array,
            rng: jax.random.PRNGKeyArray,
            model_props: ModelProperties = ModelProperties(),
            *args,
            **kwargs,
    ) -> [chex.Array, chex.Array]:
        """Predict the next state with true reward."""
        model_rng = None
        reward_rng = None
        if rng is not None:
            rng, model_rng = jax.random.split(rng, 2)
            rng, reward_rng = jax.random.split(rng, 2)
        next_obs = pred_fn(
            parameters=parameters,
            obs=obs,
            action=action,
            rng=model_rng,
            model_props=model_props,
        )
        act, _ = jnp.split(action, axis=-1, indices_or_sections=[act_dim])
        reward = reward_fn(obs, act, next_obs, reward_rng)
        return next_obs, reward

    @staticmethod
    def _evaluate_for_active_exploration(
            pred_with_uncertainty: Callable,
            parameters: PyTree,
            obs: chex.Array,
            action: chex.Array,
            rng: jax.random.PRNGKeyArray,
            model_props: ModelProperties = ModelProperties(),
            use_log_uncertainties: bool = True,
            *args,
            **kwargs
    ) -> [chex.Array, chex.Array]:
        """Predicts the next state with intrinsic reward"""
        model_rng = None
        if rng is not None:
            rng, model_rng = jax.random.split(rng, 2)
        next_obs, next_obs_std = pred_with_uncertainty(
            parameters=parameters,
            obs=obs,
            action=action,
            rng=model_rng,
            model_props=model_props,
        )
        if use_log_uncertainties:
            reward = jnp.sum(jnp.log(EPS + jnp.square(next_obs_std)), axis=-1)
        else:
            reward = jnp.sum(next_obs_std, axis=-1)

        return next_obs, reward

    @property
    def model_params(self) -> PyTree:
        return self.model.parameter_state

    @property
    def model_opt_state(self) -> PyTree:
        return jnp.empty(1)

    @property
    def init_model_params(self) -> PyTree:
        return self.model.init_parameter_state

    @property
    def init_model_opt_state(self) -> PyTree:
        return jnp.empty(1)

    def _train_step(
            self,
            tran: Transition,
            model_params=None,
            model_opt_state=None,
            val: Optional[Transition] = None,
            rng: Optional[jax.random.PRNGKey] = None,
            alpha: Optional[jax.Array] = jnp.ones(1),
            num_steps: int = 1000,
            *args,
            **kwargs,
    ) -> [PyTree, PyTree, chex.Array, ModelSummary]:
        x = jnp.concatenate([tran.obs, tran.action], axis=-1)
        y = tran.next_obs
        trained_state = self.model._train(
            x=x,
            y=y,
            num_steps=num_steps,
            key=rng,
        )

        trained_params = GPModelParameters(
            gp_parameter_state=trained_state,
            dataset=model_params.dataset,
        )

        summary = ModelSummary()
        return trained_params, jnp.empty(1), jnp.ones(self.obs_dim), summary

    def update_model(self, model_params: PyTree, model_opt_state: PyTree, alpha: chex.Array):
        super().update_model(model_params, model_opt_state, alpha)
        self.model.parameter_state = GPModelParameters(
            gp_parameter_state=model_params.gp_parameter_state,
            dataset=self.model.parameter_state.dataset
        )
        self.model.opt_state = model_opt_state

    def update_model_posterior(self, buffer: ReplayBuffer):
        tran = buffer.get_full_normalized_data()
        x = jnp.concatenate([tran.obs, tran.action], axis=-1)
        y = tran.next_obs
        self.model.update_dataset(x, y)
