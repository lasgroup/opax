from typing import Optional, Callable, Union

import chex
import jax
import jax.numpy as jnp
from opax.utils.utils import sample_normal_dist
from opax.utils.replay_buffer import Transition
from opax.models.bayesian_dynamics_model import BayesianDynamicsModel
from opax.utils.type_aliases import ModelProperties
from jaxtyping import PyTree


@jax.jit
def sample(predictions, idx, s_rng):
    """
    :param predictions: (Ne, n_state, 2)
    :param idx: (1, )
    :return:
    """
    pred = predictions[idx]
    mu, sig = jnp.split(pred,
                        2,
                        axis=-1
                        )

    sampled_obs = sample_normal_dist(
        mu,
        sig,
        s_rng
    )
    return sampled_obs


class HUCRLModel(BayesianDynamicsModel):

    def __init__(self,
                 beta: float = 1.0,
                 *args,
                 **kwargs
                 ):

        super(HUCRLModel, self).__init__(*args, **kwargs)
        self.beta = beta
        self._init_fn()

    def _init_fn(self):

        super()._init_fn()

        def predict(parameters,
                    obs,
                    action,
                    rng,
                    model_props: ModelProperties = ModelProperties(),
                    sampling_idx: Optional[int] = None,
                    ):
            return self._predict(
                predict_fn=self.model._predict,
                parameters=parameters,
                obs=obs,
                act_dim=self.act_dim,
                action=action,
                rng=rng,
                num_ensembles=self.model.num_ensembles,
                beta=self.beta,
                model_props=model_props,
                pred_diff=self.pred_diff,
                use_optimism=True,
                sampling_idx=sampling_idx,
            )

        self.predict = jax.jit(predict)

        def predict_with_mean(parameters,
                              obs,
                              action,
                              rng,
                              model_props: ModelProperties = ModelProperties(),
                              sampling_idx: Optional[int] = None,
                              ):
            return self._predict(
                predict_fn=self.model._predict,
                parameters=parameters,
                obs=obs,
                act_dim=self.act_dim,
                action=action,
                rng=rng,
                num_ensembles=self.model.num_ensembles,
                beta=self.beta,
                model_props=model_props,
                pred_diff=self.pred_diff,
                use_optimism=False,
                sampling_idx=sampling_idx,
            )

        self.predict_with_mean = jax.jit(predict_with_mean)

        def evaluate_for_exploration(
                parameters,
                obs,
                action,
                rng,
                model_props: ModelProperties = ModelProperties(),
                sampling_idx: Optional[int] = None,
        ):
            return self._evaluate(
                pred_fn=self.predict,
                reward_fn=self.reward_model.predict,
                act_dim=self.act_dim,
                parameters=parameters,
                obs=obs,
                action=action,
                rng=rng,
                model_props=model_props,
                sampling_idx=sampling_idx,
            )

        self.evaluate_for_exploration = jax.jit(evaluate_for_exploration)

        def evaluate(
                parameters,
                obs,
                action,
                rng,
                model_props: ModelProperties = ModelProperties(),
                sampling_idx: Optional[int] = None,
        ):
            return self._evaluate(
                pred_fn=self.predict_with_mean,
                reward_fn=self.reward_model.predict,
                act_dim=self.act_dim,
                parameters=parameters,
                obs=obs,
                action=action,
                rng=rng,
                model_props=model_props,
                sampling_idx=sampling_idx,
            )

        self.evaluate = jax.jit(evaluate)

        def _train_step(
                tran: Transition,
                model_params,
                model_opt_state,
                val: Optional[Transition] = None,
                *args,
                **kwargs,
        ):
            return self._train(
                train_fn=self.model._train_step,
                predict_fn=self.model._predict,
                calibrate_fn=self.model.calculate_calibration_alpha,
                tran=tran,
                model_params=model_params,
                model_opt_state=model_opt_state,
                val=val,
            )

        self._train_step = jax.jit(_train_step)

    @staticmethod
    def _predict(predict_fn: Callable,
                 parameters: PyTree,
                 act_dim: int,
                 obs: chex.Array,
                 action: chex.Array,
                 rng: jax.random.PRNGKeyArray,
                 num_ensembles: int,
                 beta: float,
                 model_props: ModelProperties = ModelProperties(),
                 pred_diff: bool = 1,
                 use_optimism: bool = 1,
                 sampling_idx: Optional[int] = None,
                 ):
        """predict with optimism."""
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
        next_obs_tot = predict_fn(parameters, obs_action)
        mean, std = jnp.split(next_obs_tot, 2, axis=-1)

        if rng is None:
            mean, _ = jnp.split(next_obs_tot, 2, axis=-1)
            next_obs = jnp.mean(mean, axis=0)

        else:
            def get_epistemic_estimate(mean, std, eta, rng):
                next_obs_eps_std = alpha * jnp.std(mean, axis=0)
                al_uncertainty = jnp.sqrt(jnp.mean(jnp.square(std), axis=0))
                next_state_mean = jnp.mean(mean, axis=0) + beta * next_obs_eps_std * eta * use_optimism
                next_obs = sample_normal_dist(
                    next_state_mean,
                    al_uncertainty,
                    rng,
                )
                return next_obs

            # sample_rng = jax.random.split(
            #    rng,
            #    batch_size
            # )
            # next_obs = jax.vmap(get_epistemic_estimate, in_axes=(1, 1, 0, 0), out_axes=0)(
            #    mean,
            #    std,
            #    eta,
            #    sample_rng
            # )
            next_obs = get_epistemic_estimate(mean, std, eta, rng)
        next_obs = next_obs * scale_out + bias_out + pred_diff * obs
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
            sampling_idx: Optional[int] = None,
    ):
        """Predict next state with optimism and obtain the next reward."""
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
