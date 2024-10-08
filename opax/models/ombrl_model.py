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


class OMBRLModel(BayesianDynamicsModel):

    def __init__(self,
                 beta: float = 1.0,
                 sample_with_eps_std: bool = False,
                 *args,
                 **kwargs
                 ):

        super(OMBRLModel, self).__init__(*args, **kwargs)
        self.beta = beta
        self.sample_with_eps_std = sample_with_eps_std
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
                action=action,
                rng=rng,
                num_ensembles=self.model.num_ensembles,
                model_props=model_props,
                pred_diff=self.pred_diff,
                use_exploration=self.sample_with_eps_std,
                sampling_idx=sampling_idx,
                return_epistemic_uncertainty=True,
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
                action=action,
                rng=rng,
                num_ensembles=self.model.num_ensembles,
                model_props=model_props,
                pred_diff=self.pred_diff,
                use_exploration=False,
                sampling_idx=sampling_idx,
                return_epistemic_uncertainty=True,
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
                parameters=parameters,
                obs=obs,
                action=action,
                rng=rng,
                beta=self.beta,
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
                parameters=parameters,
                obs=obs,
                action=action,
                rng=rng,
                beta=0.0,
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
                 obs: chex.Array,
                 action: chex.Array,
                 rng: jax.random.PRNGKeyArray,
                 num_ensembles: int,
                 model_props: ModelProperties = ModelProperties(),
                 pred_diff: bool = True,
                 use_exploration: bool = True,
                 sampling_idx: Optional[int] = None,
                 return_epistemic_uncertainty: bool = False,
                 ):
        """predict with optimism."""
        alpha = model_props.alpha
        bias_obs = model_props.bias_obs
        bias_act = model_props.bias_act
        bias_out = model_props.bias_out
        scale_obs = model_props.scale_obs
        scale_act = model_props.scale_act
        scale_out = model_props.scale_out
        transformed_obs = (obs - bias_obs) / scale_obs
        transformed_act = (action - bias_act) / scale_act
        obs_action = jnp.concatenate([transformed_obs, transformed_act], axis=-1)
        next_obs_tot = predict_fn(parameters, obs_action)
        mean, std = jnp.split(next_obs_tot, 2, axis=-1)

        if rng is None:
            mean, _ = jnp.split(next_obs_tot, 2, axis=-1)
            next_obs = jnp.mean(mean, axis=0)
            next_obs_eps_std = alpha * jnp.std(mean, axis=0)

        else:
            def get_epistemic_estimate(mean, std, rng):
                next_obs_eps_std = alpha * jnp.std(mean, axis=0)
                al_uncertainty = jnp.sqrt(jnp.mean(jnp.square(std), axis=0))
                next_state_mean = jnp.mean(mean, axis=0)
                total_var = jnp.square(al_uncertainty) + jnp.square(next_obs_eps_std) * use_exploration
                next_obs = sample_normal_dist(
                    next_state_mean,
                    jnp.sqrt(total_var),
                    rng,
                )
                return next_obs, next_obs_eps_std

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
            next_obs, next_obs_eps_std = get_epistemic_estimate(mean, std, rng)
        next_obs = next_obs * scale_out + bias_out + pred_diff * obs
        if return_epistemic_uncertainty:
            return next_obs, jnp.linalg.norm(next_obs_eps_std, axis=-1)
        else:
            return next_obs

    @staticmethod
    def _evaluate(
            pred_fn: Callable,
            reward_fn: Callable,
            beta: float,
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
        next_obs, next_obs_std = pred_fn(
            parameters=parameters,
            obs=obs,
            action=action,
            rng=model_rng,
            model_props=model_props,
        )

        reward = reward_fn(obs, action, next_obs, reward_rng)
        reward = reward + beta * next_obs_std
        return next_obs, reward
