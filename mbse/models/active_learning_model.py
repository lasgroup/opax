from typing import Optional
import jax
import jax.numpy as jnp
from mbse.utils.utils import sample_normal_dist
from mbse.models.hucrl_model import HUCRLModel
from mbse.models.bayesian_dynamics_model import BayesianDynamicsModel, sample, SamplingType
from mbse.utils.type_aliases import ModelProperties

EPS = 1e-6


def evaluate_for_exploration(
        pred_fn,
        parameters,
        obs,
        action,
        rng,
        model_props: ModelProperties = ModelProperties(),
        sampling_idx: Optional[int] = None,
        use_log_uncertainties: bool = False,
        use_al_uncertainties: bool = False,
):
    model_rng = None
    if rng is not None:
        rng, model_rng = jax.random.split(rng, 2)
    next_obs, eps_uncertainty, al_uncertainty = pred_fn(
        parameters=parameters,
        obs=obs,
        action=action,
        rng=model_rng,
        model_props=model_props,
        sampling_idx=sampling_idx,
    )
    if use_log_uncertainties:
        if use_al_uncertainties:
            frac = eps_uncertainty / (al_uncertainty + EPS)
            reward = jnp.sum(jnp.log(1 + jnp.square(frac)), axis=-1)
        else:
            reward = jnp.sum(jnp.log(EPS + jnp.square(eps_uncertainty)), axis=-1)
    else:
        if use_al_uncertainties:
            frac = eps_uncertainty / (al_uncertainty + EPS)
            reward = jnp.sum(jnp.square(frac), axis=-1)
        else:
            reward = jnp.sum(jnp.square(eps_uncertainty), axis=-1)
    return next_obs, reward


class ActiveLearningPETSModel(BayesianDynamicsModel):
    def __init__(self,
                 use_log_uncertainties=False,
                 use_al_uncertainties=False,
                 *args,
                 **kwargs
                 ):
        super(ActiveLearningPETSModel, self).__init__(*args, **kwargs)
        self.use_log_uncertainties = use_log_uncertainties
        self.use_al_uncertainties = use_al_uncertainties
        self._init_fn()

    def _init_fn(self):
        super()._init_fn()

        def predict_with_uncertainty(
                parameters,
                obs,
                action,
                rng,
                sampling_idx=self.sampling_idx,
                model_props: ModelProperties = ModelProperties(),
        ):
            return self._predict_with_uncertainty(
                predict_fn=self.model._predict,
                parameters=parameters,
                obs=obs,
                action=action,
                rng=rng,
                sampling_type=self.sampling_type,
                num_ensembles=self.model.num_ensembles,
                sampling_idx=sampling_idx,
                model_props=model_props,
                pred_diff=self.pred_diff,
            )

        self.predict_with_uncertainty = jax.jit(predict_with_uncertainty)

        def _evaluate_for_exploration(
                parameters,
                obs,
                action,
                rng,
                model_props: ModelProperties = ModelProperties(),
                sampling_idx: Optional[int] = None,
        ):
            return evaluate_for_exploration(
                pred_fn=self.predict_with_uncertainty,
                parameters=parameters,
                obs=obs,
                action=action,
                rng=rng,
                model_props=model_props,
                sampling_idx=sampling_idx,
                use_log_uncertainties=self.use_log_uncertainties,
                use_al_uncertainties=self.use_al_uncertainties,
            )

        self.evaluate_for_exploration = jax.jit(_evaluate_for_exploration)
        sampling_type = SamplingType()
        sampling_type.set_type('mean')

        def predict_with_mean(
                parameters,
                obs,
                action,
                rng,
                model_props: ModelProperties = ModelProperties(),
                sampling_idx: int = 0,
        ):
            return self._predict(
                predict_fn=self.model._predict,
                parameters=parameters,
                obs=obs,
                action=action,
                rng=rng,
                sampling_type=sampling_type,
                num_ensembles=self.model.num_ensembles,
                sampling_idx=sampling_idx,
                model_props=model_props,
                pred_diff=self.pred_diff,
            )

        self.predict_with_mean = jax.jit(predict_with_mean)

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
                model_props=model_props,
                sampling_idx=sampling_idx,
            )

        self.evaluate = jax.jit(evaluate)
        self.predict = self.predict_with_mean

    @staticmethod
    def _predict_with_uncertainty(
            predict_fn,
            parameters,
            obs,
            action,
            rng,
            sampling_type,
            num_ensembles,
            sampling_idx,
            model_props: ModelProperties = ModelProperties(),
            pred_diff: bool = 1,
    ):
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
        epistemic_uncertainty = jnp.std(mean, axis=0)
        aleatoric_uncertainty = jnp.sqrt(jnp.mean(jnp.square(std), axis=0))

        sampling_scheme = 'mean' if rng is None \
            else sampling_type.name

        if sampling_scheme == 'mean':
            mean, _ = jnp.split(next_obs_tot, 2, axis=-1)
            next_obs = jnp.mean(mean, axis=0)

        elif sampling_scheme == 'TS1':
            model_rng, sample_rng = jax.random.split(rng, 2)
            model_idx = jax.random.randint(
                model_rng,
                shape=(1,),
                minval=0,
                maxval=num_ensembles)

            model_idx = model_idx.squeeze()

            next_obs = sample(
                next_obs_tot,
                model_idx,
                sample_rng
            )
        elif sampling_scheme == 'TSInf':
            assert sampling_idx.shape[0] == 1, \
                'Set sampling indexes size to be particle size'
            next_obs = sample(
                next_obs_tot,
                sampling_idx,
                rng
            )

        elif sampling_scheme == 'DS':
            mean, std = jnp.split(next_obs_tot, 2, axis=-1)
            obs_mean = jnp.mean(mean, axis=0)
            al_var = jnp.mean(jnp.square(std), axis=0)
            ep_var = jnp.var(mean, axis=0)
            obs_var = al_var + ep_var
            obs_std = jnp.sqrt(obs_var)
            next_obs = sample_normal_dist(
                obs_mean,
                obs_std,
                rng,
            )
        next_obs = next_obs * scale_out + bias_out + pred_diff * obs
        return next_obs, epistemic_uncertainty * scale_out, aleatoric_uncertainty * scale_out


class ActiveLearningHUCRLModel(HUCRLModel):

    def __init__(self,
                 use_log_uncertainties=False,
                 use_al_uncertainties=False,
                 *args,
                 **kwargs
                 ):

        super(ActiveLearningHUCRLModel, self).__init__(*args, **kwargs)
        self.use_log_uncertainties = use_log_uncertainties
        self.use_al_uncertainties = use_al_uncertainties
        self._init_fn()

    def _init_fn(self):
        super()._init_fn()

        def predict_with_uncertainty(parameters,
                                     obs,
                                     action,
                                     rng,
                                     model_props: ModelProperties = ModelProperties(),
                                     sampling_idx: Optional[int] = None,
                                     ):
            return self._predict_with_uncertainty(
                predict_fn=self.model._predict,
                parameters=parameters,
                obs=obs,
                act_dim=self.act_dim,
                action=action,
                rng=rng,
                beta=self.beta,
                model_props=model_props,
                pred_diff=self.pred_diff,
                sampling_idx=sampling_idx,
            )

        self.predict_with_uncertainty = jax.jit(predict_with_uncertainty)

        def _evaluate_for_exploration(
                parameters,
                obs,
                action,
                rng,
                model_props: ModelProperties = ModelProperties(),
                sampling_idx: Optional[int] = None,
        ):
            return evaluate_for_exploration(
                pred_fn=self.predict_with_uncertainty,
                parameters=parameters,
                obs=obs,
                action=action,
                rng=rng,
                model_props=model_props,
                sampling_idx=sampling_idx,
                use_log_uncertainties=self.use_log_uncertainties,
                use_al_uncertainties=self.use_al_uncertainties,
            )

        self.evaluate_for_exploration = jax.jit(_evaluate_for_exploration)

        def predict_without_optimism(parameters,
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

        self.predict_without_optimism = jax.jit(predict_without_optimism)
        self.predict = self.predict_without_optimism

        def evaluate(
                parameters,
                obs,
                action,
                rng,
                model_props: ModelProperties = ModelProperties(),
                sampling_idx: Optional[int] = None,
        ):
            return self._evaluate(
                pred_fn=self.predict_without_optimism,
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

    @staticmethod
    def _predict_with_uncertainty(
            predict_fn,
            parameters,
            act_dim: int,
            obs,
            action,
            rng,
            beta,
            model_props: ModelProperties = ModelProperties(),
            pred_diff: bool = 1,
            sampling_idx: Optional[int] = None,
    ):
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
            next_obs = jnp.mean(mean, axis=0)
            next_obs_eps_std = jnp.std(mean, axis=0)
            al_uncertainty = jnp.sqrt(jnp.mean(jnp.square(std), axis=0))

        else:
            def get_epistemic_estimate(mean, std, eta, rng):
                next_obs_eps_std = jnp.std(mean, axis=0)
                al_uncertainty = jnp.sqrt(jnp.mean(jnp.square(std), axis=0))
                next_state_mean = jnp.mean(mean, axis=0) + beta * alpha * next_obs_eps_std * eta
                next_obs = sample_normal_dist(
                    next_state_mean,
                    al_uncertainty,
                    rng,
                )
                return next_obs, next_obs_eps_std, al_uncertainty

            next_obs, next_obs_eps_std, al_uncertainty = get_epistemic_estimate(mean, std, eta, rng)
        next_obs = next_obs * scale_out + bias_out + pred_diff * obs
        return next_obs, next_obs_eps_std * scale_out, al_uncertainty * scale_out
