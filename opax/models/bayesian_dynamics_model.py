from typing import Sequence, Callable, Optional

import chex
import numpy as np
import gym
from flax import linen as nn
from opax.utils.models import FSVGDEnsemble, ProbabilisticEnsembleModel
import jax
import jax.numpy as jnp
from opax.utils.utils import sample_normal_dist
from opax.utils.replay_buffer import Transition
from opax.models.dynamics_model import DynamicsModel, ModelSummary
from opax.utils.type_aliases import ModelProperties
from opax.models.reward_model import RewardModel
from typing import List, Union
from opax.utils.utils import gaussian_log_likelihood
from opax.utils.network_utils import mse
from jaxtyping import PyTree

class SamplingType:
    name: str = 'TS1'
    name_types: List[str] = \
        ['All', 'DS', 'TS1', 'TSInf', 'mean']

    def set_type(self, name):
        assert name in self.name_types, \
            'name must be in ' + ' '.join(map(str, self.name_types))

        self.name = name


@jax.jit
def sample(predictions: chex.Array, idx: chex.Array, s_rng: jax.random.PRNGKeyArray):
    """
    :param predictions: (num ensembles, batch size, 2 x dim states)
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


class BayesianDynamicsModel(DynamicsModel):

    def __init__(self,
                 action_space: gym.spaces.box,
                 observation_space: gym.spaces.box,
                 reward_model: RewardModel,
                 model_class: str = "ProbabilisticEnsembleModel",
                 num_ensemble: int = 10,
                 features: Sequence[int] = (256, 256),
                 non_linearity: Callable = nn.swish,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 sig_min: float = 1e-3,
                 sig_max: float = 1e3,
                 deterministic: bool = False,
                 seed: int = 0,
                 *args,
                 **kwargs
                 ):
        """

        :param action_space: gym.spaces.box, action space of the env
        :param observation_space: gym.spaces.box, observation space of the env
        :param reward_model: RewardModel
        :param model_class: str, indicates the type of model being used FSVGD vs ensembles
        :param num_ensemble: int, number of ensembles
        :param features: Sequence[int], number of features in the neural network
        :param non_linearity: Callable, nonlinear activation function
        :param lr: float, learning rate for the model training
        :param weight_decay: float, weight decay for the model training
        :param sig_min: float, minimal aleatoric std
        :param sig_max: float, maximal aleatoric std
        :param deterministic: bool, boolean to indicate if the model is deterministic or probabilistic
        :param seed: int, seed for model initialization
        :param args:
        :param kwargs:
        """

        super(BayesianDynamicsModel, self).__init__(*args, **kwargs)
        self.reward_model = reward_model
        if model_class == "ProbabilisticEnsembleModel":
            model_cls = ProbabilisticEnsembleModel
        elif model_class == "fSVGDEnsemble":
            model_cls = FSVGDEnsemble
        else:
            assert False, "Model class must be ProbabilisticEnsembleModel or fSVGDEnsemble."

        obs_dim = np.prod(observation_space.shape)
        act_dim = np.prod(action_space.shape)
        sample_obs = observation_space.sample()
        sample_act = action_space.sample()
        obs_action = jnp.concatenate([sample_obs, sample_act], axis=-1)
        self.model = model_cls(
            example_input=obs_action,
            num_ensemble=num_ensemble,
            features=features,
            output_dim=obs_dim,
            non_linearity=non_linearity,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed,
            sig_min=sig_min,
            sig_max=sig_max,
            deterministic=deterministic,
        )
        self.sampling_type = SamplingType()
        self.sampling_idx = jnp.zeros(1)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._init_fn()
        self.evaluate_for_exploration = self.evaluate

    def _init_fn(self):
        """Defines basic functions for model and jits them."""
        def predict(parameters,
                    obs,
                    action,
                    rng,
                    sampling_idx=self.sampling_idx,
                    model_props: ModelProperties = ModelProperties(),
                    ):
            return self._predict(
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

        self.predict = jax.jit(predict)

        def evaluate(
                parameters,
                obs,
                action,
                rng,
                sampling_idx=self.sampling_idx,
                model_props: ModelProperties = ModelProperties(),
        ):
            return self._evaluate(
                pred_fn=self.predict,
                reward_fn=self.reward_model.predict,
                parameters=parameters,
                obs=obs,
                action=action,
                rng=rng,
                sampling_idx=sampling_idx,
                model_props=model_props,
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

        def _predict_raw(
                parameters,
                tran: Transition,
                model_props: ModelProperties = ModelProperties(),
        ):
            return self._predict_raw(
                predict_fn=self.model._predict,
                parameters=parameters,
                tran=tran,
                model_props=model_props,
                pred_diff=self.pred_diff,
            )

        self.predict_raw = jax.jit(_predict_raw)

    def set_sampling_type(self, name):
        self.sampling_type.set_type(name)

    def set_sampling_idx(self, idx):
        self.sampling_idx = jnp.clip(
            idx,
            0,
            self.model.num_ensembles
        )

    @staticmethod
    def _predict_raw(
            predict_fn: Callable,
            parameters: PyTree,
            tran: Transition,
            model_props: ModelProperties = ModelProperties(),
            pred_diff: bool = 1,
    ) -> [chex.Array, chex.Array]:
        """
        :param predict_fn: Callable, forward pass through the dynamics model
        :param parameters: PyTree, model parameters
        :param tran: Transitions, transitions for which the state is predicted
        :param model_props: ModelProperties
        :param pred_diff: bool, whether the model predicts diff or next state
        :return
        :output next_obs_mean: chex.Array -> mean prediction of each ensemble member
        :output next_obs_std: chex.Array -> std prediction of each ensemble member
        """
        alpha = model_props.alpha
        bias_obs = model_props.bias_obs
        bias_act = model_props.bias_act
        bias_out = model_props.bias_out
        scale_obs = model_props.scale_obs
        scale_act = model_props.scale_act
        scale_out = model_props.scale_out
        obs = tran.obs
        action = tran.action
        # normalize obs and action
        transformed_obs = (obs - bias_obs) / scale_obs
        transformed_act = (action - bias_act) / scale_act
        # get next mean and uncertainty prediction from the model
        obs_action = jnp.concatenate([transformed_obs, transformed_act], axis=-1)
        next_obs_tot = predict_fn(parameters, obs_action)
        mean, std = jnp.split(next_obs_tot, 2, axis=-1)
        # unnormalize predictions
        next_obs_mean = mean * scale_out + bias_out + pred_diff * obs
        next_obs_std = std * scale_out
        return next_obs_mean, next_obs_std

    @staticmethod
    def _predict(predict_fn: Callable,
                 parameters: PyTree,
                 obs: chex.Array,
                 action: chex.Array,
                 rng: jax.random.PRNGKeyArray,
                 sampling_type: SamplingType,
                 num_ensembles: int,
                 sampling_idx: chex.Array,
                 model_props: ModelProperties = ModelProperties(),
                 pred_diff: bool = True,
                 ):
        """
        :param predict_fn: Callable, prediction function
        :param parameters: PyTree, model parameters
        :param obs: chex.Array
        :param action: chex.Array
        :param rng: jax.random.PRNGKeyArray
        :param sampling_type: SamplingType
        :param num_ensembles: int
        :param sampling_idx: chex.Array
        :param model_props: ModelProperties
        :param pred_diff: bool
        :return:
        """

        # normalize and predict the next state distribution with each ensemble
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
        next_obs = next_obs_tot

        # predict with the ensembles using either mean, TS1, TSInf or DS
        sampling_scheme = 'mean' if rng is None \
            else sampling_type.name

        if sampling_scheme == 'mean':
            mean, al_uncertainty = jnp.split(next_obs_tot, 2, axis=-1)
            mean = jnp.mean(mean, axis=0)
            al_uncertainty = jnp.sqrt(jnp.mean(jnp.square(al_uncertainty), axis=0))
            if rng is not None:
                next_obs = sample_normal_dist(
                    mean,
                    al_uncertainty,
                    rng,
                )
            else:
                next_obs = mean

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
        return next_obs

    @staticmethod
    def _train(train_fn: Callable, predict_fn: Callable, calibrate_fn: Callable, tran: Transition,
               model_params: PyTree, model_opt_state: PyTree, val: Optional[Transition] = None) -> \
            [PyTree, PyTree, chex.Array, ModelSummary]:
        """Train dynamics model and validate"""
        alpha = jnp.ones(tran.obs.shape[-1])
        best_score = jnp.zeros(1)
        x = jnp.concatenate([tran.obs, tran.action], axis=-1)
        new_model_params, new_model_opt_state, likelihood, grad_norm = train_fn(
            params=model_params,
            opt_state=model_opt_state,
            x=x,
            y=tran.next_obs,
        )
        val_logl = jnp.zeros_like(likelihood)
        val_mse = jnp.zeros_like(likelihood)
        val_al_std = jnp.zeros_like(likelihood)
        val_ep_std = jnp.zeros_like(likelihood)
        if val is not None:
            val_x = jnp.concatenate([val.obs, val.action], axis=-1)
            val_y = val.next_obs
            y_pred = predict_fn(new_model_params, val_x)
            val_likelihood = jax.vmap(
                gaussian_log_likelihood,
                in_axes=(None, 0, 0),
                out_axes=0
            )
            mean, std = jnp.split(y_pred, 2, axis=-1)
            val_al_std = std.mean()
            eps_std = jnp.std(mean, axis=0)
            val_ep_std = eps_std.mean()
            logl = val_likelihood(val_y, mean, std)
            val_logl = logl.mean()
            val_mse = jax.vmap(
                lambda pred: mse(val_y, pred),
            )(mean)
            val_mse = val_mse.mean()
            alpha, best_score = calibrate_fn(new_model_params, val_x, val_y)

        summary = ModelSummary(
            model_likelihood=likelihood.astype(float),
            grad_norm=grad_norm.astype(float),
            val_logl=val_logl.astype(float),
            val_mse=val_mse.astype(float),
            val_al_std=val_al_std.astype(float),
            val_eps_std=val_ep_std.astype(float),
            calibration_alpha=alpha.mean().astype(float),
            calibration_error=best_score.mean().astype(float),
        )

        return new_model_params, new_model_opt_state, alpha, summary

    @property
    def model_params(self) -> PyTree:
        return self.model.particles

    @property
    def model_opt_state(self) -> PyTree:
        return self.model.opt_state

    @property
    def init_model_params(self) -> PyTree:
        return self.model.init_particles

    @property
    def init_model_opt_state(self) -> PyTree:
        return self.model.init_opt_state

    def update_model(self, model_params: PyTree, model_opt_state: PyTree, alpha: Union[chex.Array, float]):
        super().update_model(model_params, model_opt_state, alpha)
        self.model.particles = model_params
        self.model.opt_state = model_opt_state

    @staticmethod
    def _evaluate(
            pred_fn: Callable,
            reward_fn: Callable,
            parameters: PyTree,
            obs: chex.Array,
            action: chex.Array,
            rng: jax.random.PRNGKeyArray,
            sampling_idx: chex.Array,
            model_props: ModelProperties = ModelProperties()
    ) -> [chex.Array, chex.Array]:
        """Predict the next state and reward."""
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
            sampling_idx=sampling_idx,
            model_props=model_props,
        )
        reward = reward_fn(obs, action, next_obs, reward_rng)
        return next_obs, reward
