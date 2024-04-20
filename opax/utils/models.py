import functools
from typing import Sequence, Callable, Union, NamedTuple, Dict

import jax.numpy as jnp
from flax import linen as nn
from jax import jit, vmap, value_and_grad
from opax.utils.network_utils import MLP
import jax
from opax.utils.utils import gaussian_log_likelihood, rbf_kernel
import optax
from jax.scipy.stats import norm
from jaxutils import Dataset
from jax.tree_util import register_pytree_node_class
from jax import random
from jax.scipy.stats import multivariate_normal
from functools import partial

EPS = 1e-6


def _calibration_errors(calibration_score_fn, params, xs, ys, ps, alpha, output_dim) -> jax.Array:
    ps_hat = calibration_score_fn(params, xs, ys, ps, alpha)
    ps = jnp.repeat(ps[..., jnp.newaxis], repeats=output_dim, axis=1)
    return jnp.mean((ps - ps_hat) ** 2, axis=0)


def _calculate_calibration_alpha(calibration_error_fn, params, xs, ys, num_ps, output_dim) -> jax.Array:
    # We flip so that we rather take more uncertainty model than less
    ps = jnp.linspace(0, 1, num_ps + 1)[1:]
    test_alpha = jnp.flip(jnp.linspace(0, 10, 100)[1:])
    test_alphas = jnp.repeat(test_alpha[..., jnp.newaxis], repeats=output_dim, axis=1)
    errors = vmap(calibration_error_fn, in_axes=(None, None, None, None, 0))(
        params, xs, ys, ps, test_alphas)
    indices = jnp.argmin(errors, axis=0)
    best_alpha = test_alpha[indices]
    assert best_alpha.shape == (output_dim,)
    return best_alpha, jnp.diag(errors[indices])


def _predict(apply_fn, params, x, sig_max, sig_min, rng=None, deterministic=False):
    forward = jax.vmap(apply_fn, (0, None))
    predictions = forward(params, x)
    mu, sig = jnp.split(predictions, 2, axis=-1)
    sig = nn.softplus(sig)
    sig = jnp.clip(sig, 0, sig_max) + sig_min
    eps = jnp.ones_like(sig) * sig_min
    sig = (1 - deterministic) * sig + deterministic * eps
    predictions = jnp.concatenate([mu, sig], axis=-1)
    return predictions


class ProbabilisticEnsembleModel(object):
    def __init__(
            self,
            example_input: jnp.ndarray,
            num_ensemble: int = 10,
            features: Sequence[int] = [256, 256],
            output_dim: int = 1,
            non_linearity: Callable = nn.swish,
            lr: float = 1e-3,
            weight_decay: float = 1e-4,
            seed: int = 0,
            sig_min: float = 1e-3,
            sig_max: float = 1e3,
            deterministic: bool = False,
            initialize_train_fn: bool = True,
    ):
        self.output_dim = output_dim
        self.mlp = MLP(
            features=features,
            output_dim=2 * output_dim,
            non_linearity=non_linearity
        )
        self.num_ensembles = num_ensemble
        # vmap init function with respect to seed sequence
        init = vmap(self.mlp.init, (0, None))
        self.net = self.mlp.apply
        self.rng = jax.random.PRNGKey(seed)
        seed_sequence = jax.random.split(self.rng, self.num_ensembles + 1)
        self.rng = seed_sequence[0]
        seed_sequence = seed_sequence[1:]
        particles = init(seed_sequence, example_input)
        self.sig_min = sig_min
        self.sig_max = sig_max
        self.optimizer = optax.adamw(learning_rate=lr,
                                     weight_decay=weight_decay)
        optimizer_state = self.optimizer.init(particles)
        self.particles = particles
        self.opt_state = optimizer_state
        self.init_particles = particles
        self.init_opt_state = optimizer_state
        self.example_input = example_input
        self._predict = jit(lambda params, x:
                            _predict(
                                apply_fn=self.net,
                                params=params,
                                x=x,
                                sig_max=self.sig_max,
                                sig_min=self.sig_min,
                                deterministic=deterministic,
                            )
                            )
        self.num_ps = 10
        if initialize_train_fn:
            self._train_step = jit(lambda params, opt_state, x, y: self._train_fn(
                predict_fn=self._predict,
                update_fn=self.optimizer.update,
                params=params,
                opt_state=opt_state,
                x=x,
                y=y,
            ))

        def calculate_calibration_score(params, xs, ys, ps, alpha):
            return self._calculate_calibration_score(
                predict_fn=self._predict,
                params=params,
                xs=xs,
                ys=ys,
                ps=ps,
                alpha=alpha,
                output_dim=self.output_dim,
            )

        self.calculate_calibration_score_fn = calculate_calibration_score

        def calibration_errors(params, xs, ys, ps, alpha):
            return _calibration_errors(
                calibration_score_fn=self.calculate_calibration_score_fn,
                params=params,
                xs=xs,
                ys=ys,
                ps=ps,
                alpha=alpha,
                output_dim=self.output_dim,
            )

        self.calibration_errors = calibration_errors

        def calculate_calibration_alpha(params, xs, ys):
            return _calculate_calibration_alpha(
                calibration_error_fn=self.calibration_errors,
                params=params,
                xs=xs,
                ys=ys,
                num_ps=self.num_ps,
                output_dim=self.output_dim,
            )

        self.calculate_calibration_alpha = jax.jit(calculate_calibration_alpha)

    @property
    def params(self):
        return self.particles

    def predict(self, x):
        return self._predict(self.particles, x)

    @staticmethod
    def _train_fn(predict_fn, update_fn, params, opt_state, x, y, prior_particles=None):
        likelihood = jax.vmap(gaussian_log_likelihood, in_axes=(None, 0, 0), out_axes=0)

        def likelihood_loss(model_params):
            predictions = predict_fn(model_params, x)
            mu, sig = jnp.split(predictions, 2, axis=-1)
            logl = likelihood(y, mu, sig)
            return -logl.mean()

        # vmap over ensemble
        loss, grads = value_and_grad(likelihood_loss)(params)
        updates, new_opt_state = update_fn(grads,
                                           opt_state,
                                           params=params)
        new_params = optax.apply_updates(params, updates)
        grad_norm = optax.global_norm(grads)
        return new_params, new_opt_state, loss, grad_norm

    def train_step(self, x, y):
        new_params, new_opt_state, loss, grad_norm = self._train_step(
            params=self.particles,
            opt_state=self.opt_state,
            x=x,
            y=y
        )
        self.particles = new_params
        self.opt_state = new_opt_state
        return loss, grad_norm

    @staticmethod
    def _calculate_calibration_score(predict_fn, params, xs, ys, ps, alpha, output_dim):
        assert alpha.shape == (output_dim,)

        def calculate_score(x, y):
            predictions = predict_fn(params, x)
            mean, std = jnp.split(predictions, 2, axis=-1)
            mu = jnp.mean(mean, axis=0)
            eps_std = jnp.std(mean, axis=0)
            al_uncertainty = jnp.sqrt(jnp.mean(jnp.square(std), axis=0))
            cdfs = vmap(norm.cdf)(y, mu, eps_std * alpha + al_uncertainty)

            def check_cdf(cdf):
                assert cdf.shape == ()
                return cdf <= ps

            return vmap(check_cdf, out_axes=1)(cdfs)

        cdfs = vmap(calculate_score)(xs, ys)
        return jnp.mean(cdfs, axis=0)


class FSVGDEnsemble(ProbabilisticEnsembleModel):
    def __init__(self,
                 n_prior_particles: Union[int, None] = None,
                 prior_bandwidth: float = None,
                 k_bandwidth: float = 0.1,
                 initialize_train_fn: bool = True,
                 *args, **kwargs):
        super(FSVGDEnsemble, self).__init__(*args, **kwargs, initialize_train_fn=False)
        n_prior_particles = n_prior_particles or self.num_ensembles
        init = vmap(self.mlp.init, (0, None))
        seed_sequence = jax.random.split(self.rng, n_prior_particles + 1)
        self.rng = seed_sequence[0]
        seed_sequence = seed_sequence[1:]
        self.priors = init(seed_sequence, self.example_input)
        self.prior_bandwidth = prior_bandwidth
        self.k_bandwidth = k_bandwidth

        if initialize_train_fn:
            def train_step(
                    params,
                    opt_state,
                    x,
                    y,
                    prior_particles,
                    rng,
            ):
                return self._train_fn(predict_fn=self._predict,
                                      update_fn=self.optimizer.update,
                                      params=params,
                                      opt_state=opt_state,
                                      x=x,
                                      y=y,
                                      prior_particles=prior_particles,
                                      rng=rng,
                                      prior_bandwidth=self.prior_bandwidth,
                                      k_bandwidth=self.k_bandwidth,
                                      )

            self._train_step = jit(train_step)

    def _prior(self, prior_particles, x):
        predictions = self._predict(prior_particles, x)
        altered_predictions = predictions
        altered_predictions = altered_predictions.at[..., self.output_dim:].set(
            jnp.log(altered_predictions[..., self.output_dim:] + EPS))
        var = jax.vmap(lambda x: jnp.cov(x, rowvar=False),
                       in_axes=-1,
                       out_axes=-1)(altered_predictions)
        mean = jnp.mean(altered_predictions, axis=0)
        return mean, var

    @staticmethod
    def _train_fn(
            predict_fn,
            update_fn,
            params,
            opt_state,
            x,
            y,
            prior_particles,
            rng,
            prior_bandwidth,
            k_bandwidth
    ):
        # mean_prior, k_prior = self._prior(prior_particles, x)
        rbf = lambda z, v: rbf_kernel(z, v, bandwidth=prior_bandwidth)
        kernel = lambda x: rbf(x, x)  # K(x, x)
        k_prior = kernel(x)
        k_prior = jnp.stack([k_prior, k_prior], axis=-1)

        k_rbf = lambda z, v: rbf_kernel(z, v, bandwidth=k_bandwidth)

        def fsvgdloss(model_params):
            predictions, pred_vjp = jax.vjp(lambda p: predict_fn(p, x), model_params)
            k_pred, k_pred_vjp = jax.vjp(
                lambda x: k_rbf(x, predictions), predictions)
            grad_k = k_pred_vjp(-jnp.ones(k_pred.shape))[0]

            def neg_log_post(predictions):
                mean_pred, std_pred = jnp.split(predictions, 2, axis=-1)
                log_post = gaussian_log_likelihood(y, mean_pred, std_pred)
                return -log_post.mean()

            likelihood = lambda x, cov_x: \
                jax.scipy.stats.multivariate_normal.logpdf(x,
                                                           mean=jnp.zeros(x.shape[0]),
                                                           cov=cov_x + 1e-4 * jnp.eye(x.shape[0]))
            likelihood = jax.vmap(likelihood, in_axes=-1, out_axes=-1)

            def neg_log_prior(predictions):
                mean_pred, std_pred = jnp.split(predictions, 2, axis=-1)
                log_sigma = jnp.log(std_pred + EPS)
                altered_predictions = jnp.stack([mean_pred, log_sigma], axis=-1)
                log_prior = likelihood(altered_predictions, k_prior)
                return -log_prior.mean() / mean_pred.shape[-2]

            def neg_total_likelihood(predictions):
                log_post = neg_log_post(predictions)
                log_pior = neg_log_prior(predictions)
                return log_post + log_pior

            log_post, log_posterior_grad = jax.vmap(value_and_grad(neg_total_likelihood, 0))(predictions)
            stein_grad = (jnp.einsum('ij,jkm', k_pred, log_posterior_grad)
                          + grad_k)
            grad = pred_vjp(stein_grad)[0]
            return log_post.mean(), grad

        loss, grads = fsvgdloss(params)
        updates, new_opt_state = update_fn(grads,
                                           opt_state,
                                           params=params)
        new_params = optax.apply_updates(params, updates)
        grad_norm = optax.global_norm(grads)
        return new_params, new_opt_state, loss, grad_norm

    def train_step(self, x, y):
        self.rng, train_rng = jax.random.split(self.rng)
        new_params, new_opt_state, log_post, grad_norm = self._train_step(
            params=self.particles,
            opt_state=self.opt_state,
            x=x,
            y=y,
            prior_particles=self.priors,
            rng=train_rng
        )
        self.particles = new_params
        self.opt_state = new_opt_state
        return log_post, grad_norm


class KDEfWGDEnsemble(FSVGDEnsemble):
    def __init__(self, *args, **kwargs):
        super(KDEfWGDEnsemble, self).__init__(initialize_train_fn=True,
                                              *args,
                                              **kwargs)

    @staticmethod
    def _train_fn(
            predict_fn,
            update_fn,
            params,
            opt_state,
            x,
            y,
            prior_particles,
            rng,
            prior_bandwidth,
            k_bandwidth
    ):
        # mean_prior, k_prior = self._prior(prior_particles, x)
        rbf = lambda z, v: rbf_kernel(z, v, bandwidth=prior_bandwidth)
        kernel = lambda x: rbf(x, x)  # K(x, x)
        k_prior = kernel(x)
        k_prior = jnp.stack([k_prior, k_prior], axis=-1)

        k_rbf = lambda z, v: rbf_kernel(z, v, bandwidth=k_bandwidth)

        def kdeloss(model_params):
            predictions, pred_vjp = jax.vjp(lambda p: predict_fn(p, x), model_params)
            k_pred, k_pred_vjp = jax.vjp(
                lambda x: k_rbf(x, predictions), predictions)
            grad_k = k_pred_vjp(-jnp.ones(k_pred.shape))[0]

            def neg_log_post(predictions):
                mean_pred, std_pred = jnp.split(predictions, 2, axis=-1)
                log_post = gaussian_log_likelihood(y, mean_pred, std_pred)
                return -log_post.mean()

            likelihood = lambda x, cov_x: \
                jax.scipy.stats.multivariate_normal.logpdf(x,
                                                           mean=jnp.zeros(x.shape[0]),
                                                           cov=cov_x + 1e-4 * jnp.eye(x.shape[0]))
            likelihood = jax.vmap(likelihood, in_axes=-1, out_axes=-1)

            def neg_log_prior(predictions):
                mean_pred, std_pred = jnp.split(predictions, 2, axis=-1)
                log_sigma = jnp.log(std_pred + EPS)
                altered_predictions = jnp.stack([mean_pred, log_sigma], axis=-1)
                log_prior = likelihood(altered_predictions, k_prior)
                return -log_prior.mean() / mean_pred.shape[-2]

            def neg_total_likelihood(predictions):
                log_post = neg_log_post(predictions)
                log_pior = neg_log_prior(predictions)
                return log_post + log_pior

            log_post, log_posterior_grad = jax.vmap(value_and_grad(neg_total_likelihood, 0))(predictions)
            k_i = jnp.sum(k_pred, axis=1)
            stein_grad = log_posterior_grad + jax.vmap(lambda x, y: x / y)(grad_k, k_i)
            # stein_grad = (jnp.einsum('ij,jkm', k_pred, log_posterior_grad)
            #              + grad_k)
            grad = pred_vjp(stein_grad)[0]
            return log_post.mean(), grad

        loss, grads = kdeloss(params)
        updates, new_opt_state = update_fn(grads,
                                           opt_state,
                                           params=params)
        new_params = optax.apply_updates(params, updates)
        grad_norm = optax.global_norm(grads)
        return new_params, new_opt_state, loss, grad_norm


@jit
def make_positive(x):
    return jnp.logaddexp(x, 0)


class GPModelParameters(NamedTuple):
    gp_parameter_state: Dict
    dataset: Dataset


@register_pytree_node_class
class GPModel(object):
    def __init__(self,
                 input_dim: int = 1,
                 output_dim: int = 1,
                 seed: int = 0,
                 lr: float = 0.1,
                 weight_decay: float = 0.0,
                 num_train_steps: int = 1000,
                 min_std: float = 1e-3,
                 max_std: float = 1e-3,
                 ):
        self.input_dim = input_dim

        self.rng = jax.random.PRNGKey(seed)
        key, self.rng = jax.random.split(self.rng, 2)
        self.output_dim = output_dim

        self.v_kernel = vmap(self.kernel, in_axes=(0, None, None), out_axes=0)
        self.m_kernel = vmap(self.v_kernel, in_axes=(None, 0, None), out_axes=1)
        self.m_kernel_multiple_output = vmap(self.m_kernel, in_axes=(None, None, 0), out_axes=0)
        self.v_kernel_multiple_output = vmap(self.v_kernel, in_axes=(None, None, 0), out_axes=0)
        self.kernel_multiple_output = vmap(self.kernel, in_axes=(None, None, 0), out_axes=0)

        dataset = Dataset(X=jnp.empty((1,) + (input_dim,)), y=jnp.empty((1,) + (output_dim,)))
        self.min_std = min_std
        self.max_std = max_std
        params = self.initialize_parameters(key)
        self.parameter_state = GPModelParameters(
            gp_parameter_state=params,
            dataset=dataset,
        )
        self.init_parameter_state = self.parameter_state
        self.lr = lr
        self.weight_decay = weight_decay
        self.tx = optax.adamw(learning_rate=self.lr, weight_decay=self.weight_decay)
        self.num_train_steps = num_train_steps
        self._init_fn()

    def initialize_parameters(self, key):
        cat_input = jnp.ones(shape=(self.input_dim,))
        parameters = dict()
        # Inout dimension is state_dim + action_dim, we have one lengthscale for each dimension
        # Ouput dimension is state_dim, because we have one GP per state dimension
        # So we have lengthscales for each state dimension
        parameters["lengthscale"] = random.normal(key=key, shape=(self.output_dim, cat_input.size))
        parameters['noise_pseudostd'] = random.normal(key=key, shape=(self.output_dim,)) * self.min_std
        return parameters

    def kernel(self, x, y, params):
        assert x.ndim == y.ndim == 1
        assert params["lengthscale"].shape == x.shape
        return jnp.exp(- jnp.sum((x - y) ** 2 / make_positive(params["lengthscale"]) ** 2))

    def tree_flatten(self):
        children = self.parameter_state
        aux = (self.input_dim, self.output_dim, self.lr, self.weight_decay)
        return (children, aux)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        new_class = cls(*aux_data)
        parameter_state = children
        new_class.parameter_state = parameter_state
        return new_class

    def _init_fn(self):
        self.num_ps = 10

        def calculate_calibration_score(params, xs, ys, ps, alpha):
            return self._calculate_calibration_score(
                predict_fn=self._predict,
                params=params,
                xs=xs,
                ys=ys,
                ps=ps,
                alpha=alpha,
                output_dim=self.output_dim,
            )

        self.calculate_calibration_score_fn = calculate_calibration_score

        def calibration_errors(params, xs, ys, ps, alpha):
            return _calibration_errors(
                calibration_score_fn=self.calculate_calibration_score_fn,
                params=params,
                xs=xs,
                ys=ys,
                ps=ps,
                alpha=alpha,
                output_dim=self.output_dim,
            )

        self.calibration_errors = calibration_errors

        def calculate_calibration_alpha(params, xs, ys):
            return _calculate_calibration_alpha(
                calibration_error_fn=self.calibration_errors,
                params=params,
                xs=xs,
                ys=ys,
                num_ps=self.num_ps,
                output_dim=self.output_dim,
            )

        self.calculate_calibration_alpha = jax.jit(calculate_calibration_alpha)

    def _predict(self, xs, parameter_state: GPModelParameters):
        return vmap(self._predict_single, in_axes=(0, None))(jnp.atleast_2d(xs), parameter_state)

    def _predict_single(self, x, parameter_state: GPModelParameters):
        assert x.ndim == 1
        output = parameter_state.dataset.y
        inputs = parameter_state.dataset.X

        # inputs = jnp.concatenate([expanded_xs, us], axis=1)
        covariance_matrix = self.m_kernel_multiple_output(inputs, inputs, parameter_state.gp_parameter_state)
        # Todo: take care of the noise term
        num_points = covariance_matrix.shape[1]
        noise_term = jnp.repeat(jnp.eye(num_points)[None, ...], repeats=self.output_dim, axis=0)
        noise_std = jnp.clip(make_positive(parameter_state.gp_parameter_state['noise_pseudostd'][:, None, None]),
                             self.min_std, self.max_std)
        noise_term = noise_std * noise_term
        noisy_covariance_matrix = covariance_matrix + noise_term

        k_x_X = vmap(self.v_kernel, in_axes=(None, None, 0), out_axes=0)(inputs, x,
                                                                         parameter_state.gp_parameter_state)
        cholesky_tuples = vmap(jax.scipy.linalg.cho_factor)(noisy_covariance_matrix)

        # Compute std
        denoised_var = vmap(jax.scipy.linalg.cho_solve, in_axes=((0, None), 0))((cholesky_tuples[0], False), k_x_X)
        var = vmap(self.kernel, in_axes=(None, None, 0))(x, x, parameter_state.gp_parameter_state) - vmap(jnp.dot)(
            k_x_X, denoised_var)
        std = jnp.sqrt(var)

        # Compute mean
        denoised_mean = vmap(jax.scipy.linalg.cho_solve, in_axes=((0, None), 1))((cholesky_tuples[0], False), output)
        mean = vmap(jnp.dot)(k_x_X, denoised_mean)

        return mean, std, mean, std + jnp.clip(make_positive(parameter_state.gp_parameter_state['noise_pseudostd']),
                                                             self.min_std, self.max_std)

    def loss(self, params, data: Dataset, key=None):
        assert data.X.ndim == 2 and data.y.ndim == 2
        assert data.X.shape[0] == data.y.shape[0] and data.X.shape[1] == self.input_dim and data.y.shape[
            1] == self.output_dim
        num_points = data.X.shape[0]

        inputs = data.X

        covariance_matrix = self.m_kernel_multiple_output(inputs, inputs, params)
        # Todo: here we need to add noise term
        noise_term = jnp.repeat(jnp.eye(num_points)[None, ...], repeats=self.output_dim, axis=0)
        noise_std = jnp.clip(make_positive(params['noise_pseudostd'][:, None, None]),
                             self.min_std, self.max_std)
        noise_term = noise_std * noise_term
        noisy_covariance_matrix = covariance_matrix + noise_term

        log_pdf = vmap(multivariate_normal.logpdf, in_axes=(1, None, 0))(data.y, jnp.zeros(num_points, ),
                                                                         noisy_covariance_matrix)
        return - jnp.sum(log_pdf)

    @partial(jit, static_argnums=0)
    def _step_jit(self, opt_state: optax.OptState, param_vec_stack: jnp.array, data: Dataset,
                  key: jax.random.PRNGKey, ):
        loss, grads = jax.value_and_grad(self.loss)(param_vec_stack, data, key)
        updates, opt_state = self.tx.update(grads, opt_state, param_vec_stack)
        param_vec_stack = optax.apply_updates(param_vec_stack, updates)
        return opt_state, param_vec_stack, loss

    @functools.partial(jax.jit, static_argnums=(0, 4))
    def _train(self, x, y, key, num_steps: int = 1000):
        data = Dataset(x, y)
        key, subkey = random.split(key)
        params = self.initialize_parameters(subkey)
        opt_state = self.tx.init(params)

        def train_step(carry, ins):
            opt_state, params, subkey = carry[0], carry[1], carry[2]
            new_opt_state, new_params, loss = self._step_jit(opt_state, params, data, subkey)
            outs = [loss]
            key, subkey = random.split(subkey, 2)
            carry = [
                new_opt_state, new_params, key, carry[-1] + 1
            ]
            return carry, outs

        carry = [opt_state, params, subkey, 0]
        carry, outs = jax.lax.scan(train_step, carry, xs=None, length=num_steps)
        params = carry[1]
        return params

    @staticmethod
    def _calculate_calibration_score(predict_fn, params, xs, ys, ps, alpha, output_dim):
        assert alpha.shape == (output_dim,)

        out_mean, out_std, y_mean, y_std = predict_fn(xs, params)
        cdfs = vmap(norm.cdf)(ys, y_mean, y_std * alpha)

        def check_cdf(cdf):
            assert cdf.shape == ()
            return cdf <= ps

        cdfs = vmap(vmap(check_cdf, out_axes=1), out_axes=1)(cdfs)
        return jnp.mean(cdfs, axis=1)

    def update_dataset(self, x, y):
        dataset = Dataset(X=x, y=y)
        self.parameter_state = GPModelParameters(
            gp_parameter_state=self.parameter_state.gp_parameter_state,
            dataset=dataset,
        )
