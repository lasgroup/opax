"""Generate colored noise. Taken from: https://github.com/felixpatzelt/colorednoise/blob/master/colorednoise.py"""
import numpy as np
from jax.numpy import sqrt, newaxis
from jax.numpy.fft import irfft, rfftfreq
import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable, Union, Optional, NamedTuple
from numpy.random import default_rng, Generator, RandomState
from numpy import integer
from mbse.optimizers.dummy_policy_optimizer import DummyPolicyOptimizer, BestSequences
from mbse.utils.type_aliases import ModelProperties
from mbse.utils.utils import sample_trajectories


@partial(jax.jit, static_argnums=(0, 1, 3))
def powerlaw_psd_gaussian(exponent: float, size: int, rng: jax.random.PRNGKey, fmin: float = 0) -> jax.Array:
    """Gaussian (1/f)**beta noise.
    Based on the algorithm in:
    Timmer, J. and Koenig, M.:
    On generating power law noise.
    Astron. Astrophys. 300, 707-710 (1995)
    Normalised to unit variance
    Parameters:
    -----------
    exponent : float
        The power-spectrum of the generated noise is proportional to
        S(f) = (1 / f)**beta
        flicker / pink noise:   exponent beta = 1
        brown noise:            exponent beta = 2
        Furthermore, the autocorrelation decays proportional to lag**-gamma
        with gamma = 1 - beta for 0 < beta < 1.
        There may be finite-size issues for beta close to one.
    shape : int or iterable
        The output has the given shape, and the desired power spectrum in
        the last coordinate. That is, the last dimension is taken as time,
        and all other components are independent.
    fmin : float, optional
        Low-frequency cutoff.
        Default: 0 corresponds to original paper.

        The power-spectrum below fmin is flat. fmin is defined relative
        to a unit sampling rate (see numpy's rfftfreq). For convenience,
        the passed value is mapped to max(fmin, 1/samples) internally
        since 1/samples is the lowest possible finite frequency in the
        sample. The largest possible value is fmin = 0.5, the Nyquist
        frequency. The output for this value is white noise.
    random_state :  int, numpy.integer, numpy.random.Generator, numpy.random.RandomState,
                    optional
        Optionally sets the state of NumPy's underlying random number generator.
        Integer-compatible values or None are passed to np.random.default_rng.
        np.random.RandomState or np.random.Generator are used directly.
        Default: None.
    Returns
    -------
    out : array
        The samples.
    Examples:
    ---------
    # generate 1/f noise == pink noise == flicker noise
    """

    # Make sure size is a list so we can iterate it and assign to it.
    try:
        size = list(size)
    except TypeError:
        size = [size]

    # The number of samples in each time series
    samples = size[-1]

    # Calculate Frequencies (we asume a sample rate of one)
    # Use fft functions for real output (-> hermitian spectrum)
    f = rfftfreq(samples)

    # Validate / normalise fmin
    if 0 <= fmin <= 0.5:
        fmin = max(fmin, 1. / samples)  # Low frequency cutoff
    else:
        raise ValueError("fmin must be chosen between 0 and 0.5.")

    # Build scaling factors for all frequencies

    # s_scale = f
    # ix = npsum(s_scale < fmin)  # Index of the cutoff
    # if ix and ix < len(s_scale):
    #    s_scale[:ix] = s_scale[ix]
    # s_scale = s_scale ** (-exponent / 2.)
    s_scale = f
    ix = jnp.sum(s_scale < fmin)  # Index of the cutoff

    def cutoff(x, idx):
        x_idx = jax.lax.dynamic_slice(x, start_indices=(idx,), slice_sizes=(1,))
        y = jnp.ones_like(x) * x_idx
        indexes = jnp.arange(0, x.shape[0], step=1)
        first_idx = indexes < idx
        z = (1 - first_idx) * x + first_idx * y
        return z

    def no_cutoff(x, idx):
        return x

    s_scale = jax.lax.cond(
        jnp.logical_and(ix < len(s_scale), ix),
        cutoff,
        no_cutoff,
        s_scale,
        ix
    )
    s_scale = s_scale ** (-exponent / 2.)

    # Calculate theoretical output standard deviation from scaling
    w = s_scale[1:].copy()
    w = w.at[-1].set(w[-1] * (1 + (samples % 2)) / 2.)  # correct f = +-0.5
    sigma = 2 * sqrt(jnp.sum(w ** 2)) / samples

    # Adjust size to generate one Fourier component per frequency
    size[-1] = len(f)

    # Add empty dimension(s) to broadcast s_scale along last
    # dimension of generated random power + phase (below)
    dims_to_add = len(size) - 1
    s_scale = s_scale[(newaxis,) * dims_to_add + (Ellipsis,)]

    # prepare random number generator
    key_sr, key_si, rng = jax.random.split(rng, 3)
    sr = jax.random.normal(key=key_sr, shape=s_scale.shape) * s_scale
    si = jax.random.normal(key=key_si, shape=s_scale.shape) * s_scale

    # If the signal length is even, frequencies +/- 0.5 are equal
    # so the coefficient must be real.
    if not (samples % 2):
        si = si.at[..., -1].set(0)
        sr = sr.at[..., -1].set(sr[..., -1] * sqrt(2))  # Fix magnitude

    # Regardless of signal length, the DC component must be real
    si = si.at[..., 0].set(0)
    sr = sr.at[..., 0].set(sr[..., 0] * sqrt(2))  # Fix magnitude

    # Combine power + corrected phase to Fourier components
    s = sr + 1J * si

    # Transform to real time series & scale to unit variance
    y = irfft(s, n=samples, axis=-1) / sigma
    return y


"""Generate colored noise."""


def powerlaw_psd_gaussian_numpy(exponent, size, fmin=0, random_state=None):
    """Gaussian (1/f)**beta noise.
    Based on the algorithm in:
    Timmer, J. and Koenig, M.:
    On generating power law noise.
    Astron. Astrophys. 300, 707-710 (1995)
    Normalised to unit variance
    Parameters:
    -----------
    exponent : float
        The power-spectrum of the generated noise is proportional to
        S(f) = (1 / f)**beta
        flicker / pink noise:   exponent beta = 1
        brown noise:            exponent beta = 2
        Furthermore, the autocorrelation decays proportional to lag**-gamma
        with gamma = 1 - beta for 0 < beta < 1.
        There may be finite-size issues for beta close to one.
    shape : int or iterable
        The output has the given shape, and the desired power spectrum in
        the last coordinate. That is, the last dimension is taken as time,
        and all other components are independent.
    fmin : float, optional
        Low-frequency cutoff.
        Default: 0 corresponds to original paper.

        The power-spectrum below fmin is flat. fmin is defined relative
        to a unit sampling rate (see numpy's rfftfreq). For convenience,
        the passed value is mapped to max(fmin, 1/samples) internally
        since 1/samples is the lowest possible finite frequency in the
        sample. The largest possible value is fmin = 0.5, the Nyquist
        frequency. The output for this value is white noise.
    random_state :  int, numpy.integer, numpy.random.Generator, numpy.random.RandomState,
                    optional
        Optionally sets the state of NumPy's underlying random number generator.
        Integer-compatible values or None are passed to np.random.default_rng.
        np.random.RandomState or np.random.Generator are used directly.
        Default: None.
    Returns
    -------
    out : array
        The samples.
    Examples:
    ---------
    # generate 1/f noise == pink noise == flicker noise

    """
    from numpy import sqrt, newaxis
    from numpy.fft import irfft, rfftfreq
    from numpy import sum as npsum

    # Make sure size is a list so we can iterate it and assign to it.
    try:
        size = list(size)
    except TypeError:
        size = [size]

    # The number of samples in each time series
    samples = size[-1]

    # Calculate Frequencies (we asume a sample rate of one)
    # Use fft functions for real output (-> hermitian spectrum)
    f = rfftfreq(samples)

    # Validate / normalise fmin
    if 0 <= fmin <= 0.5:
        fmin = max(fmin, 1. / samples)  # Low frequency cutoff
    else:
        raise ValueError("fmin must be chosen between 0 and 0.5.")

    # Build scaling factors for all frequencies
    s_scale = f
    ix = npsum(s_scale < fmin)  # Index of the cutoff
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = s_scale ** (-exponent / 2.)

    # Calculate theoretical output standard deviation from scaling
    w = s_scale[1:].copy()
    w[-1] *= (1 + (samples % 2)) / 2.  # correct f = +-0.5
    sigma = 2 * sqrt(npsum(w ** 2)) / samples

    # Adjust size to generate one Fourier component per frequency
    size[-1] = len(f)

    # Add empty dimension(s) to broadcast s_scale along last
    # dimension of generated random power + phase (below)
    dims_to_add = len(size) - 1
    s_scale = s_scale[(newaxis,) * dims_to_add + (Ellipsis,)]

    # prepare random number generator
    normal_dist = _get_normal_distribution(random_state)

    # Generate scaled random power + phase
    # sr = np.random.normal(scale=s_scale)
    # si = np.random.normal(scale=s_scale)
    sr = normal_dist(scale=s_scale, size=size)
    si = normal_dist(scale=s_scale, size=size)

    # If the signal length is even, frequencies +/- 0.5 are equal
    # so the coefficient must be real.
    if not (samples % 2):
        si[..., -1] = 0
        sr[..., -1] *= sqrt(2)  # Fix magnitude

    # Regardless of signal length, the DC component must be real
    si[..., 0] = 0
    sr[..., 0] *= sqrt(2)  # Fix magnitude

    # Combine power + corrected phase to Fourier components
    s = sr + 1J * si

    # Transform to real time series & scale to unit variance
    y = irfft(s, n=samples, axis=-1) / sigma

    return y


def _get_normal_distribution(random_state):
    normal_dist = None
    if isinstance(random_state, (integer, int)) or random_state is None:
        random_state = default_rng(random_state)
        normal_dist = random_state.normal
    elif isinstance(random_state, (Generator, RandomState)):
        normal_dist = random_state.normal
    else:
        raise ValueError(
            "random_state must be one of integer, numpy.random.Generator, "
            "numpy.random.Randomstate"
        )
    return normal_dist


class ICEMHyperparams(NamedTuple):
    """
    maxiter: maximum iterations.
    grad_norm_threshold: tolerance for stopping optimization.
    make_psd: whether to zero negative eigenvalues after quadratization.
    psd_delta: The delta value to make the problem PSD. Specifically, it will
        ensure that d^2c/dx^2 and d^2c/du^2, i.e. the hessian of cost function
        with respect to state and control are always positive definite.
    alpha_0: initial line search value.
    alpha_min: minimum line search value.
    """
    num_samples: int = 500
    num_elites: int = 50
    init_std: float = 0.5
    alpha: float = 0.0
    num_steps: int = 1
    exponent: float = 0.0
    elite_set_fraction: float = 0.3


class ICemTO(DummyPolicyOptimizer):
    def __init__(self,
                 horizon: int,
                 action_dim: tuple,
                 dynamics_model_list: list,
                 seed: int = 0,
                 n_particles: int = 10,
                 num_samples: int = 500,
                 num_elites: int = 50,
                 init_std: float = 0.5,
                 alpha: float = 0.0,
                 num_steps: int = 5,
                 exponent: float = 0.0,
                 elite_set_fraction: float = 0.3,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.horizon = horizon
        self.n_particles = n_particles
        self.opt_params = ICEMHyperparams(
            num_samples=num_samples,
            num_elites=num_elites,
            init_std=init_std,
            alpha=alpha,
            num_steps=num_steps,
            exponent=exponent,
            elite_set_fraction=elite_set_fraction,
        )
        self.action_dim = action_dim
        self.opt_dim = (horizon,) + action_dim
        self.seed = seed
        assert isinstance(dynamics_model_list, list)
        self.dynamics_model_list = dynamics_model_list
        self.best_sequences = BestSequences(
            evaluation_sequences=jnp.zeros((len(self.dynamics_model_list),) + self.opt_dim),
            exploration_sequence=jnp.zeros(self.opt_dim),
        )
        self._init_fn()

    def _init_fn(self):
        self.optimize_for_eval_fns = []
        for i in range(len(self.dynamics_model_list)):
            self.optimize_for_eval_fns.append(partial(
                self._get_action_sequence, model_index=i
            ))
        self.optimize = self.optimize_for_eval_fns[0]

    @partial(jax.jit, static_argnums=(0, 1))
    def _optimize_with_params(
            self,
            evaluate_fn: Callable,
            initial_state: jax.Array,
            dynamics_params,
            model_props: ModelProperties = ModelProperties(),
            initial_actions: Optional[jax.Array] = None,
            key: Optional = None,
            optimizer_key: Optional = None,
            sampling_idx: Optional[Union[jnp.ndarray, int]] = None,
    ):
        initial_state = jnp.repeat(jnp.expand_dims(initial_state, 0), self.n_particles, 0)
        eval_func = lambda seq, x, k: sample_trajectories(
            evaluate_fn=evaluate_fn,
            parameters=dynamics_params,
            init_state=x,
            horizon=self.horizon,
            key=k,
            actions=seq,
            model_props=model_props,
            sampling_idx=sampling_idx,
        )

        def sum_rewards(seq):
            def get_average_reward(obs, eval_key):
                transition = eval_func(seq, obs, eval_key)
                return transition.reward.mean()

            if key is not None:
                optimizer_key = jax.random.split(key=key, num=self.n_particles)
                returns = jax.vmap(get_average_reward)(initial_state, optimizer_key)
            else:
                returns = jax.vmap(get_average_reward, in_axes=(0, None))(initial_state, key)
            return returns.mean()

        if optimizer_key is None:
            optimizer_key = jax.random.PRNGKey(self.seed)
        get_best_action = lambda best_val, best_seq, val, seq: [val[-1].squeeze(), seq[-1]]
        get_curr_best_action = lambda best_val, best_seq, val, seq: [best_val, best_seq]
        num_prev_elites_per_iter = max(int(self.opt_params.elite_set_fraction * self.opt_params.num_elites), 1)

        def step(carry, ins):
            key = carry[0]
            mu = carry[1]
            sig = carry[2]
            best_val = carry[3]
            best_seq = carry[4]
            prev_elites = carry[5]
            mu = mu.reshape(-1, 1).squeeze()
            sig = sig.reshape(-1, 1).squeeze()
            sampling_rng = jax.random.split(key=key, num=self.opt_params.num_samples + 1)
            key = sampling_rng[0]
            sampling_rng = sampling_rng[1:]
            opt_size = self.opt_dim[0] * self.opt_dim[1]
            colored_samples = jax.vmap(
                lambda rng: powerlaw_psd_gaussian(exponent=self.opt_params.exponent, size=opt_size, rng=rng))(
                sampling_rng)
            action_samples = mu + colored_samples * sig
            action_samples = jnp.clip(action_samples, a_max=1, a_min=-1)
            action_samples = action_samples.reshape((-1,) + self.opt_dim)
            action_samples = jnp.concatenate([action_samples, prev_elites], axis=0)
            values = jax.vmap(sum_rewards)(action_samples)
            best_elite_idx = np.argsort(values, axis=0).squeeze()[-self.opt_params.num_elites:]
            elites = action_samples[best_elite_idx]
            elite_values = values[best_elite_idx]
            elite_mean = jnp.mean(elites, axis=0)
            elite_var = jnp.var(elites, axis=0)
            mean = mu.reshape(self.opt_dim) * self.opt_params.alpha + (1 - self.opt_params.alpha) * elite_mean
            var = jnp.square(sig.reshape(self.opt_dim)) * self.opt_params.alpha + (
                        1 - self.opt_params.alpha) * elite_var
            std = jnp.sqrt(var)
            best_elite = elite_values[-1].squeeze()
            bests = jax.lax.cond(best_val <= best_elite,
                                 get_best_action,
                                 get_curr_best_action,
                                 best_val,
                                 best_seq,
                                 elite_values,
                                 elites)
            best_val = bests[0]
            best_seq = bests[-1]
            outs = [best_val, best_seq]
            elite_set = jnp.atleast_2d(elites[-num_prev_elites_per_iter:]).reshape((-1,) + self.opt_dim)
            carry = [key, mean, std, best_val, best_seq, elite_set]
            return carry, outs

        best_value = -jnp.inf
        if initial_actions is None:
            mean = jnp.zeros(self.opt_dim)
        else:
            assert initial_actions.shape == self.opt_dim
            mean = initial_actions
        std = jnp.ones(self.opt_dim) * self.opt_params.init_std
        best_sequence = mean
        if optimizer_key is None:
            optimizer_key = jax.random.PRNGKey(self.seed)
        prev_elites = jnp.zeros((num_prev_elites_per_iter,) + self.opt_dim)
        carry = [optimizer_key, mean, std, best_value, best_sequence, prev_elites]
        carry, outs = jax.lax.scan(step, carry, xs=None, length=self.opt_params.num_steps)
        return outs[1][-1, ...], outs[0][-1, ...]

    def _get_action_sequence(
            self,
            model_index,
            dynamics_params,
            obs,
            key=None,
            optimizer_key=None,
            model_props: ModelProperties = ModelProperties(),
            initial_actions: Optional[jax.Array] = None,
            sampling_idx: Optional[Union[jnp.ndarray, int]] = None,
    ):
        if initial_actions is None:
            prev_best = jnp.zeros_like(self.best_sequences.evaluation_sequences[model_index])
            prev_best = prev_best.at[:-1].set(self.best_sequences.evaluation_sequences[model_index][1:])
            last_input = self.best_sequences.evaluation_sequences[model_index][-1]
            prev_best = prev_best.at[-1].set(last_input)
            initial_actions = prev_best

        def optimize(init_state, rng, opt_rng):
            return self._optimize_with_params(
                evaluate_fn=self.dynamics_model_list[model_index].evaluate,
                initial_state=init_state,
                dynamics_params=dynamics_params,
                model_props=model_props,
                initial_actions=initial_actions,
                key=rng,
                optimizer_key=opt_rng,
                sampling_idx=sampling_idx,
            )

        best_sequence, best_reward = jax.vmap(optimize)(obs, key, optimizer_key)
        sequence_copy = self.best_sequences.evaluation_sequences.at[model_index].set(best_sequence[0])
        self.best_sequences = BestSequences(
            evaluation_sequences=sequence_copy,
            exploration_sequence=self.best_sequences.exploration_sequence,
        )
        return best_sequence, best_reward

    def optimize_for_exploration(
            self,
            dynamics_params,
            obs,
            key=None,
            optimizer_key=None,
            model_props: ModelProperties = ModelProperties(),
            initial_actions: Optional[jax.Array] = None,
            sampling_idx: Optional[Union[jnp.ndarray, int]] = None,
    ):
        if initial_actions is None:
            prev_best = jnp.zeros_like(self.best_sequences.exploration_sequence)
            prev_best = prev_best.at[:-1].set(self.best_sequences.exploration_sequence[1:])
            last_input = self.best_sequences.exploration_sequence[-1]
            prev_best = prev_best.at[-1].set(last_input)
            initial_actions = prev_best

        def optimize(init_state, rng, opt_rng):
            return self._optimize_with_params(
                evaluate_fn=self.dynamics_model.evaluate_for_exploration,
                initial_state=init_state,
                dynamics_params=dynamics_params,
                model_props=model_props,
                initial_actions=initial_actions,
                key=rng,
                optimizer_key=opt_rng,
                sampling_idx=sampling_idx,
            )

        best_sequence, best_reward = jax.vmap(optimize)(obs, key, optimizer_key)
        self.best_sequences = BestSequences(
            evaluation_sequences=self.best_sequences.evaluation_sequences,
            exploration_sequence=best_sequence[0],
        )
        return best_sequence, best_reward

    @property
    def dynamics_model(self):
        return self.dynamics_model_list[0]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    exponent_list = [0.0, 0.25, 1.0, 2.0, 3.0, 5.0, 10.0]
    # gridspec inside gridspec
    fig, axs = plt.subplots(2, len(exponent_list))
    fig.suptitle('Vertically stacked subplots')
    for i, exponent in enumerate(exponent_list):
        rng = jax.random.PRNGKey(seed=0)
        size = 10000
        samples = powerlaw_psd_gaussian(
            exponent=exponent,
            size=size,
            rng=rng,
        )
        rng = np.random.default_rng(seed=0)
        samples_np = powerlaw_psd_gaussian_numpy(
            exponent=exponent,
            size=size,
            random_state=rng,
        )
        x = np.arange(0, size)
        y = np.asarray(samples)
        axs[0, i].plot(x, y, label="Jax " + str(exponent))
        y = np.asarray(samples_np)
        axs[1, i].plot(x, y, label="Numpy " + str(exponent))
    check = True
