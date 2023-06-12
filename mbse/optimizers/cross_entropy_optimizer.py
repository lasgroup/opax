import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
from mbse.optimizers.dummy_optimizer import DummyOptimizer
from typing import Optional


class CrossEntropyOptimizer(DummyOptimizer):

    def __init__(
            self,
            num_samples: int = 500,
            num_elites: int = 50,
            seed: int = 0,
            init_std: float = 5,
            alpha: float = 0.0,
            *args,
            **kwargs,
    ):
        super(CrossEntropyOptimizer, self).__init__(*args, **kwargs)
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.seed = seed
        self.init_std = init_std
        assert 0 <= alpha < 1, "alpha must be between [0, 1]"
        self.alpha = alpha

    # @partial(jit, static_argnums=(0, 1))
    def step(self, func, mean, std, key):
        mean = mean.reshape(-1, 1).squeeze()
        std = std.reshape(-1, 1).squeeze()
        samples = mean + jax.random.multivariate_normal(
            key=key,
            mean=jnp.zeros_like(mean),
            cov=jnp.diag(jnp.ones_like(mean)),
            shape=(self.num_samples, )) * std
        samples = samples.reshape((self.num_samples,) + self.action_dim)
        samples = self.clip_action(samples)
        values = vmap(func)(samples)

        best_elite_idx = np.argsort(values, axis=0).squeeze()[-self.num_elites:]

        elites = samples[best_elite_idx]
        elite_values = values[best_elite_idx]
        return elites, elite_values

    # @partial(jit, static_argnums=(0, 1))
    def optimize(self, func, rng=None, mean: Optional[jax.Array] = None):
        best_value = -jnp.inf
        if mean is None:
            mean = jnp.zeros(self.action_dim)
        else:
            assert mean.shape == self.action_dim
        std = jnp.ones(self.action_dim)*self.init_std
        best_sequence = mean
        get_best_action = lambda best_val, best_seq, val, seq: [val[-1].squeeze(), seq[-1]]
        get_curr_best_action = lambda best_val, best_seq, val, seq: [best_val, best_seq]
        if rng is None:
            rng = jax.random.PRNGKey(self.seed)

        def step(carry, ins):
            key = carry[0]
            mu = carry[1]
            sig = carry[2]
            best_val = carry[3]
            best_seq = carry[4]
            key, sample_key = jax.random.split(key, 2)
            elites, elite_values = self.step(func, mu, sig, sample_key)
            elite_mean = jnp.mean(elites, axis=0)
            elite_var = jnp.var(elites, axis=0)
            mean = mu * self.alpha + (1 - self.alpha) * elite_mean
            var = jnp.square(sig) * self.alpha + (1 - self.alpha) * elite_var
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
            carry = [key, mean, std, best_val, best_seq]

            return carry, outs
        carry = [rng, mean, std, best_value, best_sequence]
        carry, outs = jax.lax.scan(step, carry, xs=None, length=self.num_steps)
        return outs[1][-1, ...], outs[0][-1, ...]


