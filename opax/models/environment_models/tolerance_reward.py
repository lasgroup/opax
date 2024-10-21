from typing import Tuple

import jax
import jax.numpy as jnp


class Sigmoids:
    def __init__(self, sigmoid: str, value_at_the_margin: float = 0.1):
        self.sigmoid = sigmoid
        self.value_at_the_margin = value_at_the_margin

    def __call__(self, x, value_at_1):
        if self.sigmoid == "gaussian":
            return self._gaussian(x, value_at_1)
        elif self.sigmoid == "hyperbolic":
            return self._hyperbolic(x, value_at_1)
        elif self.sigmoid == "long_tail":
            return self._long_tail(x, value_at_1)
        elif self.sigmoid == 'reciprocal':
            return self._reciprocal(x, value_at_1)
        elif self.sigmoid == 'cosine':
            return self._cosine(x, value_at_1)
        elif self.sigmoid == 'linear':
            return self._linear(x, value_at_1)
        elif self.sigmoid == 'quadratic':
            return self._quadratic(x, value_at_1)
        elif self.sigmoid == 'tanh_squared':
            return self._tanh_squared(x, value_at_1)

    @staticmethod
    def _gaussian(x, value_at_1):
        scale = jnp.sqrt(-2 * jnp.log(value_at_1))
        return jnp.exp(-0.5 * (x * scale) ** 2)

    @staticmethod
    def _hyperbolic(x, value_at_1):
        scale = jnp.arccosh(1 / value_at_1)
        return 1 / jnp.cosh(x * scale)

    @staticmethod
    def _long_tail(x, value_at_1):
        scale = jnp.sqrt(1 / value_at_1 - 1)
        return 1 / ((x * scale) ** 2 + 1)

    @staticmethod
    def _reciprocal(x, value_at_1):
        scale = 1 / value_at_1 - 1
        return 1 / (jnp.abs(x) * scale + 1)

    @staticmethod
    def _cosine(x, value_at_1):
        scale = jnp.arccos(2 * value_at_1 - 1) / jnp.pi
        scaled_x = x * scale
        cos_pi_scaled_x = jnp.cos(jnp.pi * scaled_x)
        return jnp.where(jnp.abs(scaled_x) < 1, (1 + cos_pi_scaled_x) / 2, 0.0)

    @staticmethod
    def _linear(x, value_at_1):
        scale = 1 - value_at_1
        scaled_x = x * scale
        return jnp.where(jnp.abs(scaled_x) < 1, 1 - scaled_x, 0.0)

    @staticmethod
    def _quadratic(x, value_at_1):
        scale = jnp.sqrt(1 - value_at_1)
        scaled_x = x * scale
        return jnp.where(jnp.abs(scaled_x) < 1, 1 - scaled_x ** 2, 0.0)

    @staticmethod
    def _tanh_squared(x, value_at_1):
        scale = jnp.arctanh(jnp.sqrt(1 - value_at_1))
        return 1 - jnp.tanh(x * scale) ** 2


class ToleranceReward:
    def __init__(self,
                 bounds: Tuple[float, float] = (0.0, 0.0),
                 margin: float = 0.0,
                 sigmoid: str = 'gaussian',
                 value_at_margin: float = 0.1):
        self.bounds = bounds
        self.margin = margin
        self.value_at_margin = value_at_margin
        self._sigmoid = sigmoid
        self.sigmoid = Sigmoids(sigmoid)

        lower, upper = bounds
        self.lower = lower
        self.upper = upper
        if lower > upper:
            raise ValueError('Lower bound must be <= upper bound.')
        if margin < 0:
            raise ValueError('`margin` must be non-negative.')

    def __call__(self, x: jax.Array) -> jax.Array:
        in_bounds = jnp.logical_and(self.lower <= x, x <= self.upper)
        if self.margin == 0:
            return jnp.where(in_bounds, 1.0, 0.0)
        else:
            d = jnp.where(x < self.lower, self.lower - x, x - self.upper) / self.margin
            return jnp.where(in_bounds, 1.0, self.sigmoid(d, self.value_at_margin))
