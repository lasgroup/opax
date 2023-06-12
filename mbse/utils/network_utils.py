from typing import Sequence, Callable
import jax.numpy as jnp
from flax import linen as nn
from jax import jit, vmap


class MLP(nn.Module):
    features: Sequence[int]
    output_dim: int
    non_linearity: Callable = nn.swish

    @nn.compact
    def __call__(self, x, train=False):
        input = x
        for feat in self.features:
            input = nn.Dense(feat)(input)
            input = self.non_linearity(input)
        out = nn.Dense(features=self.output_dim)(input)
        return out

@vmap
@jit
def mse(x, y):
    return jnp.mean(jnp.square(x-y))

