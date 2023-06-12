import jax.numpy as jnp
from typing import Optional, Union


class DummyOptimizer(object):
    def __init__(self,
                 action_dim=(1, ),
                 upper_bound: Optional[Union[float, jnp.ndarray]] = 1.0,
                 num_steps: int = 20,
                 lower_bound: Optional[Union[float, jnp.ndarray]] = None,
                 *args,
                 **kwargs):

        self.num_steps = num_steps
        self.action_dim = action_dim
        if upper_bound is None:
            upper_bound = jnp.ones(self.action_dim)*jnp.inf
        elif isinstance(upper_bound, float):
            upper_bound = jnp.ones(self.action_dim)*upper_bound
        self.upper_bound = upper_bound
        if lower_bound is None:
            lower_bound = - upper_bound
        elif isinstance(lower_bound, float):
            lower_bound = jnp.ones(self.action_dim)*lower_bound
        self.lower_bound = lower_bound

    def optimize(self, func, rng=None):
        pass

    def clip_action(self, action):
        return jnp.clip(action, a_min=self.lower_bound, a_max=self.upper_bound)
