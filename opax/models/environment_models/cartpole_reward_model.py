import jax.numpy as jnp
from opax.models.reward_model import RewardModel
from functools import partial
import jax


class CartPoleRewardModel():
    _CART_RANGE = (-.25, .25)
    _ANGLE_COSINE_RANGE = (.995, 1)

    def __init__(self, num_poles: int = 1, action_cost: float = 0.0):
        self.num_poles = num_poles
        self.pos_factor = 1
        self.angle_factor = 10
        self.action_cost = action_cost

    def _get_reward(self, obs, action):
        cart_position = obs[..., 0]
        pole_angle_cosine = obs[..., 1: 1 + self.num_poles]
        # penalize non zero position
        pos_reward = - jnp.square(cart_position)
        # penalize non zero angle
        angle_reward = - jnp.square(pole_angle_cosine - jnp.ones_like(pole_angle_cosine)).sum(-1)
        action_reward = - jnp.square(action).sum(-1)
        return pos_reward * self.pos_factor + angle_reward * self.angle_factor + self.action_cost * action_reward

    @partial(jax.jit, static_argnums=0)
    def predict(self, obs, action, next_obs=None, rng=None):
        reward = self._get_reward(obs, action)
        reward = reward.reshape(-1).squeeze()
        return reward
