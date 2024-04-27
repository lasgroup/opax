import jax.numpy as jnp
import jax
from opax.models.reward_model import RewardModel
from functools import partial


class SwimmerRewardModel(RewardModel):
    """Get Pendulum Reward."""

    def __init__(self, ctrl_cost_weight: float = 0.0, scarce_reward: bool = False, tol: float = 5e-2,
                 distance_weight: float = 1.0):
        super().__init__()
        self.ctrl_cost_weight = ctrl_cost_weight
        self.scarce_reward = scarce_reward
        self.tol = tol
        self.distance_weight = distance_weight

    @partial(jax.jit, static_argnums=0)
    def predict(self, obs, action, next_obs=None, rng=None):
        nose_to_target = obs[..., -2:]
        if self.scarce_reward:
            reward_ctrl = -jnp.square(action).sum(-1)
            reward = self.ctrl_cost_weight * reward_ctrl
            dist = jnp.linalg.norm(nose_to_target, axis=-1)
            reward = reward + (dist < self.tol) * 100
        else:
            reward_dist = -jnp.linalg.norm(nose_to_target, axis=-1)
            reward_ctrl = -jnp.square(action).sum(-1)
            reward = self.distance_weight * reward_dist + self.ctrl_cost_weight * reward_ctrl
        reward = reward.reshape(-1).squeeze()
        return reward
