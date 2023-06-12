import jax.numpy as jnp
import jax
from mbse.models.reward_model import RewardModel
from functools import partial

_RUN_SPEED = 10


class ReacherRewardModel(RewardModel):
    """Get Pendulum Reward."""

    def __init__(self, ctrl_cost: float = 0.0, scarce_reward: bool = False, tol: float = 5e-2):
        super().__init__()
        self.scarce_reward = scarce_reward
        self.ctrl_cost = ctrl_cost
        self.tol = tol

    @partial(jax.jit, static_argnums=0)
    def predict(self, obs, action, next_obs=None, rng=None):
        vec = obs[..., -2:]
        if self.scarce_reward:
            reward_ctrl = -jnp.square(action).sum(-1)
            reward = self.ctrl_cost * reward_ctrl
            dist = jnp.linalg.norm(vec, axis=-1)
            reward = reward + (dist < self.tol) * 100
        else:
            reward_dist = -jnp.linalg.norm(vec, axis=-1)
            reward_ctrl = -jnp.square(action).sum(-1)
            reward = reward_dist + self.ctrl_cost * reward_ctrl
        reward = reward.reshape(-1).squeeze()
        return reward
