from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
from mbse.models.dynamics_model import DynamicsModel
from mbse.models.reward_model import RewardModel
import jax.numpy as jnp
from functools import partial
import jax
from mbse.utils.type_aliases import ModelProperties


class MountainCarRewardModel(RewardModel):

    def __init__(self, env: Continuous_MountainCarEnv = Continuous_MountainCarEnv(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = env

    @partial(jax.jit, static_argnums=(0,))
    def predict(self, obs, action, next_obs=None, rng=None):
        pos = next_obs[..., 0]
        velocity = next_obs[..., 1]
        terminate = jnp.logical_and(pos >= self.env.goal_position, velocity >= self.env.goal_velocity)
        # reward = - (action ** 2) * 0.1 - 100 * ((pos - self.env.goal_position) ** 2 +
        #                                        (velocity - self.env.goal_velocity) ** 2)
        reward = - (action[..., 0] ** 2) * 0.1 + 100 * terminate
        return reward.reshape(-1).squeeze()


class MountainCarDynamics(DynamicsModel):

    def __init__(self, env: Continuous_MountainCarEnv = Continuous_MountainCarEnv(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = env
        self.reward_model = MountainCarRewardModel(env=env)
        self.obs_dim = 2
        self.act_dim = 1

    @partial(jax.jit, static_argnums=(0,))
    def predict(self, obs, action, rng=None, *args, **kwargs):
        obs = jnp.atleast_2d(obs).reshape(-1, 2)
        action = jnp.atleast_2d(action).reshape(-1, 1)
        pos = obs[..., 0]
        velocity = obs[..., 1]
        force = jnp.clip(action, a_min=-1, a_max=1) * self.env.max_action
        next_velocity = velocity + force * self.env.power - 0.0025 * jnp.cos(3 * pos)
        next_velocity = jnp.clip(next_velocity, a_min=-self.env.max_speed, a_max=self.env.max_speed)
        next_position = pos + next_velocity
        next_position = jnp.clip(next_position, self.env.min_position, self.env.max_position)
        out_of_bounds = jnp.logical_and(next_position - self.env.min_position <= 0.0, next_velocity < 0.0)
        next_velocity = (1 - out_of_bounds) * next_velocity
        next_obs = jnp.concatenate([next_position, next_velocity], axis=-1).reshape(-1, 2).squeeze()
        return next_obs

    @partial(jax.jit, static_argnums=(0,))
    def evaluate(self,
                 obs,
                 action,
                 parameters=None,
                 rng=None,
                 sampling_idx=None,
                 model_props: ModelProperties = ModelProperties()):
        next_state = self.predict(obs=obs, action=action, rng=rng)
        reward = self.reward_model.predict(obs=obs, action=action, next_obs=next_state)
        return next_state, reward
