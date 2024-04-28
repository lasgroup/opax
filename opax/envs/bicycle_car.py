from gym import spaces
from gym import Env
from typing import Optional
import numpy as np
from opax.models.environment_models.bicyclecar_model import BicycleCarReward, BicycleCarModel
import jax.numpy as jnp


class BicycleEnv(Env):

    def __init__(self,
                 dynamics_model: BicycleCarModel = BicycleCarModel(dt=1/30.),
                 reward_model: BicycleCarReward = BicycleCarReward(),
                 _np_random: Optional[np.random.Generator] = None,
                 render_mode: str = 'rgb_array',
                 ):
        super(BicycleEnv).__init__()
        self.render_mode = render_mode
        self.reward_model = reward_model
        self.dynamics_model = dynamics_model
        self.goal = np.asarray(self.reward_model.goal)

        high = np.asarray([np.inf,
                           np.inf,
                           1,
                           1,
                           np.inf,
                           np.inf,
                           np.inf]
                          )
        low = -high
        self.observation_space = spaces.Box(
            high=high,
            low=low,
        )
        self.dim_state = (7,)
        self.dim_action = (2,)
        high = np.ones(2)
        low = -high
        self.action_space = spaces.Box(
            high=high,
            low=low,
        )

        self.init_pos = np.array([1.42, -1.04, jnp.pi])
        self.init_state = np.concatenate([
            self.init_pos[..., :2],
            np.sin(self.init_pos[2]).reshape(-1),
            np.cos(self.init_pos[2]).reshape(-1),
            np.zeros(3)],
            axis=-1)
        self.state = self.init_state
        self._np_random = _np_random
        self.current_step = 0

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed, options=options)
        self.state = self.init_state
        self.current_step = 0
        return self.state, {}

    def step(self, action):
        state = jnp.asarray(self.state).reshape(-1, 7)
        action = jnp.asarray(action).reshape(-1, 2)
        next_state = self.dynamics_model.predict(state, action)
        reward = self.reward_model.predict(self.state, action)
        self.current_step += 1
        self.state = np.asarray(next_state).reshape(7)
        return next_state.squeeze(), np.asarray(reward).squeeze().item(), False, False, {}


if __name__ == "__main__":
    from gym.wrappers.time_limit import TimeLimit

    env = BicycleEnv(reward_model=BicycleCarReward())
    env = TimeLimit(env, max_episode_steps=1000)
    obs, _ = env.reset(seed=0)
    for i in range(1999):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, _ = env.reset()
        print(obs)
    env.close()
