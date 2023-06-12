from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
from mbse.models.environment_models.mountain_car import MountainCarDynamics
from typing import Optional
import numpy as np
from gym.envs.classic_control import utils


class CustomMountainCar(Continuous_MountainCarEnv):

    def __init__(self, dynamics_model: MountainCarDynamics = MountainCarDynamics(), render_mode='rgb_array',
                 *args, **kwargs):
        super().__init__(render_mode=render_mode, *args, **kwargs)
        self.dynamics_model = dynamics_model
        self.observation_space.sample = self.sample_obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(options, -0.6, -0.4)
        self.state = np.array([self.np_random.uniform(low=low, high=high), 0])

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def step(self, a):
        next_obs, reward = self.dynamics_model.evaluate(obs=self.state, action=a)
        self.state = next_obs
        if self.render_mode == "human":
            self.render()
        return next_obs, reward.squeeze().item(), False, False, {}

    def sample_obs(self):
        pos = self.np_random.uniform(low=self.min_position, high=self.max_position)
        velocity = self.np_random.uniform(low=-self.max_speed, high=self.max_speed)
        return np.asarray([pos, velocity])
