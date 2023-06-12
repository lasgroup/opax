from mbse.models.environment_models.acrobot import AcrobotDynamics
from gym.envs.classic_control.acrobot import AcrobotEnv
from gym.spaces.box import Box
from typing import Optional
import numpy as np


class AcrobotEnvContinuous(AcrobotEnv):

    def __init__(self, dynamics_model: AcrobotDynamics = AcrobotDynamics(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = Box(
            shape=(1, ),
            low=-1,
            high=1,
        )
        self.dynamics_model = dynamics_model

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        # low, high = utils.maybe_parse_reset_bounds(
        #     options, -0.1, 0.1  # default low
        # )  # default high
        # self.state = self.np_random.uniform(low=low, high=high, size=(4,)).astype(
        #     np.float32
        # )
        theta_1, theta_2 = 0.0, 0.0
        w_1, w_2 = 0.0, 0.0
        self.state = np.asarray([theta_1, theta_2, w_1, w_2])
        # self.state = self.dynamics_model.get_full_obs(condensed_state)
        state = self.dynamics_model.get_full_obs(self.state)
        if self.render_mode == "human":
            self.render()
        return state, {}

    def step(self, a):
        full_obs = self.dynamics_model.get_full_obs(self.state)
        next_obs, reward = self.dynamics_model.evaluate(obs=full_obs, action=a)
        self.state = self.dynamics_model.get_condensed_obs(next_obs)
        if self.render_mode == "human":
            self.render()
        return next_obs, reward, False, False, {}


