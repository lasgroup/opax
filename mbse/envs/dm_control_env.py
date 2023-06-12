import gym
from gym.utils import seeding
import numpy as np
from dm_control.rl.control import Environment
from typing import Optional


class DeepMindBridge(gym.Env):
    def __init__(self, env: Environment, render_mode: str = 'rgb_array'):
        self._env = env
        self._env._flat_observation = True
        super().__init__()
        self.render_mode = render_mode

    def step(self, action):
        time_step = self._env.step(action)
        obs = time_step.observation['observations']
        reward = time_step.reward or 0
        truncate = False
        terminate = False
        if self._env._reset_next_step:
            terminate = self._env._task.get_termination(self._env._physics)
        self._env._reset_next_step = False
        return obs, reward, terminate, truncate, {}

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    @property
    def observation_space(self):
        spec = self._env.observation_spec()['observations']
        return gym.spaces.Box(-np.inf, np.inf, spec.shape, dtype=spec.dtype)

    def render(self, **kwargs):
        mode = self.render_mode
        if 'camera_id' not in kwargs.keys():
            kwargs['camera_id'] = 0
        return self._env.physics.render(**kwargs)

    def reset(self, seed: [Optional] = None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
            self._env._task._random = np.random.RandomState(seed)
        time_step = self._env.reset()
        obs = time_step.observation['observations']
        return obs, {}

    def seed(self, seed=None):
        self._env.task.random.seed(seed)


if __name__ == "__main__":
    from dm_control.suite.cheetah import run
    env = run(time_limit=float('inf'), environment_kwargs={'flat_observation': True})
    env = DeepMindBridge(env=env, render_mode="rgb_array")
    obs, _ = env.reset()
    for i in range(100):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        env.render()