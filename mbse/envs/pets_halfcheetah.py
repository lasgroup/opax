""" Taken from:  https://github.com/facebookresearch/mbrl-lib/blob/main/mbrl/env/pets_halfcheetah.py"""

from copy import deepcopy
import numpy as np
from mbse.models.environment_models.halfcheetah_reward_model import HalfCheetahReward
from mbse.envs.dm_control_env import DeepMindBridge
from dm_control.suite.cheetah import Cheetah, _DEFAULT_TIME_LIMIT, get_model_and_assets, Physics
from dm_control.rl.control import Environment
import collections
from dm_control.utils import containers
import os
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.spaces import Box
from typing import Optional

SUITE = containers.TaggedTasks()


@SUITE.add('benchmarking')
def run(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the run task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = PetsCheetah(random=random)
    environment_kwargs = environment_kwargs or {}
    return Environment(physics, task, time_limit=time_limit,
                       **environment_kwargs)


class PetsCheetah(Cheetah):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_observation(self, physics):
        """Returns an observation of the state, ignoring horizontal position."""
        obs = collections.OrderedDict()
        # Ignores horizontal position to maintain translational invariance.
        obs["position"] = physics.data.qpos.copy()
        obs['velocity'] = physics.velocity()
        return obs

    def sample_state(self, physics, factor=5.0, vel_noise_factor=10):
        """Sets the state of the environment at the start of each episode."""
        # The indexing below assumes that all joints have a single DOF.
        assert physics.model.nq == physics.model.njnt
        lower, upper = physics.model.jnt_range.T * factor
        upper_v = np.ones_like(lower) * vel_noise_factor
        lower_v = - upper_v
        physics.data.qpos = self.random.uniform(lower, upper)
        physics.data.qvel = self.random.uniform(upper_v, lower_v)

        # Stabilize the model before the actual simulation.
        physics.step(nstep=200)

        physics.data.time = 0
        super().initialize_episode(physics)


class HalfCheetahEnvDM(DeepMindBridge):
    def __init__(self, reward_model: HalfCheetahReward, *args, **kwargs):
        self.prev_qpos = None
        self.reward_model = reward_model
        env = run(time_limit=float('inf'), environment_kwargs={'flat_observation': True})
        super().__init__(env=env, *args, **kwargs)
        self.env = env

    def step(self, action):
        obs, reward, terminate, truncate, info = super().step(action)
        reward = self.reward_model.predict(obs=obs, action=action, next_obs=obs)
        reward = reward.astype(float).item()
        return obs, reward, terminate, truncate, info

    def sample_obs(self):
        physics = deepcopy(self.env.physics)
        task = deepcopy(self.env._task)
        return task.get_observation(physics)


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(self, render_mode: str = None, reward_model: HalfCheetahReward = HalfCheetahReward()):
        self.prev_qpos = None
        dir_path = os.path.dirname(os.path.realpath(__file__))
        observation_space = Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64)
        mujoco_env.MujocoEnv.__init__(
            self,
            "%s/assets/half_cheetah.xml" % dir_path,
            5,
            observation_space,
            render_mode,
        )
        utils.EzPickle.__init__(self)
        self.reward_model = reward_model

    def step(self, action):
        self.prev_qpos = np.copy(self.data.qpos.flat)
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        reward = self.reward_model.predict(next_obs=ob, action=action, obs=ob)
        terminated = False
        reward = reward.astype(float).item()
        return ob, reward, terminated, False, {}

    def _get_obs(self):
        return np.concatenate(
            [
                (self.data.qpos[:1] - self.prev_qpos[:1]) / self.dt,
                self.data.qpos[1:],
                self.data.qvel,
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + np.random.normal(loc=0, scale=0.001, size=self.model.nq)
        qvel = self.init_qvel + np.random.normal(loc=0, scale=0.001, size=self.model.nv)
        self.set_state(qpos, qvel)
        self.prev_qpos = np.copy(self.data.qpos)
        return self._get_obs()

    def sample_obs(self):
        qpos = self.init_qpos + np.random.normal(loc=0, scale=0.1, size=self.model.nq)
        qvel = self.init_qvel + np.random.normal(loc=0, scale=0.1, size=self.model.nv)
        self.set_state(qpos, qvel)
        self.prev_qpos = np.copy(self.data.qpos)
        state = self._get_obs()
        self.reset_model()
        return state

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.25
        self.viewer.cam.elevation = -55

    @staticmethod
    def _preprocess_state_np(state):
        assert isinstance(state, np.ndarray)
        assert state.ndim in (1, 2, 3)
        d1 = state.ndim == 1
        if d1:
            # if input is 1d, expand it to 2d
            state = np.expand_dims(state, 0)
        # [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.] ->
        # [1., sin(2), cos(2)., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.]
        ret = np.concatenate(
            [
                state[..., 1:2],
                np.sin(state[..., 2:3]),
                np.cos(state[..., 2:3]),
                state[..., 3:],
            ],
            axis=state.ndim - 1,
        )
        if d1:
            # and squeeze it back afterwards
            ret = ret.squeeze()
        return ret

    @staticmethod
    def preprocess_fn(state):
        if isinstance(state, np.ndarray):
            return HalfCheetahEnv._preprocess_state_np(state)
        raise ValueError("Invalid state type (must be np.ndarray).")


if __name__ == "__main__":
    from gym.wrappers.time_limit import TimeLimit
    env = HalfCheetahEnv(reward_model=HalfCheetahReward())
    env = TimeLimit(env, max_episode_steps=1000)
    obs, _ = env.reset()
    for i in range(1999):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
