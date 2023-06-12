import numpy as np

from typing import Optional, Union, Dict
from mbse.utils.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper


class VecExtractDictObs(VecEnvWrapper):
    """
    A vectorized wrapper for extracting dictionary observations.

    :param venv: The vectorized environment
    :param key: The key of the dictionary observation
    """

    def __init__(self, venv: VecEnv, key: str):
        self.key = key
        super().__init__(venv=venv, observation_space=venv.observation_space.spaces[self.key])

    def reset(self, seed: Optional[int] = None) -> Union[np.ndarray, Dict]:
        obs, info = self.venv.reset(seed=seed)
        return obs[self.key], info

    def step_wait(self) -> VecEnvStepReturn:
        obs, reward, terminate, truncate, info = self.venv.step_wait()
        return obs[self.key], reward, terminate, truncate, info
