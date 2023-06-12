"""Common aliases for type hints"""
from typing import Any, Dict, Tuple, Union

import gym
import numpy as np


from mbse.utils.vec_env import VecEnv

GymEnv = Union[gym.Env, VecEnv]
GymObs = Union[Tuple, Dict[str, Any], np.ndarray, int]
GymStepReturn = Tuple[GymObs, float, bool, Dict]
