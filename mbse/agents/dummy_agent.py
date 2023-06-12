import jax.numpy as jnp
from mbse.utils.replay_buffer import ReplayBuffer
import numpy as np
from typing import Callable


class DummyAgent(object):

    def __init__(self, train_steps: int = 1, batch_size: int = 256, num_epochs: int = -1, max_train_steps: int = 5000):
        self.train_steps = train_steps
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.max_train_steps = max_train_steps
        self.act_in_train = lambda obs, rng: self.act(obs, rng, eval=False)
        pass

    def act(self, obs: np.ndarray, rng=None, eval: bool = False, eval_idx: int = 0):
        return np.asarray(self.act_in_jax(jnp.asarray(obs), rng, eval=eval, eval_idx=eval_idx))

    def act_in_jax(self, obs: jnp.ndarray, rng=None, eval: bool = False, eval_idx: int = 0):
        NotImplementedError

    def train_step(self,
                   rng,
                   buffer: ReplayBuffer,
                   validate: bool = True,
                   log_results: bool = True,
                   ):
        NotImplementedError

    def prepare_agent_for_rollout(self):
        pass
