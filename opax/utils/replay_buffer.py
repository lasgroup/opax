import functools

import chex
import jax.numpy as jnp
import jax
from flax import struct
import numpy as np
from typing import Union, Tuple, Optional
from copy import deepcopy

EPS = 1e-8


def identity_transform(obs: chex.Array, action: Optional[chex.Array] = None, next_state: Optional[chex.Array] = None) \
        -> [chex.Array, Optional[chex.Array], Optional[chex.Array]]:
    return obs, action, next_state


def inverse_identitiy_transform(transformed_obs: chex.Array, transformed_action: Optional[chex.Array] = None,
                                transformed_next_state: Optional[chex.Array] = None) -> [chex.Array,
                                                                                         Optional[chex.Array],
                                                                                         Optional[chex.Array]]:
    return identity_transform(transformed_obs, transformed_action, transformed_next_state)


@struct.dataclass
class Transition:
    obs: Union[np.ndarray, jnp.ndarray]
    action: Union[np.ndarray, jnp.ndarray]
    next_obs: Union[np.ndarray, jnp.ndarray]
    reward: Union[np.ndarray, jnp.ndarray]
    done: Union[np.ndarray, jnp.ndarray]

    @property
    def shape(self):
        return self.obs.shape[:-1]

    def reshape(self, dim_1, dim_2):
        tran = Transition(
            obs=self.obs.reshape(dim_1, dim_2, -1),
            action=self.action.reshape(dim_1, dim_2, -1),
            next_obs=self.next_obs.reshape(dim_1, dim_2, -1),
            reward=self.reward.reshape(dim_1, dim_2, -1),
            done=self.done.reshape(dim_1, dim_2, -1)
        )
        return tran

    # def get_idx(self, idx):
    #     tran = Transition(
    #         obs=self.obs[idx],
    #         action=self.action[idx],
    #         next_obs=self.next_obs[idx],
    #         reward=self.reward[idx],
    #         done=self.done[idx]
    #     )
    #     return tran
    #
    # def __iter__(self):
    #     for index in range(self.shape[0]):
    #         yield self.get_idx(index)

    def get_data(self):
        return self.obs, self.action, self.next_obs, self.reward, self.done


def merge_transitions(tran_a: Transition, tran_b: Transition, axis=0):
    obs = jnp.concatenate([tran_a.obs, tran_b.obs], axis=axis)
    action = jnp.concatenate([tran_a.action, tran_b.action], axis=axis)
    next_obs = jnp.concatenate([tran_a.next_obs, tran_b.next_obs], axis=axis)
    reward = jnp.concatenate([tran_a.reward, tran_b.reward], axis=axis)
    done = jnp.concatenate([tran_a.done, tran_b.done], axis=axis)
    return Transition(
        obs,
        action,
        next_obs,
        reward,
        done,
    )


def transition_to_jax(tran: Transition):
    return Transition(
        obs=jnp.asarray(tran.obs),
        action=jnp.asarray(tran.action),
        next_obs=jnp.asarray(tran.next_obs),
        reward=jnp.asarray(tran.reward),
        done=jnp.asarray(tran.done),
    )


class Normalizer(object):
    def __init__(self, input_shape: Tuple):
        self.mean = np.zeros(*input_shape)
        self.std = np.ones(*input_shape)
        self.size = 0

    def update(self, x: chex.Array):
        new_size = x.shape[0]
        total_size = new_size + self.size
        new_mean = (self.mean * self.size + np.sum(x, axis=0)) / total_size
        new_s_n = np.square(self.std) * self.size + np.sum(np.square(x - new_mean),
                                                           axis=0
                                                           ) + self.size * np.square(self.mean -
                                                                                     new_mean)

        new_var = new_s_n / total_size
        # new_var = (np.square(self.std)+np.square(self.mean))*self.size + \
        #          np.sum(np.square(x), axis=0)
        # new_var = new_var/total_size - np.square(new_mean)
        new_std = np.sqrt(new_var)
        self.mean = new_mean
        self.std = np.maximum(new_std, np.ones_like(new_std) * EPS)
        self.size = total_size

    def normalize(self, x: chex.Array):
        return (x - self.mean) / self.std

    def inverse(self, x: chex.Array):
        return x * self.std + self.mean


class ReplayBuffer(object):
    def __init__(self,
                 obs_shape: Tuple,
                 action_shape: Tuple,
                 max_size: int = int(1e6),
                 normalize: bool = False,
                 action_normalize: bool = False,
                 learn_deltas: bool = False
                 ):
        self.max_size = max_size
        self.current_ptr = 0
        self.size = 0
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.normalize = normalize
        self.learn_deltas = learn_deltas
        self.obs, self.action, self.next_obs, self.reward, self.done = None, None, None, None, None
        self.state_normalizer, self.action_normalizer, self.reward_normalizer = None, None, None
        self.next_state_normalizer = None
        self.action_normalize = action_normalize
        self.reset()

    def add(self, transition: Transition):
        """Add new transition to the buffer."""
        size = transition.shape[0]
        start = self.current_ptr
        end = self.current_ptr + size
        if end > self.max_size:
            idx_range = list(range(start, self.max_size))
            idx_range += list(range(0, end - self.max_size))
        else:
            idx_range = list(range(start, end))
        self.obs[idx_range] = transition.obs
        self.action[idx_range] = transition.action
        self.next_obs[idx_range] = transition.next_obs
        self.reward[idx_range] = transition.reward.reshape(-1, 1)
        self.done[idx_range] = transition.done.reshape(-1, 1)
        # self.obs[start:end] = transition.obs
        # self.action[start:end] = transition.action
        # self.next_obs[start:end] = transition.next_obs
        # self.reward[start:end] = transition.reward.reshape(-1, 1)
        # self.done[start:end] = transition.done.reshape(-1, 1)
        # self.obs = self.obs.at[start:end].set(transition.obs)
        # self.action = self.action.at[start:end].set(transition.action)
        # self.next_obs = self.next_obs.at[start:end].set(transition.next_obs)
        # self.reward = self.reward.at[start:end].set(transition.reward.reshape(-1, 1))
        # self.done = self.done.at[start:end].set(transition.done.reshape(-1, 1))
        self.size = min(self.size + size, self.max_size)
        if self.normalize:
            self.state_normalizer.update(self.obs[idx_range])
            if self.action_normalize:
                self.action_normalizer.update(self.action[idx_range])
            if self.learn_deltas:
                self.next_state_normalizer.update(self.next_obs[idx_range] - self.obs[idx_range])
            else:
                self.next_state_normalizer = deepcopy(self.state_normalizer)
            self.reward_normalizer.update(self.reward[idx_range])
        self.current_ptr = end % self.max_size

    def sample(self, rng: jax.random.PRNGKeyArray, batch_size: int = 256) -> Transition:
        ind = jax.random.randint(rng, (batch_size,), 0, self.size)
        obs = jnp.asarray(self.obs)[ind]
        next_state = jnp.asarray(self.next_obs)[ind]
        if self.learn_deltas:
            next_state = self.next_state_normalizer.normalize(next_state - obs)
        else:
            next_state = self.state_normalizer.normalize(next_state)

        return Transition(
            self.state_normalizer.normalize(obs),
            self.action_normalizer.normalize(jnp.asarray(self.action)[ind]),
            next_state,
            self.reward_normalizer.normalize(jnp.asarray(self.reward)[ind]),
            jnp.asarray(self.done)[ind],
        )

    def get_full_normalized_data(self) -> Transition:
        """Return all (normalized) data"""
        obs = jnp.asarray(self.obs[:self.size])
        next_state = jnp.asarray(self.next_obs[:self.size])
        if self.learn_deltas:
            next_state = self.next_state_normalizer.normalize(next_state - obs)
        else:
            next_state = self.state_normalizer.normalize(next_state)

        return Transition(
            self.state_normalizer.normalize(obs),
            self.action_normalizer.normalize(jnp.asarray(self.action[:self.size])),
            next_state,
            self.reward_normalizer.normalize(jnp.asarray(self.reward[:self.size])),
            jnp.asarray(self.done[:self.size]),
        )

    def reset(self):
        """Empty and reset replay buffer."""
        self.current_ptr = 0
        self.size = 0
        self.obs = np.zeros((self.max_size, *self.obs_shape))
        self.action = np.zeros((self.max_size, *self.action_shape))
        self.next_obs = np.zeros((self.max_size, *self.obs_shape))
        self.reward = np.zeros((self.max_size, 1))
        self.done = np.zeros((self.max_size, 1))

        self.state_normalizer = Normalizer(self.obs_shape)
        self.action_normalizer = Normalizer(self.action_shape)
        self.reward_normalizer = Normalizer((1,))
        self.next_state_normalizer = Normalizer(self.obs_shape)


@struct.dataclass
class NormalizerState:
    mean: jax.Array
    std: jax.Array
    size: int


class JaxNormalizer:
    def __init__(self, input_shape: Tuple):
        self.input_shape = input_shape

    def initialize_normalizer_state(self):
        mean = jnp.zeros(*self.input_shape)
        std = jnp.ones(*self.input_shape)
        size = 0
        return NormalizerState(
            mean=mean,
            std=std,
            size=size,
        )

    @staticmethod
    @jax.jit
    def update(x: jax.Array, state: NormalizerState) -> NormalizerState:
        new_size = x.shape[0]
        total_size = new_size + state.size
        new_mean = (state.mean * state.size + jnp.sum(x, axis=0)) / total_size
        new_s_n = jnp.square(state.std) * state.size + jnp.sum(jnp.square(x - new_mean),
                                                               axis=0
                                                               ) + state.size * jnp.square(state.mean -
                                                                                           new_mean)
        new_var = new_s_n / total_size
        new_std = jnp.sqrt(new_var)
        mean = new_mean
        std = jnp.maximum(new_std, jnp.ones_like(new_std) * EPS)
        size = total_size
        return NormalizerState(mean=mean, std=std, size=size)

    @staticmethod
    @jax.jit
    def normalize(x: chex.Array, state: NormalizerState):
        return (x - state.mean) / state.std

    @staticmethod
    @jax.jit
    def inverse(x: chex.Array, state: NormalizerState):
        return x * state.std + state.mean


@struct.dataclass
class BufferState:
    state_normalizer_state: NormalizerState
    action_normalizer_state: NormalizerState
    reward_normalizer_state: NormalizerState
    next_state_normalizer_state: NormalizerState
    tran: Transition
    size: int
    current_ptr: int


class JaxReplayBuffer(object):
    def __init__(self,
                 obs_shape: Tuple,
                 action_shape: Tuple,
                 max_size: int = int(1e6),
                 normalize: bool = False,
                 action_normalize: bool = False,
                 learn_deltas: bool = False,
                 *args,
                 **kwargs,
                 ):
        self.max_size = max_size
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.normalize = normalize
        self.learn_deltas = learn_deltas

        # self.obs, self.action, self.next_obs, self.reward, self.done = None, None, None, None, None
        self.state_normalizer, self.action_normalizer, self.reward_normalizer = None, None, None
        self.next_state_normalizer = None
        self.action_normalize = action_normalize
        _ = self.reset()

    def initialize_buffer_state(self) -> BufferState:
        current_ptr = 0
        size = 0
        obs = jnp.zeros((self.max_size, *self.obs_shape))
        action = jnp.zeros((self.max_size, *self.action_shape))
        next_obs = jnp.zeros((self.max_size, *self.obs_shape))
        reward = jnp.zeros((self.max_size, 1))
        done = jnp.zeros((self.max_size, 1))
        tran = Transition(
            obs=obs,
            action=action,
            next_obs=next_obs,
            reward=reward,
            done=done,
        )

        return BufferState(
            state_normalizer_state=self.state_normalizer.initialize_normalizer_state(),
            action_normalizer_state=self.action_normalizer.initialize_normalizer_state(),
            reward_normalizer_state=self.reward_normalizer.initialize_normalizer_state(),
            next_state_normalizer_state=self.next_state_normalizer.initialize_normalizer_state(),
            tran=tran,
            size=size,
            current_ptr=current_ptr,
        )

    @functools.partial(jax.jit, static_argnums=0)
    def add(self, transition: Transition, state: BufferState) -> BufferState:
        """Add new transition to the buffer"""
        size = transition.shape[0]
        start = state.current_ptr
        roll = jnp.minimum(0, self.max_size - start - size)
        roll_fn = lambda x: jax.lax.cond(roll, lambda: jnp.roll(x, roll, axis=0), lambda: x)
        tran = jax.tree_util.tree_map(lambda x: roll_fn(x), state.tran)
        position = start + roll
        update_fn = lambda data, update: jax.lax.dynamic_update_slice_in_dim(data, update, position, axis=0)
        new_tran = jax.tree_util.tree_map(lambda x, y: update_fn(x, y), tran, transition)
        current_ptr = (position + size) % self.max_size  # end % self.max_size
        size = jnp.minimum(state.size + size, self.max_size)

        state_normalizer_state = state.state_normalizer_state
        action_normalizer_state = state.action_normalizer_state
        reward_normalizer_state = state.reward_normalizer_state
        next_state_normalizer_state = state.next_state_normalizer_state
        if self.normalize:
            state_normalizer_state = self.state_normalizer.update(transition.obs, state.state_normalizer_state)
            if self.action_normalize:
                action_normalizer_state = self.action_normalizer.update(transition.action,
                                                                        state.action_normalizer_state)
            if self.learn_deltas:
                next_state_normalizer_state = self.next_state_normalizer.update(transition.next_obs - transition.obs,
                                                                                state.next_state_normalizer_state)
            else:
                next_state_normalizer_state = state_normalizer_state
            reward_normalizer_state = self.reward_normalizer.update(transition.reward,
                                                                    state.reward_normalizer_state)
        return BufferState(
            state_normalizer_state=state_normalizer_state,
            action_normalizer_state=action_normalizer_state,
            reward_normalizer_state=reward_normalizer_state,
            next_state_normalizer_state=next_state_normalizer_state,
            tran=new_tran,
            size=size,
            current_ptr=current_ptr,
        )

    @functools.partial(jax.jit, static_argnums=(0, 3))
    def sample(self, rng: jax.random.PRNGKeyArray, state: BufferState, batch_size: int = 256) -> Transition:
        ind = jax.random.randint(rng, (batch_size,), 0, state.size)
        sampled_tran = jax.tree_util.tree_map(lambda x: jnp.take(x, ind, axis=0, mode='wrap'), state.tran)
        if self.learn_deltas:
            next_state = sampled_tran.next_obs - sampled_tran.obs
            next_state = self.next_state_normalizer.normalize(next_state, state.next_state_normalizer_state)
        else:
            next_state = self.state_normalizer.normalize(sampled_tran.next_obs, state.state_normalizer_state)
        return Transition(
            self.state_normalizer.normalize(sampled_tran.obs, state.state_normalizer_state),
            self.action_normalizer.normalize(sampled_tran.action, state.action_normalizer_state),
            next_state,
            self.reward_normalizer.normalize(sampled_tran.reward, state.reward_normalizer_state),
            sampled_tran.done
        )

    # def get_full_normalized_data(self, state: BufferState):
    #
    #     obs = self.obs[:self.size]
    #     next_state = self.next_obs[:self.size]
    #     if self.learn_deltas:
    #         next_state = self.next_state_normalizer.normalize(next_state - obs)
    #     else:
    #         next_state = self.state_normalizer.normalize(next_state)
    #
    #     return Transition(
    #         self.state_normalizer.normalize(obs),
    #         self.action_normalizer.normalize(self.action[:self.size]),
    #         next_state,
    #         self.reward_normalizer.normalize(self.reward[:self.size]),
    #         self.done[:self.size],
    #     )

    def reset(self) -> BufferState:
        self.state_normalizer = JaxNormalizer(self.obs_shape)
        self.action_normalizer = JaxNormalizer(self.action_shape)
        self.reward_normalizer = JaxNormalizer((1,))
        self.next_state_normalizer = JaxNormalizer(self.obs_shape)
        return self.initialize_buffer_state()


if __name__ == '__main__':
    obs_shape = (2,)
    action_shape = (1,)

    buffer = JaxReplayBuffer(obs_shape, action_shape, normalize=True)
    initial_state = buffer.initialize_buffer_state()


    def step(carry, ins):
        state = carry[0]
        rng = carry[1]
        rng, sample_rng = jax.random.split(rng, 2)
        x_shape = (1,) + obs_shape
        u_shape = (1,) + action_shape
        scale_rng, sample_rng = jax.random.split(sample_rng, 2)
        tran = Transition(
            obs=jax.random.uniform(scale_rng, shape=x_shape),
            action=jnp.ones(u_shape),
            next_obs=jnp.ones(x_shape),
            reward=jnp.ones((1, 1)),
            done=jnp.ones((1, 1)),
        )
        next_state = buffer.add(tran, state)
        sample = buffer.sample(rng=sample_rng, state=next_state, batch_size=10)
        carry = [next_state, rng]
        out = []
        jax.debug.print('sample = {x}', x=sample.obs)
        jax.debug.print('size = {x}', x=next_state.size)
        jax.debug.print('obs_std = {x}', x=next_state.state_normalizer_state.std)
        jax.debug.print('obs_mean = {x}', x=next_state.state_normalizer_state.mean)
        return carry, out


    carry = [initial_state, jax.random.PRNGKey(0)]
    carry, outs = jax.lax.scan(step, carry, xs=None, length=10)
