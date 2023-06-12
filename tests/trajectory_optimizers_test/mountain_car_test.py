import numpy as np

from mbse.models.environment_models.mountain_car import MountainCarDynamics
from mbse.envs.custom_mountain_car_env import CustomMountainCar
from mbse.utils.replay_buffer import ReplayBuffer, Transition
from mbse.optimizers.sac_based_optimizer import SACOptimizer
import time
from gym.wrappers.time_limit import TimeLimit
from gym.wrappers.rescale_action import RescaleAction
import jax
from mbse.utils.vec_env.env_util import make_vec_env


def rollout_random_policy(env, num_steps, rng):
    rng, reset_rng = jax.random.split(rng, 2)
    reset_seed = jax.random.randint(
        reset_rng,
        (1,),
        minval=0,
        maxval=num_steps).item()
    obs, _ = env.reset(seed=reset_seed)
    num_points = num_steps
    obs_shape = (num_points,) + env.observation_space.shape
    action_space = (num_points,) + env.action_space.shape
    obs_vec = np.zeros(obs_shape)
    action_vec = np.zeros(action_space)
    reward_vec = np.zeros((num_points,))
    next_obs_vec = np.zeros(obs_shape)
    done_vec = np.zeros((num_points,))
    next_rng = rng
    for step in range(num_steps):
        next_rng, actor_rng = jax.random.split(next_rng, 2)
        action = env.action_space.sample()
        next_obs, reward, terminate, truncate, info = env.step(action)
        obs = env.envs[0].env.env.sample_obs()

        obs_vec[step: (step + 1)] = obs
        action_vec[step: (step + 1)] = action
        reward_vec[step: (step + 1)] = reward.reshape(-1)
        next_obs_vec[step: (step + 1)] = next_obs
        done_vec[step: (step + 1)] = terminate.reshape(-1)
        # obs_vec = obs_vec.at[step].set(jnp.asarray(obs))
        # action_vec = action_vec.at[step].set(jnp.asarray(action))
        # reward_vec = reward_vec.at[step].set(jnp.asarray(reward))
        # next_obs_vec = next_obs_vec.at[step].set(jnp.asarray(next_obs))
        # done_vec = done_vec.at[step].set(jnp.asarray(terminate))
        # obs = np.concatenate([x['current_env_state'].reshape(1, -1) for x in info], axis=0)
        # for idx, done in enumerate(dones):
        #    if done:
        #        reset_rng, next_reset_rng = jax.random.split(reset_rng, 2)
        #        reset_seed = jax.random.randint(
        #            reset_rng,
        #            (1,),
        #            minval=0,
        #            maxval=num_steps).item()
        #        obs[idx], _ = self.env.reset(seed=reset_seed)

    transitions = Transition(
        obs=obs_vec,
        action=action_vec,
        reward=reward_vec,
        next_obs=next_obs_vec,
        done=done_vec,
    )
    reset_seed = jax.random.randint(
        reset_rng,
        (1,),
        minval=0,
        maxval=num_steps).item()
    obs, _ = env.reset(seed=reset_seed)
    return transitions


wrapper_cls = lambda x: RescaleAction(
    TimeLimit(x, max_episode_steps=1000),
    min_action=-1,
    max_action=1,
)
env = wrapper_cls(CustomMountainCar(render_mode='human'))
true_dynamics = MountainCarDynamics()
dynamics_model_list = [true_dynamics]

horizon = 20

obs, _ = env.reset()
sac_kwargs = {
    'discount': 0.99,
    'init_ent_coef': 1.0,
    'lr_actor': 0.01,
    'weight_decay_actor': 1e-5,
    'lr_critic': 0.01,
    'weight_decay_critic': 1e-5,
    'lr_alpha': 0.001,
    'weight_decay_alpha': 0.0,
    'actor_features': [64, 64],
    'critic_features': [64, 64],
    'scale_reward': 1,
    'tune_entropy_coef': True,
    'tau': 0.01,
    'batch_size': 256,
    'train_steps': 300,
}

policy_optimizer = SACOptimizer(
    dynamics_model_list=dynamics_model_list,
    horizon=horizon,
    action_dim=(1,),
    train_steps_per_model_update=10,
    transitions_per_update=500,
    sac_kwargs=sac_kwargs,
    reset_actor_params=False,
)

buffer = ReplayBuffer(
    obs_shape=env.observation_space.shape,
    action_shape=env.action_space.shape,
    max_size=100000,
    normalize=False,
    action_normalize=False,
    learn_deltas=False
)

rng = jax.random.PRNGKey(seed=0)
rollout_rng, rng = jax.random.split(rng, 2)
transitions = rollout_random_policy(env=make_vec_env(CustomMountainCar, wrapper_class=wrapper_cls),
                                    num_steps=10000, rng=rollout_rng)

buffer.add(transitions)
train_rng, rng = jax.random.split(rng, 2)
t = time.time()
train_summary = policy_optimizer.train(
    rng=train_rng,
    buffer=buffer,
)
print("Time take to train in [s]", time.time() - t)
obs, _ = env.reset()
for i in range(200):
    start_time = time.time()
    action = policy_optimizer.get_action(obs=obs, rng=rng)
    obs, reward, terminate, truncate, info = env.step(action)
t = time.time()
train_rng, rng = jax.random.split(rng, 2)
train_summary2 = policy_optimizer.train(
    rng=train_rng,
    buffer=buffer,
)
print("Time take to train in [s]", time.time() - t)
obs, _ = env.reset()
for i in range(200):
    start_time = time.time()
    action = policy_optimizer.get_action(obs=obs, rng=rng)
    obs, reward, terminate, truncate, info = env.step(action)
