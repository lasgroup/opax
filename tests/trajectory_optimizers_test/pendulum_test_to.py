import numpy as np

from mbse.models.environment_models.pendulum_swing_up import CustomPendulumEnv, PendulumReward, PendulumDynamicsModel
from mbse.optimizers.cem_trajectory_optimizer import CemTO
from mbse.optimizers.trajax_trajectory_optimizer import TraJaxTO, ILQRHyperparams
import time
from mbse.utils.vec_env.env_util import make_vec_env
from gym.wrappers.time_limit import TimeLimit
from gym.wrappers.rescale_action import RescaleAction

from jax.config import config
config.update("jax_log_compiles", 1)

wrapper_cls = lambda x: RescaleAction(
    TimeLimit(x, max_episode_steps=200),
    min_action=-1,
    max_action=1,
)
env = wrapper_cls(CustomPendulumEnv(render_mode='human'))
true_dynamics = PendulumDynamicsModel(env)
dynamics_model_list = [true_dynamics]

horizon = 20

obs, _ = env.reset()
policy_optimizer = TraJaxTO(
    dynamics_model_list=dynamics_model_list,
    horizon=horizon,
    action_dim=(1,),
    params=ILQRHyperparams(maxiter=100),
)

time_stamps = []
for i in range(200):
    start_time = time.time()
    action_sequence, returns = policy_optimizer.optimize(obs=obs.reshape(1, -1), dynamics_params=None)
    time_taken = time.time() - start_time
    if i == 0:
        print("Time taken", time_taken)
    else:
        time_stamps.append(time_taken)
    action = np.asarray((action_sequence[0, 0, :]))
    obs, reward, terminate, truncate, info = env.step(action)
    env.render()

time_stamps = np.asarray(time_taken)
print("avergage time taken", time_stamps.mean())
time_stamps = []
obs, _ = env.reset()
for i in range(200):
    start_time = time.time()
    action_sequence, returns = policy_optimizer.optimize(obs=obs.reshape(1, -1), dynamics_params=None)
    time_taken = time.time() - start_time
    if i == 0:
        print("Time taken", time_taken)
    else:
        time_stamps.append(time_taken)
    action = np.asarray((action_sequence[0, 0, :]))
    obs, reward, terminate, truncate, info = env.step(action)
    env.render()

time_stamps = np.asarray(time_taken)
print("avergage time taken", time_stamps.mean())

policy_optimizer = CemTO(
    dynamics_model_list=dynamics_model_list,
    horizon=horizon,
    action_dim=(1,),
    upper_bound=1,
)
obs, _ = env.reset()
time_stamps = []
for i in range(200):
    start_time = time.time()
    action_sequence, returns = policy_optimizer.optimize(obs=obs.reshape(1, -1), dynamics_params=None)
    time_taken = time.time() - start_time
    if i == 0:
        print("Time taken", time_taken)
    else:
        time_stamps.append(time_taken)
    action = np.asarray((action_sequence[0, 0, :]))
    obs, reward, terminate, truncate, info = env.step(action)
    env.render()

time_stamps = np.asarray(time_taken)
print("avergage time taken", time_stamps.mean())
policy_optimizer.reset()
obs, _ = env.reset()
time_stamps = []
for i in range(200):
    start_time = time.time()
    action_sequence, returns = policy_optimizer.optimize(obs=obs.reshape(1, -1), dynamics_params=None)
    time_taken = time.time() - start_time
    if i == 0:
        print("Time taken", time_taken)
    else:
        time_stamps.append(time_taken)
    action = np.asarray((action_sequence[0, 0, :]))
    obs, reward, terminate, truncate, info = env.step(action)
    env.render()

time_stamps = np.asarray(time_taken)
print("avergage time taken", time_stamps.mean())
# start_time = time.time()
# action_sequence, returns = policy_optimizer.optimize(obs=initial_state, dynamics_params=None,
#                                                      initial_actions=initial_actions)
# print("Time taken: ", time.time() - start_time)
