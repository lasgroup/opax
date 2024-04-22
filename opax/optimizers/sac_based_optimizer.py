import copy

import jax.random

from opax.agents.actor_critic.sac import SACAgent, SACTrainingState, soft_update
from gym.spaces import Box
from opax.utils.replay_buffer import Transition, ReplayBuffer, JaxReplayBuffer, BufferState
from typing import Callable, Union, Optional, Any
import jax.numpy as jnp
from opax.utils.utils import sample_trajectories, get_idx, tree_stack, convert_to_jax
import functools
from opax.optimizers.dummy_policy_optimizer import DummyPolicyOptimizer
from opax.utils.type_aliases import ModelProperties, PolicyProperties
from opax.models.active_learning_model import ActiveLearningHUCRLModel, ActiveLearningPETSModel
import flax.struct

EPS = 1e-6


@functools.partial(
    jax.jit, static_argnums=(0, 2, 4)
)
def _simulate_dynamics(horizon: int,
                       obs: jax.Array,
                       policy: Callable,
                       actor_params,
                       evaluate_fn: Callable,
                       dynamics_params=None,
                       key=None,
                       model_props: ModelProperties = ModelProperties(),
                       sampling_idx: Optional[Union[jnp.ndarray, int]] = None,
                       policy_props: PolicyProperties = PolicyProperties(),
                       ):
    def sample_trajectories_for_state(state: jax.Array, sample_key):
        return sample_trajectories(
            evaluate_fn=evaluate_fn,
            parameters=dynamics_params,
            init_state=state,
            horizon=horizon,
            key=sample_key,
            policy=policy,
            actor_params=actor_params,
            model_props=model_props,
            sampling_idx=sampling_idx,
            policy_props=policy_props,
        )

    key = jax.random.split(key, obs.shape[0])
    transitions = jax.vmap(sample_trajectories_for_state)(obs, key)

    def flatten(arr):
        new_arr = arr.reshape(-1, arr.shape[-1])
        return new_arr

    transitions = Transition(
        obs=flatten(transitions.obs),
        action=flatten(transitions.action),
        reward=flatten(transitions.reward),
        next_obs=flatten(transitions.next_obs),
        done=flatten(transitions.done),
    )
    return transitions


@flax.struct.dataclass
class SacOptimizerState:
    agent_train_state: SACTrainingState
    policy_props: PolicyProperties


class SACOptimizer(DummyPolicyOptimizer):
    def __init__(self,
                 action_dim: tuple,
                 dynamics_model_list: list,
                 horizon: int = 20,
                 n_particles: int = 10,
                 transitions_per_update: int = 10,
                 simulated_buffer_size: int = 1000000,
                 train_steps_per_model_update: int = 20,
                 sim_transitions_ratio: float = 0.0,
                 normalize: bool = False,
                 action_normalize: bool = False,
                 sac_kwargs: Optional[dict] = None,
                 reset_actor_params: bool = False,
                 reset_optimizer: bool = True,
                 reset_buffer: bool = True,
                 target_soft_update_tau: float = 0.05,
                 evaluation_frequency: int = -1,
                 evaluation_samples: int = 100,
                 evaluation_horizon: int = 100,
                 use_best_trained_policy: bool = False,
                 *args,
                 **kwargs,
                 ):
        """

        :param action_dim:
        :param dynamics_model_list:
        :param horizon:
        :param n_particles:
        :param transitions_per_update: Number of initial states to sample per sac update
        :param simulated_buffer_size: max replay buffer size of simulated buffer
        :param train_steps_per_model_update: Number of total trainiting steps per model update.
        :param sim_transitions_ratio:
        :param normalize:
        :param action_normalize:
        :param sac_kwargs: sac kwargs -> stores sac training parameters, e.g., number of gradient steps per update.
        :param reset_actor_params: boolean to indicate if actor should reset after every model update.
        :param reset_optimizer: boolean to indicate if optimizer should reset after every model update.
        :param reset_buffer: boolean to indicate if buffer should reset after every model update.
        :param target_soft_update_tau: float -> soft update for tracking policy params over multiple model updates.
        :param evaluation_frequency:
        :param evaluation_samples:
        :param evaluation_horizon:
        :param use_best_trained_policy:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        assert isinstance(dynamics_model_list, list)
        self.dynamics_model_list = dynamics_model_list
        obs_dim = self.dynamics_model.obs_dim
        self.active_exploration_agent = False
        if isinstance(self.dynamics_model, ActiveLearningPETSModel) or isinstance(
                self.dynamics_model, ActiveLearningHUCRLModel):
            self.dynamics_model_list.append(dynamics_model_list[0])
            self.active_exploration_agent = True
        dummy_obs_space = Box(low=-1, high=1, shape=(obs_dim,))
        dummy_act_space = Box(low=-1, high=1, shape=action_dim)
        if sac_kwargs is not None:
            self.agent_list = [SACAgent(
                action_space=dummy_act_space,
                observation_space=dummy_obs_space,
                **sac_kwargs,
            ) for model in self.dynamics_model_list]
        else:
            self.agent_list = [SACAgent(
                action_space=dummy_act_space,
                observation_space=dummy_obs_space,
            ) for model in self.dynamics_model_list]

        init_optimizer_state = [SacOptimizerState(
            agent_train_state=agent.training_state,
            policy_props=agent.policy_props,

        ) for agent in self.agent_list]
        self.init_optimizer_state = tree_stack(init_optimizer_state)
        self.optimizer_state = copy.deepcopy(self.init_optimizer_state)
        self.target_optimizer_state = copy.deepcopy(self.optimizer_state)
        self.target_soft_update_tau = target_soft_update_tau
        self.horizon = horizon
        self.n_particles = n_particles
        self.transitions_per_update = transitions_per_update
        self.simulated_buffer_size = simulated_buffer_size
        self.normalize = normalize
        self.action_normalize = action_normalize
        self.obs_dim = (obs_dim,)
        self.action_dim = action_dim
        self.train_steps_per_model_update = train_steps_per_model_update
        self.sim_transitions_ratio = sim_transitions_ratio
        self.reset_actor_params = reset_actor_params
        self.sim_buffer_kwargs = {
            'obs_shape': self.obs_dim,
            'action_shape': self.action_dim,
            'max_size': self.simulated_buffer_size,
            'normalize': self.normalize,
            'action_normalize': self.action_normalize,
        }

        self.simulation_buffers = [JaxReplayBuffer(
            learn_deltas=False,
            **self.sim_buffer_kwargs,
        ) for _ in self.agent_list]

        self.buffer_states = [buffer.initialize_buffer_state() for buffer in self.simulation_buffers]
        self.reset_optimizer = reset_optimizer
        self.reset_buffer = reset_buffer
        self.evaluate_agent = evaluation_frequency > 0
        self.evaluation_samples = evaluation_samples
        self.evaluation_horizon = evaluation_horizon
        self.evaluation_frequency = evaluation_frequency
        self.use_best_trained_policy = use_best_trained_policy
        self._init_fn()

    def get_action_for_eval(self, obs: jax.Array, rng, agent_idx: int):
        policy = self.agent_list[0].get_eval_action
        agent_state = get_idx(self.optimizer_state, agent_idx)
        normalized_obs = (obs - agent_state.policy_props.policy_bias_obs) / (agent_state.policy_props.policy_scale_obs
                                                                             + EPS)
        action = policy(
            actor_params=agent_state.agent_train_state.actor_params,
            obs=normalized_obs,
            rng=rng,
        )
        return action

    def get_action(self, obs: jax.Array, rng):
        return self.get_action_for_eval(obs=obs, rng=rng, agent_idx=0)

    def get_action_for_exploration(self, obs: jax.Array, rng: jax.random.PRNGKeyArray, *args, **kwargs):
        policy = self.agent_list[0].get_action
        if self.active_exploration_agent:
            agent_state = get_idx(self.optimizer_state, -1)
        else:
            agent_state = get_idx(self.optimizer_state, 0)
        normalized_obs = (obs - agent_state.policy_props.policy_bias_obs) / (
                agent_state.policy_props.policy_scale_obs + EPS)
        action = policy(
            actor_params=agent_state.agent_train_state.actor_params,
            obs=normalized_obs,
            rng=rng,
        )
        return action

    def _init_fn(self):

        def train_agent_step(
                train_rng,
                train_state,
                sim_transitions,
        ):
            return self.train_agent_step(
                train_rng=train_rng,
                train_state=train_state,
                sim_transitions=sim_transitions,
                agent_train_fn=self.agent_list[0].step,
                agent_train_steps=self.agent_list[0].train_steps,
            )

        # self.train_step = jax.jit(jax.vmap(train_agent_step))
        self.train_step = jax.jit(train_agent_step)

    @staticmethod
    def train_agent_step(train_rng,
                         train_state,
                         sim_transitions,
                         agent_train_fn,
                         agent_train_steps,
                         ):
        carry = [
            train_rng,
            train_state
        ]
        ins = [sim_transitions]
        carry, outs = jax.lax.scan(agent_train_fn, carry, ins, length=agent_train_steps)
        next_train_state = carry[1]
        summary = get_idx(outs[-1], -1)
        return next_train_state, summary

    @functools.partial(jax.jit, static_argnums=(0, 4, 6, 7, 9, 10, 11))
    def train_single_agent(self,
                           rng: jax.random.PRNGKeyArray,
                           true_obs: jax.Array,
                           obs_size: jax.Array,
                           sim_buffer: JaxReplayBuffer,
                           simulation_buffer_state: BufferState,
                           train_steps: int,
                           batch_size: int,
                           optimizer_state: SACTrainingState,
                           policy: Callable,
                           evaluate: Callable,
                           eval_policy: Callable,
                           dynamics_params: Optional = None,
                           model_props: ModelProperties = ModelProperties(),
                           sampling_idx: Optional[Union[jnp.ndarray, int]] = None,
                           ):

        eval_rng, rng = jax.random.split(rng, 2)
        eval_idx = jax.random.randint(eval_rng, (self.evaluation_samples,), 0, obs_size)
        eval_obs = jnp.take(true_obs, eval_idx, axis=0, mode='wrap')
        eval_sim_key, rng = jax.random.split(rng, 2)

        def step(carry, ins):
            # sample initial obs from buffer
            sim_buffer_state, opt_state, prev_reward, key = carry[0], carry[1], carry[2], carry[-1]
            best_reward, best_opt_state = carry[3], carry[4]
            # batch_sim_buffer = int(self.sim_transitions_ratio * self.transitions_per_update) * \
            #                   (sim_buffer_state.size > 0)
            # batch_true_buffer = int(self.transitions_per_update - batch_sim_buffer)
            buffer_rng, key = jax.random.split(key, 2)
            # if batch_sim_buffer > 0:
            #    true_buffer_rng, sim_buffer_rng = jax.random.split(buffer_rng, 2)
            #    ind = jax.random.randint(true_buffer_rng, (batch_true_buffer,), 0, true_obs.shape[0])
            #    true_obs_sample = true_obs[ind]
            #    sim_trans = sim_buffer.sample(rng=sim_buffer_rng, state=sim_buffer_state, batch_size=batch_sim_buffer)
            #    sim_obs_sample = sim_trans.obs * sim_buffer_state.state_normalizer_state.std + \
            #                     sim_buffer_state.state_normalizer_state.mean
            #    obs = jnp.concatenate([true_obs_sample, sim_obs_sample], axis=0)
            # else:

            # sample transitions_per_update different initial states from the buffer
            # and simulate them for a fixed horizon.
            ind = jax.random.randint(buffer_rng, (self.transitions_per_update,), 0, obs_size)
            obs = jnp.take(true_obs, ind, axis=0, mode='wrap')

            simulation_key, key = jax.random.split(key, 2)
            simulated_transitions = _simulate_dynamics(
                obs=obs,
                policy=policy,
                actor_params=opt_state.agent_train_state.actor_params,
                evaluate_fn=evaluate,
                dynamics_params=dynamics_params,
                key=simulation_key,
                model_props=model_props,
                sampling_idx=sampling_idx,
                horizon=self.horizon,
                policy_props=opt_state.policy_props,
            )
            new_sim_buffer_state = sim_buffer.add(
                transition=simulated_transitions,
                state=sim_buffer_state
            )
            new_policy_props = PolicyProperties(
                policy_bias_obs=new_sim_buffer_state.state_normalizer_state.mean,
                policy_scale_obs=new_sim_buffer_state.state_normalizer_state.std,
            )

            sim_buffer_rng, key = jax.random.split(key, 2)
            sim_transitions = sim_buffer.sample(rng=sim_buffer_rng,
                                                state=new_sim_buffer_state,
                                                batch_size=int(train_steps * batch_size)
                                                )
            sim_transitions = sim_transitions.reshape(train_steps, batch_size)
            train_rng, key = jax.random.split(key, 2)
            # sample transitions from the sim transitions buffer and perform sac updates.
            agent_train_state, summary = self.train_step(
                train_rng=train_rng,
                train_state=opt_state.agent_train_state,
                sim_transitions=sim_transitions,
            )

            new_opt_state = SacOptimizerState(
                agent_train_state=agent_train_state,
                policy_props=new_policy_props,
            )

            if self.evaluate_agent:
                def evaluate_policy():
                    simulated_transitions = _simulate_dynamics(
                        obs=eval_obs,
                        policy=eval_policy,
                        actor_params=new_opt_state.agent_train_state.actor_params,
                        evaluate_fn=evaluate,
                        dynamics_params=dynamics_params,
                        key=eval_sim_key,
                        model_props=model_props,
                        sampling_idx=sampling_idx,
                        horizon=self.evaluation_horizon,
                        policy_props=new_opt_state.policy_props,
                    )
                    reward = simulated_transitions.reward.reshape(
                        self.evaluation_samples, self.evaluation_horizon
                    ).mean()

                    def get_new_reward():
                        return reward, new_opt_state,

                    def get_prev_best_reward():
                        return best_reward, best_opt_state

                    new_best_reward, new_best_opt_state = jax.lax.cond(
                        reward > best_reward,
                        get_new_reward,
                        get_prev_best_reward,
                    )
                    return reward, new_best_reward, new_best_opt_state

                def skip_evaluation():
                    return prev_reward, best_reward, best_opt_state

                reward, new_best_reward, new_best_opt_state = \
                    jax.lax.cond(jnp.logical_or(ins % self.evaluation_frequency == 0,
                                                ins == self.train_steps_per_model_update - 1),
                                 evaluate_policy,
                                 skip_evaluation
                                 )
            else:
                reward = prev_reward
                new_best_reward, new_best_opt_state = reward, new_opt_state
            outs = [summary, reward]
            carry = [new_sim_buffer_state, new_opt_state, reward, new_best_reward, new_best_opt_state, key]
            return carry, outs

        xs = jnp.arange(self.train_steps_per_model_update)
        carry = [simulation_buffer_state, optimizer_state, 0.0, -jnp.inf, optimizer_state, rng]
        carry, outs = jax.lax.scan(step, carry, xs=xs, length=self.train_steps_per_model_update)
        if self.use_best_trained_policy:
            trained_state = carry[-2]
        else:
            trained_state = carry[1]
        return carry[0], trained_state, outs[0], outs[1]

    def train(self,
              rng: jax.random.PRNGKeyArray,
              buffer: ReplayBuffer,
              dynamics_params: Optional = None,
              model_props: ModelProperties = ModelProperties(),
              sampling_idx: Optional[Union[jnp.ndarray, int]] = None,
              reset_agent: bool = True
              ):
        # simulation_buffers = [JaxReplayBuffer(
        #    learn_deltas=False,
        #    **sim_buffer_kwargs,
        # ) for _ in self.agent_list]
        # true_obs = jnp.asarray(buffer.obs[:buffer.size])
        train_steps = self.agent_list[0].train_steps
        batch_size = self.agent_list[0].batch_size
        full_optimizer_state = self.init_optimizer_state

        # If previous agent parameters should be used
        if not self.reset_actor_params and not reset_agent:
            # if optimizer should be reset use init optimizer params.
            if self.reset_optimizer:
                full_optimizer_state = SacOptimizerState(
                    agent_train_state=SACTrainingState(
                        actor_opt_state=self.init_optimizer_state.agent_train_state.actor_opt_state,
                        actor_params=self.target_optimizer_state.agent_train_state.actor_params,
                        critic_opt_state=self.init_optimizer_state.agent_train_state.critic_opt_state,
                        critic_params=self.target_optimizer_state.agent_train_state.critic_params,
                        target_critic_params=self.target_optimizer_state.agent_train_state.target_critic_params,
                        alpha_opt_state=self.init_optimizer_state.agent_train_state.alpha_opt_state,
                        alpha_params=self.target_optimizer_state.agent_train_state.alpha_params,
                    ),
                    policy_props=self.target_optimizer_state.policy_props,
                )
            # use params from optimizer
            else:
                full_optimizer_state = SacOptimizerState(
                    agent_train_state=SACTrainingState(
                        actor_opt_state=self.target_optimizer_state.agent_train_state.actor_opt_state,
                        actor_params=self.target_optimizer_state.agent_train_state.actor_params,
                        critic_opt_state=self.target_optimizer_state.agent_train_state.critic_opt_state,
                        critic_params=self.target_optimizer_state.agent_train_state.critic_params,
                        target_critic_params=self.target_optimizer_state.agent_train_state.target_critic_params,
                        alpha_opt_state=self.target_optimizer_state.agent_train_state.alpha_opt_state,
                        alpha_params=self.target_optimizer_state.agent_train_state.alpha_params,
                    ),
                    policy_props=self.target_optimizer_state.policy_props,
                )
            # if this is an active exploration agent. Reset the last agent by default
            if self.active_exploration_agent:
                active_exploration_state = get_idx(self.init_optimizer_state, -1)
                full_optimizer_state = \
                    jax.tree_util.tree_map(lambda x, y: x.at[-1].set(y),
                                           full_optimizer_state, active_exploration_state)
                self.buffer_states[-1] = self.simulation_buffers[-1].reset()

            # if replay buffer should be reset.
            if self.reset_buffer:
                self.buffer_states = [buffer.reset() for buffer in self.simulation_buffers]
        else:
            self.buffer_states = [buffer.reset() for buffer in self.simulation_buffers]

        agent_summary = []
        agent_rewards = []
        policy = self.agent_list[0].get_action
        eval_policy = self.agent_list[0].get_eval_action
        # for i in range(self.train_steps_per_model_update):
        # agents_policy_props = []
        trained_optimizer_state = []
        # transitions_list = []
        for j in range(len(self.agent_list)):
            sim_buffer = self.simulation_buffers[j]
            sim_buffer_state = self.buffer_states[j]
            evaluate_fn = self.dynamics_model_list[j].evaluate
            optimizer_state = get_idx(full_optimizer_state, idx=j)
            if self.is_active_exploration_agent(idx=j):
                evaluate_fn = self.dynamics_model_list[j].evaluate_for_exploration
            simulation_buffer_state, optimizer_state, summary, reward = self.train_single_agent(
                rng=rng,
                true_obs=buffer.obs,
                obs_size=buffer.size,
                sim_buffer=sim_buffer,
                simulation_buffer_state=sim_buffer_state,
                train_steps=train_steps,
                batch_size=batch_size,
                optimizer_state=optimizer_state,
                policy=policy,
                evaluate=evaluate_fn,
                eval_policy=eval_policy,
                dynamics_params=dynamics_params,
                model_props=model_props,
                sampling_idx=sampling_idx,
            )
            agent_summary.append(summary)
            agent_rewards.append(reward)
            trained_optimizer_state.append(optimizer_state)
            self.buffer_states[j] = simulation_buffer_state
        full_optimizer_state = tree_stack(trained_optimizer_state)
        self.optimizer_state = full_optimizer_state

        soft_actor_params = soft_update(target_params=self.target_optimizer_state.agent_train_state.actor_params,
                                        online_params=self.optimizer_state.agent_train_state.actor_params,
                                        tau=self.target_soft_update_tau)

        soft_critic_params = soft_update(target_params=self.target_optimizer_state.agent_train_state.critic_params,
                                         online_params=self.optimizer_state.agent_train_state.critic_params,
                                         tau=self.target_soft_update_tau)

        soft_target_critic_params = soft_update(
            target_params=self.target_optimizer_state.agent_train_state.target_critic_params,
            online_params=self.optimizer_state.agent_train_state.target_critic_params,
            tau=self.target_soft_update_tau)

        soft_alpha_params = soft_update(target_params=self.target_optimizer_state.agent_train_state.alpha_params,
                                        online_params=self.optimizer_state.agent_train_state.alpha_params,
                                        tau=self.target_soft_update_tau)

        self.target_optimizer_state = SacOptimizerState(
            agent_train_state=SACTrainingState(
                actor_opt_state=self.optimizer_state.agent_train_state.actor_opt_state,
                actor_params=soft_actor_params,
                critic_opt_state=self.optimizer_state.agent_train_state.critic_opt_state,
                critic_params=soft_critic_params,
                target_critic_params=soft_target_critic_params,
                alpha_opt_state=self.optimizer_state.agent_train_state.alpha_opt_state,
                alpha_params=soft_alpha_params,
            ),
            policy_props=self.optimizer_state.policy_props)
        return agent_summary, agent_rewards

    @property
    def dynamics_model(self):
        return self.dynamics_model_list[0]

    def is_active_exploration_agent(self, idx):
        return idx == len(self.agent_list) - 1 and self.active_exploration_agent
