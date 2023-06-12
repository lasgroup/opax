import jax
import jax.numpy as jnp
from mbse.optimizers.cross_entropy_optimizer import CrossEntropyOptimizer
from mbse.utils.utils import sample_trajectories
import functools
from typing import Optional, Union
from mbse.optimizers.dummy_policy_optimizer import DummyPolicyOptimizer, BestSequences
from mbse.models.dynamics_model import ModelProperties


class CemTO(DummyPolicyOptimizer):
    def __init__(self,
                 horizon: int,
                 action_dim: tuple,
                 dynamics_model_list: list,
                 n_particles: int = 10,
                 cem_kwargs=None,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        cem_action_dim = (horizon,) + action_dim
        self.horizon = horizon
        self.n_particles = n_particles
        if cem_kwargs is None:
            cem_kwargs = {}
        self.optimizer = CrossEntropyOptimizer(action_dim=cem_action_dim, **cem_kwargs)
        assert isinstance(dynamics_model_list, list)
        self.dynamics_model_list = dynamics_model_list
        self.best_sequences = BestSequences(
            evaluation_sequences=jnp.zeros((len(self.dynamics_model_list),) + cem_action_dim),
            exploration_sequence=jnp.zeros(cem_action_dim),
        )
        self._init_fn()

    def _init_fn(self):

        self.optimize_for_eval_fns = []
        for i in range(len(self.dynamics_model_list)):
            self.optimize_for_eval_fns.append(functools.partial(
                self._get_action_sequence, model_index=i
            ))
        self.optimize = self.optimize_for_eval_fns[0]

    def optimize_for_exploration(
            self,
            dynamics_params,
            obs,
            key=None,
            optimizer_key=None,
            model_props: ModelProperties = ModelProperties(),
            initial_actions: Optional[jax.Array] = None,
            sampling_idx: Optional[Union[jnp.ndarray, int]] = None,
    ):
        if initial_actions is None:
            prev_best = jnp.zeros_like(self.best_sequences.exploration_sequence)
            prev_best = prev_best.at[:-1].set(self.best_sequences.exploration_sequence[1:])
            last_input = self.best_sequences.exploration_sequence[-1]
            prev_best = prev_best.at[-1].set(last_input)
            initial_actions = prev_best

        def optimize(init_state, rng, opt_rng):
            return self._optimize_action_sequence(
                eval_fn=self.dynamics_model.evaluate_for_exploration,
                optimize_fn=self.optimizer.optimize,
                n_particles=self.n_particles,
                horizon=self.horizon,
                params=dynamics_params,
                init_state=init_state,
                key=rng,
                optimizer_key=opt_rng,
                model_props=model_props,
                init_action_seq=initial_actions,
                sampling_idx=sampling_idx,
            )
        best_sequence, best_reward = jax.vmap(optimize)(obs, key, optimizer_key)
        self.best_sequences = BestSequences(
            evaluation_sequences=self.best_sequences.evaluation_sequences,
            exploration_sequence=best_sequence[0],
        )
        return best_sequence, best_reward

    def _get_action_sequence(
            self,
            model_index,
            dynamics_params,
            obs,
            key=None,
            optimizer_key=None,
            model_props: ModelProperties = ModelProperties(),
            initial_actions: Optional[jax.Array] = None,
            sampling_idx: Optional[Union[jnp.ndarray, int]] = None,
    ):
        if initial_actions is None:
            prev_best = jnp.zeros_like(self.best_sequences.evaluation_sequences[model_index])
            prev_best = prev_best.at[:-1].set(self.best_sequences.evaluation_sequences[model_index][1:])
            last_input = self.best_sequences.evaluation_sequences[model_index][-1]
            prev_best = prev_best.at[-1].set(last_input)
            initial_actions = prev_best

        def optimize(init_state, rng, opt_rng):
            return self._optimize_action_sequence(
                eval_fn=self.dynamics_model_list[model_index].evaluate,
                optimize_fn=self.optimizer.optimize,
                n_particles=self.n_particles,
                horizon=self.horizon,
                params=dynamics_params,
                init_state=init_state,
                key=rng,
                optimizer_key=opt_rng,
                model_props=model_props,
                sampling_idx=sampling_idx,
                init_action_seq=initial_actions,
            )

        best_sequence, best_reward = jax.vmap(optimize)(obs, key, optimizer_key)
        sequence_copy = self.best_sequences.evaluation_sequences.at[model_index].set(best_sequence[0])
        self.best_sequences = BestSequences(
            evaluation_sequences=sequence_copy,
            exploration_sequence=self.best_sequences.exploration_sequence,
        )
        return best_sequence, best_reward

    @staticmethod
    @functools.partial(jax.jit, static_argnums=(0, 1, 2, 3))
    def _optimize_action_sequence(eval_fn,
                                  optimize_fn,
                                  n_particles,
                                  horizon,
                                  params,
                                  init_state,
                                  key,
                                  optimizer_key,
                                  model_props: ModelProperties = ModelProperties(),
                                  init_action_seq: Optional[jax.Array] = None,
                                  sampling_idx: Optional[Union[jnp.ndarray, int]] = None,
                                  ):
        init_state = jnp.repeat(jnp.expand_dims(init_state, 0), n_particles, 0)
        eval_func = lambda seq, x, k: sample_trajectories(
            evaluate_fn=eval_fn,
            parameters=params,
            init_state=x,
            horizon=horizon,
            key=k,
            actions=seq,
            model_props=model_props,
            sampling_idx=sampling_idx,
        )

        def sum_rewards(seq):
            # seq = jnp.repeat(jnp.expand_dims(seq, 0), n_particles, 0)

            def get_average_reward(obs, eval_key):
                transition = eval_func(seq, obs, eval_key)
                return transition.reward.mean()

            if key is not None:
                rollout_key = jax.random.split(key=key, num=n_particles)
                returns = jax.vmap(get_average_reward)(init_state, rollout_key)
            else:
                returns = jax.vmap(get_average_reward, in_axes=(0, None))(init_state, key)
            return returns.mean()

        action_seq, reward = optimize_fn(
            func=sum_rewards,
            rng=optimizer_key,
            mean=init_action_seq,
        )
        return action_seq, reward

    @property
    def dynamics_model(self):
        return self.dynamics_model_list[0]

    def reset(self):
        self.best_sequences = BestSequences(
            evaluation_sequences=jnp.zeros_like(self.best_sequences.evaluation_sequences),
            exploration_sequence=jnp.zeros_like(self.best_sequences.exploration_sequence)
        )
