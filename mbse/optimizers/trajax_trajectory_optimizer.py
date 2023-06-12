from trajax.optimizers import ILQR, ILQRHyperparams
from typing import Union, Callable, Optional, Any
import jax
import jax.numpy as jnp
import functools
from mbse.optimizers.dummy_policy_optimizer import DummyPolicyOptimizer, BestSequences
from mbse.utils.type_aliases import ModelProperties


@functools.partial(jax.jit, static_argnums=(0, 1, 2))
def _optimize_with_params(reward_fn: Callable,
                          dynamics_fn: Callable,
                          horizon: int,
                          initial_state: jax.Array,
                          initial_actions: jax.Array,
                          optimizer_params: ILQRHyperparams,
                          dynamics_params,
                          model_props: ModelProperties = ModelProperties(),
                          init_var: Union[float, jax.Array] = 5,
                          cost_params: Optional = None,
                          key: Optional = None,
                          optimizer_key: Optional = None,
                          use_mean: Optional[jax.Array] = jnp.ones(1),
                          sampling_idx: Optional[Union[jnp.ndarray, int]] = None,
                          ):
    def dynamics(x, u, t, dynamic_params):
        action = jnp.tanh(u)
        return dynamics_fn(
            parameters=dynamic_params,
            obs=x,
            action=action,
            rng=key,
            model_props=model_props,
            sampling_idx=sampling_idx,
        )

    def cost_fn(x, u, t, params):
        action = jnp.tanh(u)
        return - reward_fn(x, action).sum()/horizon

    def sample_action(opt_key):
        sampled_action = jax.random.multivariate_normal(
            key=opt_key,
            mean=jnp.zeros_like(initial_actions.reshape(-1, 1).squeeze()),
            cov=jnp.diag(jnp.ones_like(initial_actions.reshape(-1, 1).squeeze()))
        ) * init_var
        sampled_action = sampled_action.reshape(initial_actions.shape)
        return sampled_action

    def get_zero(opt_key):
        return jnp.zeros_like(initial_actions)
    if optimizer_key is not None:
        sampled_action = jax.lax.cond(use_mean, sample_action, get_zero, optimizer_key)
        init_act = initial_actions + sampled_action
    else:
        init_act = initial_actions
    ilqr = ILQR(cost_fn, dynamics)
    out = ilqr.solve(cost_params, dynamics_params, initial_state, init_act, optimizer_params)
    return jnp.clip(jnp.tanh(out.us), -1, 1), -out.obj


class TraJaxTO(DummyPolicyOptimizer):

    def __init__(self,
                 horizon: int,
                 action_dim: tuple,
                 dynamics_model_list: list,
                 n_particles: int = 10,
                 params: ILQRHyperparams = ILQRHyperparams(),
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.horizon = horizon
        self.action_dim = action_dim
        self.optimizer_params = params
        self.n_particles = n_particles
        assert isinstance(dynamics_model_list, list)
        self.dynamics_model_list = dynamics_model_list
        opt_action_dim = (self.horizon,) + self.action_dim
        self.best_sequences = BestSequences(
            evaluation_sequences=jnp.zeros((len(self.dynamics_model_list),) + opt_action_dim),
            exploration_sequence=jnp.zeros(opt_action_dim),
        )
        self._init_fn()

    def _init_fn(self):

        self.optimize_for_eval_fns = []
        for i in range(len(self.dynamics_model_list)):
            self.optimize_for_eval_fns.append(functools.partial(
                self.optimize_action_sequence_for_evaluation, model_index=i
            ))
        self.optimize = self.optimize_for_eval_fns[0]

    def optimize_for_exploration(self,
                                 dynamics_params,
                                 obs,
                                 key=None,
                                 optimizer_key=None,
                                 model_props: ModelProperties = ModelProperties(),
                                 initial_actions: Optional[jax.Array] = None,
                                 sampling_idx: Optional[Union[jnp.ndarray, int]] = None,
                                 ):
        evaluate_fn = self.dynamics_model.evaluate_for_exploration

        if initial_actions is None:
            prev_best = jnp.zeros_like(self.best_sequences.exploration_sequence)
            prev_best = prev_best.at[:-1].set(self.best_sequences.exploration_sequence[1:])
            last_input = self.best_sequences.exploration_sequence[-1]
            prev_best = prev_best.at[-1].set(last_input)
            initial_actions = prev_best

        def optimize(init_state, rng, opt_rng):
            return self.get_exploration_sequence_and_returns_for_init_state(
                evaluate_fn=evaluate_fn,
                x0=init_state,
                optimizer_params=self.optimizer_params,
                dynamics_params=dynamics_params,
                model_props=model_props,
                key=rng,
                opt_key=opt_rng,
                sampling_idx=sampling_idx,
                initial_actions=initial_actions,
            )

        best_sequence, best_reward = jax.vmap(optimize)(obs, key, optimizer_key)
        self.best_sequences = BestSequences(
            evaluation_sequences=self.best_sequences.evaluation_sequences,
            exploration_sequence=best_sequence[0],
        )
        return best_sequence, best_reward

    def optimize_action_sequence_for_evaluation(self,
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

        predict_fn = self.dynamics_model_list[model_index].predict

        reward_fn = self.dynamics_model_list[model_index].reward_model.predict

        def optimize(init_state, rng, opt_rng):
            return self.get_sequence_and_returns_for_init_state(
                predict_fn=predict_fn,
                reward_fn=reward_fn,
                x0=init_state,
                optimizer_params=self.optimizer_params,
                dynamics_params=dynamics_params,
                model_props=model_props,
                key=rng,
                opt_key=opt_rng,
                sampling_idx=sampling_idx,
                initial_actions=initial_actions,
            )

        best_sequence, best_reward = jax.vmap(optimize)(obs, key, optimizer_key)
        sequence_copy = self.best_sequences.evaluation_sequences.at[model_index].set(best_sequence[0])
        self.best_sequences = BestSequences(
            evaluation_sequences=sequence_copy,
            exploration_sequence=self.best_sequences.exploration_sequence,
        )
        return best_sequence, best_reward

    @functools.partial(jax.jit, static_argnums=(0, 1, 2))
    def get_sequence_and_returns_for_init_state(self,
                                                predict_fn: Callable,
                                                reward_fn: Callable,
                                                x0: jax.Array,
                                                optimizer_params: Any = None,
                                                dynamics_params: Any = None,
                                                model_props: ModelProperties = ModelProperties(),
                                                key: Optional[jax.random.PRNGKeyArray] = None,
                                                opt_key: Optional[jax.random.PRNGKeyArray] = None,
                                                sampling_idx: Optional[Union[jnp.ndarray, int]] = None,
                                                initial_actions: Optional[jax.Array] = None,
                                                ):
        if initial_actions is None:
            initial_actions = jnp.zeros((self.horizon,) + self.action_dim)

        x0 = jnp.repeat(jnp.expand_dims(x0, 0), self.n_particles + 1, 0)

        def get_sequence_and_returns_for_init_state(init_state, optimizer_key, use_mean):
            return _optimize_with_params(
                reward_fn=reward_fn,
                dynamics_fn=predict_fn,
                horizon=self.horizon,
                initial_state=init_state,
                initial_actions=initial_actions,
                optimizer_params=optimizer_params,
                dynamics_params=dynamics_params,
                model_props=model_props,
                key=key,
                optimizer_key=optimizer_key,
                use_mean=use_mean,
                sampling_idx=sampling_idx,
            )

        if opt_key is not None:
            optimizer_key = jax.random.split(key=opt_key, num=self.n_particles + 1)
            use_mean = jnp.zeros(self.n_particles + 1)
            use_mean = use_mean.at[-1].set(1)
            sequence, returns = jax.vmap(get_sequence_and_returns_for_init_state)(x0, optimizer_key, use_mean)
        else:
            use_mean = jnp.ones(1)
            sequence, returns = jax.vmap(get_sequence_and_returns_for_init_state, in_axes=(0, None, None)) \
                (x0, opt_key, use_mean)
        best_elite_idx = jnp.argsort(returns, axis=0).squeeze()[-1]
        elite_action = sequence[best_elite_idx]
        return elite_action, returns[best_elite_idx]

    @functools.partial(jax.jit, static_argnums=(0, 1))
    def get_exploration_sequence_and_returns_for_init_state(self,
                                                            evaluate_fn,
                                                            x0: jax.Array,
                                                            optimizer_params: Any = None,
                                                            dynamics_params: Any = None,
                                                            model_props: ModelProperties = ModelProperties(),
                                                            key: Optional[jax.random.PRNGKeyArray] = None,
                                                            opt_key: Optional[jax.random.PRNGKeyArray] = None,
                                                            sampling_idx: Optional[Union[jnp.ndarray, int]] = None,
                                                            initial_actions: Optional[jax.Array] = None,
                                                            ):
        dynamics = evaluate_fn
        dummy_state = jnp.atleast_1d(jnp.zeros_like(x0)[..., -1])
        x0 = jnp.concatenate([x0, dummy_state], axis=-1)

        def predict_fn(obs, action, *args, **kwargs):
            state = obs[:-1]
            next_state, reward = dynamics(obs=state, action=action, *args, **kwargs)
            next_state = jnp.concatenate([next_state, reward.reshape(-1)], axis=0)
            return next_state

        def reward_fn(state, action, *args, **kwargs):
            reward = state[-1]
            return reward.mean()

        return self.get_sequence_and_returns_for_init_state(
            predict_fn=predict_fn,
            reward_fn=reward_fn,
            x0=x0,
            optimizer_params=optimizer_params,
            dynamics_params=dynamics_params,
            model_props=model_props,
            key=key,
            opt_key=opt_key,
            sampling_idx=sampling_idx,
            initial_actions=initial_actions,
        )

    @property
    def dynamics_model(self):
        return self.dynamics_model_list[0]

    def reset(self):
        self.best_sequences = BestSequences(
            evaluation_sequences=jnp.zeros_like(self.best_sequences.evaluation_sequences),
            exploration_sequence=jnp.zeros_like(self.best_sequences.exploration_sequence)
        )
