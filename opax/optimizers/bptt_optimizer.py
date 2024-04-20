from opax.agents.actor_critic.sac import safe_clip_grads, get_action, gaussian_log_likelihood
import jax
from jax import value_and_grad
import jax.numpy as jnp
from opax.optimizers.dummy_policy_optimizer import DummyPolicyOptimizer
from typing import Optional, Any, Sequence, Callable, Union
import flax
import optax
from gym.spaces.box import Box
from opax.models.active_learning_model import ActiveLearningPETSModel, ActiveLearningHUCRLModel
import jax.random as random
from copy import deepcopy
from flax import linen as nn
from opax.agents.actor_critic.sac import soft_update
from opax.utils.replay_buffer import NormalizerState, ReplayBuffer, JaxNormalizer, Transition
from opax.utils.type_aliases import ModelProperties, PolicyProperties
from opax.utils.network_utils import MLP
from opax.utils.utils import sample_trajectories, get_idx, tree_stack
import functools
import numpy as np
from optax import l2_loss
import math

EPS = 1e-6


def atanh(x: jax.Array) -> jax.Array:
    """
    Inverse of Tanh

    Taken from Pyro: https://github.com/pyro-ppl/pyro
    0.5 * torch.log((1 + x ) / (1 - x))
    """
    x = jnp.clip(x, -1 + EPS, 1 - EPS)
    y = 0.5 * jnp.log((1 + x) / (1 - x))
    return y


@functools.partial(jax.jit, static_argnums=(3, 4))
def lambda_return(reward: jax.Array, next_values: jax.Array, discount: float, lambda_: float):
    """Taken from https://github.com/danijar/dreamer/"""
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    assert reward.ndim == next_values.ndim, (reward.shape, next_values.shape)
    inputs = reward + discount * next_values * (1 - lambda_)
    returns = static_scan(
        lambda agg, inp: inp + discount * lambda_ * agg,
        inputs, next_values[-1], reverse=True)
    return returns


def static_scan(fn, inputs: jax.Array, start: jax.Array, reverse=False):
    """Taken from https://github.com/danijar/dreamer/"""
    if reverse:
        inputs = jax.tree_util.tree_map(lambda x: x[::-1], inputs)

    def step(carry, ins):
        x = carry
        inp = ins
        next = fn(x, inp)
        out = next
        carry = next
        return carry, out

    carry = start
    carry, outs = jax.lax.scan(step, carry, xs=inputs)
    if reverse:
        outs = jax.tree_util.tree_map(lambda x: x[::-1], outs)
    return outs


def get_log_prob(squashed_action: jax.Array, obs: jax.Array, params: jax.Array, apply_fn: Callable):
    mu, sig = apply_fn(params, obs)
    u = atanh(squashed_action)
    log_l = gaussian_log_likelihood(u, mu, sig)
    log_l -= jnp.sum(
        jnp.log((1 - jnp.square(squashed_action))), axis=-1
    )
    return log_l.reshape(-1, 1)


@flax.struct.dataclass
class BPTTState:
    actor_opt_state: optax.OptState
    actor_params: Any
    critic_opt_state: optax.OptState
    critic_params: Any
    target_critic_params: Any


@flax.struct.dataclass
class BPTTAgentSummary:
    actor_grad_norm: jax.Array
    critic_grad_norm: jax.Array
    actor_loss: jax.Array
    critic_loss: jax.Array
    reward: jax.Array = jnp.zeros(1)

    def dict(self):
        def get_logging_value(x):
            if isinstance(x, jax.Array) or isinstance(x, np.ndarray):
                if x.shape[-1] > 1:
                    return x[-1].item()
                else:
                    return x.item()
            else:
                return x.item()

        return {
            'actor_loss': get_logging_value(self.actor_loss),
            'critic_loss': get_logging_value(self.critic_loss),
            'critic_grad_norm': get_logging_value(self.critic_grad_norm),
            'actor_grad_norm': get_logging_value(self.actor_grad_norm),
            'reward': get_logging_value(self.reward),
        }


@flax.struct.dataclass
class BPTTNormalizerState:
    state_normalizer_state: NormalizerState
    reward_normalizer_state: NormalizerState


@functools.partial(
    jax.jit, static_argnums=(0, 2, 3, 4, 12, 13)
)
def _rollout_models(horizon: int,
                    obs: jax.Array,
                    policy: Callable,
                    critic: Callable,
                    evaluate_fn: Callable,
                    actor_params: Any,
                    critic_params: Any,
                    bptt_normalizer_state: BPTTNormalizerState,
                    dynamics_params=None,
                    key=None,
                    model_props: ModelProperties = ModelProperties(),
                    sampling_idx: Optional[Union[jnp.ndarray, int]] = None,
                    discount: float = 0.99,
                    lambda_: float = 0.97,
                    ):
    policy_props = PolicyProperties(
        policy_bias_obs=bptt_normalizer_state.state_normalizer_state.mean,
        policy_scale_obs=bptt_normalizer_state.state_normalizer_state.std,
    )

    def sample_trajectories_for_state(state: jax.Array, sample_key):
        trajectory = sample_trajectories(
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
            stop_grads=True,
        )

        next_obs = trajectory.next_obs
        next_obs = next_obs - bptt_normalizer_state.state_normalizer_state.mean / \
                   (bptt_normalizer_state.state_normalizer_state.std + EPS)
        reward = trajectory.reward.squeeze(-1)
        reward = (reward - bptt_normalizer_state.reward_normalizer_state.mean) / \
                 (bptt_normalizer_state.reward_normalizer_state.std + EPS)
        v_1, v_2 = critic(critic_params, next_obs)
        bootstrap_values = jnp.minimum(v_1, v_2)
        lambda_values = lambda_return(reward,
                                      bootstrap_values, discount, lambda_)
        return lambda_values, trajectory

    key = jax.random.split(key, obs.shape[0])
    lambda_values, trajectories = jax.vmap(sample_trajectories_for_state)(obs, key)
    return lambda_values, trajectories


@functools.partial(
    jax.jit, static_argnums=(1, 2, 3, 4, 7, 8, 9, 14, 15, 16, 17, 18)
)
def update_actor_critic(
        initial_states: jax.Array,
        horizon: int,
        policy: Callable,
        critic: Callable,
        evaluate_fn: Callable,
        bptt_state: BPTTState,
        bptt_normalizer_state: BPTTNormalizerState,
        actor_update_fn: Callable,
        critic_update_fn: Callable,
        actor_apply_fn: Callable,
        dynamics_params=None,
        key=None,
        model_props: ModelProperties = ModelProperties(),
        sampling_idx: Optional[Union[jnp.ndarray, int]] = None,
        critic_updates_per_policy_update: int = 100,
        discount: float = 0.99,
        lambda_: float = 0.97,
        loss_ent_coefficient: float = 0.001,
        tau: float = 0.005,
):
    actor_training_key, key = jax.random.split(key, 2)

    def actor_loss_fn(params):
        lambda_values, trajectories = _rollout_models(
            horizon=horizon,
            obs=initial_states,
            policy=policy,
            critic=critic,
            evaluate_fn=evaluate_fn,
            bptt_normalizer_state=bptt_normalizer_state,
            actor_params=params,
            critic_params=bptt_state.target_critic_params,
            dynamics_params=dynamics_params,
            key=key,
            model_props=model_props,
            sampling_idx=sampling_idx,
            discount=discount,
            lambda_=lambda_
        )

        def act_loss(lambda_values, obs, actions):
            obs = obs - bptt_normalizer_state.state_normalizer_state.mean / \
                  (bptt_normalizer_state.state_normalizer_state.std + EPS)
            pcont = jnp.ones(horizon)
            pcont = pcont.at[1:].set(discount)
            disc = jnp.cumprod(pcont)
            log_prob = get_log_prob(squashed_action=actions, obs=obs, params=params, apply_fn=actor_apply_fn)
            entropy_loss = -log_prob.mean()
            actor_loss = -(lambda_values * disc).mean() + entropy_loss * loss_ent_coefficient
            return actor_loss, entropy_loss

        actor_loss, entropy_loss = jax.vmap(act_loss)(lambda_values, trajectories.obs, trajectories.action)
        return actor_loss.mean(), (trajectories, lambda_values, entropy_loss.mean())

    rest, grads = value_and_grad(actor_loss_fn, has_aux=True)(
        bptt_state.actor_params
    )
    actor_loss, (trajectories, lambda_values, entropy_loss) = rest
    grads = safe_clip_grads(grads)
    updates, new_actor_opt_state = actor_update_fn(grads, bptt_state.actor_opt_state, params=bptt_state.actor_params)
    new_actor_params = optax.apply_updates(bptt_state.actor_params, updates)
    actor_grad_norm = optax.global_norm(grads)

    critic_training_key, key = jax.random.split(key, 2)
    num_transitions = initial_states.shape[0] * horizon
    trajectories = jax.tree_util.tree_map(lambda arr: arr.reshape(-1, arr.shape[-1]), trajectories)
    batch_size = math.ceil(num_transitions / critic_updates_per_policy_update)
    transition_indices = jax.random.randint(critic_training_key, minval=0, maxval=num_transitions,
                                            shape=(critic_updates_per_policy_update, batch_size))
    shuffled_transitions = jax.tree_util.tree_map(lambda x: x[transition_indices], trajectories)
    shuffled_lambda = lambda_values.reshape(-1)[transition_indices]

    def update_critic(carry, ins):
        critic_params, critic_opt_state, target_critic_params = carry[0], carry[1], carry[2]

        def critic_loss_fn(params) -> jax.Array:
            critic_fn = lambda obs: critic(params, obs)
            traj, lamb = ins[0], ins[1]
            def loss_fn(obs: jax.Array, lambda_values: jax.Array,
                        bptt_normalizer_state: BPTTNormalizerState):
                obs = obs - bptt_normalizer_state.state_normalizer_state.mean / \
                      (bptt_normalizer_state.state_normalizer_state.std + EPS)
                v_1, v_2 = critic_fn(obs)
                target = lambda_values
                v_loss = 0.5 * (l2_loss(v_1, target).mean() + l2_loss(v_2, target).mean())
                return v_loss

            # v_loss = jax.vmap(loss_fn, in_axes=(0, 0, None))(traj.next_obs, lamb,
            #                                                    bptt_normalizer_state)
            return loss_fn(traj.next_obs, lamb, bptt_normalizer_state)
            # return v_loss.mean()

        critic_loss, grads = value_and_grad(critic_loss_fn, has_aux=False)(critic_params)
        grads = safe_clip_grads(grads)
        updates, new_critic_opt_state = critic_update_fn(grads, critic_opt_state,
                                                         params=critic_params)
        new_critic_params = optax.apply_updates(critic_params, updates)
        critic_grad_norm = optax.global_norm(grads)
        new_target_params = soft_update(target_critic_params, new_critic_params, tau=tau)
        outs = [critic_loss, critic_grad_norm]
        carry = [new_critic_params, new_critic_opt_state, new_target_params]
        return carry, outs

    carry = [bptt_state.critic_params, bptt_state.critic_opt_state, bptt_state.target_critic_params]
    carry, outs = jax.lax.scan(update_critic, carry, xs=[shuffled_transitions, shuffled_lambda],
                               length=critic_updates_per_policy_update)
    new_critic_params, new_critic_opt_state, new_target_params = carry[0], carry[1], carry[2]
    critic_loss = outs[0][-1]
    critic_grad_norm = outs[1][-1]
    new_bptt_state = BPTTState(
        actor_opt_state=new_actor_opt_state,
        actor_params=new_actor_params,
        critic_opt_state=new_critic_opt_state,
        critic_params=new_critic_params,
        target_critic_params=new_target_params,
    )
    summary = BPTTAgentSummary(
        actor_grad_norm=actor_grad_norm,
        critic_grad_norm=critic_grad_norm,
        actor_loss=actor_loss,
        critic_loss=critic_loss,
    )
    return new_bptt_state, summary, trajectories


def inv_softplus(x: Union[jax.Array, float]) -> jax.Array:
    return jnp.where(x < 20.0, jnp.log(jnp.exp(x) - 1.0), x)


class Actor(nn.Module):
    features: Sequence[int]
    action_dim: int
    non_linearity: Callable = nn.relu
    init_stddev: float = float(1.0)
    sig_min: float = float(1e-6)
    sig_max: float = float(1e2)

    @nn.compact
    def __call__(self, obs):
        actor_net = MLP(self.features,
                        2 * self.action_dim,
                        self.non_linearity)

        out = actor_net(obs)
        mu, sig = jnp.split(out, 2, axis=-1)
        init_std = inv_softplus(self.init_stddev)
        sig = nn.softplus(sig + init_std)
        sig = jnp.clip(sig, self.sig_min, self.sig_max)
        return mu, sig


class Critic(nn.Module):
    features: Sequence[int]
    non_linearity: Callable = nn.swish

    @nn.compact
    def __call__(self, obs):
        critic_1 = MLP(
            features=self.features,
            output_dim=1,
            non_linearity=self.non_linearity)

        critic_2 = MLP(
            features=self.features,
            output_dim=1,
            non_linearity=self.non_linearity)
        value_1 = critic_1(obs).squeeze(-1)
        value_2 = critic_2(obs).squeeze(-1)
        return value_1, value_2


class BPTTOptimizer(DummyPolicyOptimizer):

    def __init__(self,
                 action_dim: tuple,
                 dynamics_model_list: list,
                 horizon: int = 20,
                 n_particles: int = 10,
                 transitions_per_update: int = 10,
                 train_steps: int = 20,
                 normalize: bool = False,
                 action_normalize: bool = False,
                 actor_features: Sequence[int] = [256, 256],
                 critic_features: Sequence[int] = [256, 256, 256, 256],
                 init_stddev: float = 1.0,
                 lr_actor: float = 1e-3,
                 weight_decay_actor: float = 1e-5,
                 lr_critic: float = 1e-3,
                 weight_decay_critic: float = 1e-5,
                 reset_optimizer: bool = True,
                 target_soft_update_tau: float = 0.005,
                 rng: jax.Array = random.PRNGKey(0),
                 evaluation_samples: int = 100,
                 evaluation_horizon: int = 100,
                 evaluation_frequency: int = -1,
                 critic_updates_per_policy_update: int = 100,
                 discount: float = 0.99,
                 lambda_: float = 0.97,
                 loss_ent_coefficient: float = 0.001,
                 use_best_trained_policy: bool = False,
                 *args,
                 **kwargs,
                 ):
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
        sample_obs = dummy_obs_space.sample()
        self.actor = Actor(features=actor_features, action_dim=np.prod(dummy_act_space.shape), init_stddev=init_stddev)
        self.critic = Critic(features=critic_features)
        actor_rng, critic_rng, rng = random.split(rng, 3)
        num_agents = len(self.dynamics_model_list)
        actor_rng = random.split(actor_rng, num_agents)
        critic_rng = random.split(critic_rng, num_agents)

        critic_params = jax.vmap(self.critic.init, in_axes=(0, None))(
            critic_rng,
            sample_obs,
        )

        actor_params = jax.vmap(self.actor.init, in_axes=(0, None))(
            actor_rng,
            sample_obs,
        )

        self.actor_optimizer = \
            optax.apply_if_finite(optax.adamw(learning_rate=lr_actor, weight_decay=weight_decay_actor),
                                  10000000
                                  )
        actor_opt_state = jax.vmap(self.actor_optimizer.init)(actor_params)
        self.critic_optimizer = optax.apply_if_finite(
            optax.adamw(learning_rate=lr_critic, weight_decay=weight_decay_critic),
            10000000
        )
        critic_opt_state = jax.vmap(self.critic_optimizer.init)(critic_params)
        target_critic_params = deepcopy(critic_params)
        self.training_state = jax.vmap(BPTTState)(actor_opt_state, actor_params, critic_opt_state, critic_params,
                                                  target_critic_params)
        self.init_state = deepcopy(self.training_state)

        self.horizon = horizon
        self.n_particles = n_particles
        self.transitions_per_update = transitions_per_update
        self.normalize = normalize
        self.action_normalize = action_normalize
        self.obs_dim = (obs_dim,)
        self.action_dim = action_dim
        self.train_steps = train_steps
        self.reset_optimizer = reset_optimizer
        self.evaluate_agent = evaluation_frequency > 0
        self.evaluation_samples = evaluation_samples
        self.evaluation_horizon = evaluation_horizon
        self.evaluation_frequency = evaluation_frequency
        self.discount = discount
        self.lambda_ = lambda_
        self.tau = target_soft_update_tau
        self.use_best_trained_policy = use_best_trained_policy
        self.loss_ent_coefficient = loss_ent_coefficient

        self.act_in_eval = jax.jit(
            lambda actor_params, obs, rng: get_action(
                actor_fn=self.actor.apply,
                actor_params=actor_params,
                obs=obs,
                rng=rng,
                eval=True
            )
        )
        self.act = jax.jit(
            lambda actor_params, obs, rng: get_action(
                actor_fn=self.actor.apply,
                actor_params=actor_params,
                obs=obs,
                rng=rng,
                eval=False,
            )
        )

        self.critic_updates_per_policy_update = critic_updates_per_policy_update
        self.state_normalizer = JaxNormalizer(self.obs_dim)
        self.reward_normalizer = JaxNormalizer((1,))
        self.normalizer_state = self.initialize_normalizer_state()

    def update_normalizers(self, transition: Transition, normalizer_state: BPTTNormalizerState):
        state_normalizer_state = self.state_normalizer.update(transition.obs, normalizer_state.state_normalizer_state)
        reward_normalizer_state = self.reward_normalizer.update(transition.reward,
                                                                normalizer_state.reward_normalizer_state)
        new_normalizer_state = BPTTNormalizerState(
            state_normalizer_state=state_normalizer_state,
            reward_normalizer_state=reward_normalizer_state,
        )
        return new_normalizer_state

    def initialize_normalizer_state(self):
        num_agents = len(self.dynamics_model_list)
        x = jnp.ones(num_agents)

        def initialize(x):
            return BPTTNormalizerState(
                state_normalizer_state=self.state_normalizer.initialize_normalizer_state(),
                reward_normalizer_state=self.reward_normalizer.initialize_normalizer_state(),
            )

        normalizer_state = jax.vmap(initialize)(x)
        return normalizer_state

    @functools.partial(jax.jit, static_argnums=(0, 6))
    def train_single_agent(self,
                           rng: jax.random.PRNGKeyArray,
                           true_obs: jax.Array,
                           obs_size: jax.Array,
                           bptt_state: BPTTState,
                           bptt_normalizer_state: BPTTNormalizerState,
                           evaluate: Callable,
                           dynamics_params: Optional = None,
                           model_props: ModelProperties = ModelProperties(),
                           sampling_idx: Optional[Union[jnp.ndarray, int]] = None,
                           ):
        policy = self.act
        eval_policy = self.act_in_eval
        critic = self.critic.apply
        actor_update_fn = self.actor_optimizer.update
        critic_update_fn = self.critic_optimizer.update

        eval_rng, rng = jax.random.split(rng, 2)
        eval_idx = jax.random.randint(eval_rng, (self.evaluation_samples,), 0, obs_size)
        eval_obs = jnp.take(true_obs, eval_idx, axis=0, mode='wrap')
        eval_sim_key, rng = jax.random.split(rng, 2)

        def step(carry, ins):
            opt_state, normalizer_state, prev_reward, key = carry[0], carry[1], carry[2], carry[-1]
            best_reward, best_opt_state = carry[3], carry[4]
            buffer_rng, key = jax.random.split(key, 2)
            ind = jax.random.randint(buffer_rng, (self.transitions_per_update,), 0, obs_size)
            obs = jnp.take(true_obs, ind, axis=0, mode='wrap')
            train_rng, key = jax.random.split(key, 2)
            new_opt_state, summary, transitions = update_actor_critic(
                initial_states=obs,
                horizon=self.horizon,
                policy=policy,
                critic=critic,
                evaluate_fn=evaluate,
                bptt_state=opt_state,
                bptt_normalizer_state=normalizer_state,
                actor_update_fn=actor_update_fn,
                critic_update_fn=critic_update_fn,
                actor_apply_fn=self.actor.apply,
                dynamics_params=dynamics_params,
                key=train_rng,
                model_props=model_props,
                sampling_idx=sampling_idx,
                critic_updates_per_policy_update=self.critic_updates_per_policy_update,
                discount=self.discount,
                lambda_=self.lambda_,
                loss_ent_coefficient=self.loss_ent_coefficient,
                tau=self.tau,
            )
            new_normalizer_state = normalizer_state
            if self.normalize:
                new_normalizer_state = self.update_normalizers(transitions, normalizer_state=normalizer_state)

            if self.evaluate_agent:
                def evaluate_policy():
                    lambda_values, trajectories = _rollout_models(
                        horizon=self.evaluation_horizon,
                        obs=eval_obs,
                        policy=eval_policy,
                        critic=critic,
                        evaluate_fn=evaluate,
                        actor_params=new_opt_state.actor_params,
                        critic_params=new_opt_state.critic_params,
                        bptt_normalizer_state=normalizer_state,
                        dynamics_params=dynamics_params,
                        key=eval_sim_key,
                        model_props=model_props,
                        sampling_idx=sampling_idx,
                        discount=self.discount,
                        lambda_=self.lambda_
                    )
                    reward = trajectories.reward.mean()

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
                    jax.lax.cond(jnp.logical_or(ins % self.evaluation_frequency == 0, ins == self.train_steps - 1),
                                 evaluate_policy,
                                 skip_evaluation
                                 )
            else:
                reward = prev_reward
                new_best_reward, new_best_opt_state = reward, new_opt_state
            carry = [new_opt_state, new_normalizer_state, reward, new_best_reward, new_best_opt_state, key]
            summary = BPTTAgentSummary(
                actor_grad_norm=summary.actor_grad_norm,
                critic_grad_norm=summary.critic_grad_norm,
                actor_loss=summary.actor_loss,
                critic_loss=summary.critic_loss,
                reward=reward,
            )
            outs = [summary]
            return carry, outs

        carry = [bptt_state, bptt_normalizer_state, 0.0, -jnp.inf, bptt_state, rng]
        xs = jnp.arange(self.train_steps)
        carry, outs = jax.lax.scan(step, carry, xs=xs, length=self.train_steps)
        if self.use_best_trained_policy:
            trained_state = carry[-2]
        else:
            trained_state = carry[0]
        return trained_state, carry[1], outs[0]

    def train(self,
              rng,
              buffer: ReplayBuffer,
              dynamics_params: Optional = None,
              model_props: ModelProperties = ModelProperties(),
              sampling_idx: Optional[Union[jnp.ndarray, int]] = None,
              reset_agent: bool = True
              ):
        if self.reset_optimizer:
            full_optimizer_state = self.init_state
            normalizer_state = self.initialize_normalizer_state()
        else:
            full_optimizer_state = self.training_state
            normalizer_state = self.normalizer_state
        agent_summary = []
        trained_optimizer_state = []
        trained_normalizer_state = []
        for j in range(len(self.dynamics_model_list)):
            evaluate_fn = self.dynamics_model_list[j].evaluate
            optimizer_state = get_idx(full_optimizer_state, idx=j)
            bptt_normalizer_state = get_idx(normalizer_state, idx=j)
            if self.is_active_exploration_agent(idx=j):
                evaluate_fn = self.dynamics_model_list[j].evaluate_for_exploration
                optimizer_state = get_idx(self.init_state, idx=j)
                bptt_normalizer_state = BPTTNormalizerState(
                    state_normalizer_state=self.state_normalizer.initialize_normalizer_state(),
                    reward_normalizer_state=self.reward_normalizer.initialize_normalizer_state(),
                )
            new_optimizer_state, new_normalizer_state, summary = self.train_single_agent(
                rng=rng,
                true_obs=buffer.obs,
                obs_size=buffer.size,
                bptt_state=optimizer_state,
                bptt_normalizer_state=bptt_normalizer_state,
                evaluate=evaluate_fn,
                dynamics_params=dynamics_params,
                model_props=model_props,
                sampling_idx=sampling_idx,
            )
            agent_summary.append(summary)
            trained_optimizer_state.append(new_optimizer_state)
            trained_normalizer_state.append(new_normalizer_state)
        full_optimizer_state = tree_stack(trained_optimizer_state)
        self.normalizer_state = tree_stack(trained_normalizer_state)
        self.training_state = full_optimizer_state
        return agent_summary

    @property
    def dynamics_model(self):
        return self.dynamics_model_list[0]

    def is_active_exploration_agent(self, idx):
        return idx == len(self.dynamics_model_list) - 1 and self.active_exploration_agent

    def get_action_for_eval(self, obs: jax.Array, rng, agent_idx: int):
        policy = self.act_in_eval
        agent_state = get_idx(self.training_state, agent_idx)
        normalizer_state = get_idx(self.normalizer_state, agent_idx)
        normalized_obs = (obs - normalizer_state.state_normalizer_state.mean) / \
                         (normalizer_state.state_normalizer_state.std + EPS)
        action = policy(
            actor_params=agent_state.actor_params,
            obs=normalized_obs,
            rng=rng,
        )
        return action

    def get_action(self, obs: jax.Array, rng):
        return self.get_action_for_eval(obs=obs, rng=rng, agent_idx=0)

    def get_action_for_exploration(self, obs: jax.Array, rng, *args, **kwargs):
        if self.active_exploration_agent:
            policy = self.act
            agent_state = get_idx(self.training_state, -1)
            normalizer_state = get_idx(self.normalizer_state, -1)
            normalized_obs = (obs - normalizer_state.state_normalizer_state.mean) / \
                             (normalizer_state.state_normalizer_state.std + EPS)
            action = policy(
                actor_params=agent_state.actor_params,
                obs=normalized_obs,
                rng=rng,
            )
        else:
            policy = self.act
            agent_state = get_idx(self.training_state, 0)
            normalizer_state = get_idx(self.normalizer_state, -1)
            normalized_obs = (obs - normalizer_state.state_normalizer_state.mean) / \
                             (normalizer_state.state_normalizer_state.std + EPS)
            action = policy(
                actor_params=agent_state.actor_params,
                obs=normalized_obs,
                rng=rng,
            )
        return action
