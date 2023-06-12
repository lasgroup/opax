import flax.struct
import numpy as np
from typing import Sequence, Callable, Optional
import optax
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import jit, random, value_and_grad
from mbse.utils.utils import get_idx
from mbse.utils.network_utils import MLP, mse
from mbse.utils.replay_buffer import ReplayBuffer, Transition
from flax import struct
from copy import deepcopy
import gym
from mbse.utils.utils import gaussian_log_likelihood, sample_normal_dist
from mbse.agents.dummy_agent import DummyAgent
import wandb
from typing import Any as Params
from mbse.utils.type_aliases import PolicyProperties
from mbse.utils.utils import convert_to_jax

EPS = 1e-6
ZERO = 0.0


@flax.struct.dataclass
class SACTrainingState:
    actor_opt_state: optax.OptState
    actor_params: Params
    critic_opt_state: optax.OptState
    critic_params: Params
    target_critic_params: Params
    alpha_opt_state: optax.OptState
    alpha_params: Params


@jit
def safe_clip_grads(grad_tree, max_norm=1e8):
    """Clip gradients stored as a pytree of arrays to maximum norm `max_norm`."""
    norm = optax.global_norm(grad_tree)
    eps = 1e-9
    normalize = lambda g: jnp.where(norm < max_norm, g, g * max_norm / (norm + eps))
    return jax.tree_util.tree_map(normalize, grad_tree)


# Perform Polyak averaging provided two network parameters and the averaging value tau.
# @partial(jit, static_argnums=(2, ))
def soft_update(
        target_params, online_params, tau=0.005
):
    updated_params = jax.tree_util.tree_map(
        lambda old, new: (1 - tau) * old + tau * new, target_params, online_params
    )
    return updated_params


@jit
def squash_action(action):
    squashed_action = nn.tanh(action)
    # sanity check to clip between -1 and 1
    squashed_action = jnp.clip(squashed_action, -0.999, 0.999)
    return squashed_action


# @partial(jit, static_argnums=(0,))
def get_squashed_log_prob(actor_fn, params, obs, rng):
    mu, sig = actor_fn(params, obs)
    u = sample_normal_dist(mu, sig, rng)
    log_l = gaussian_log_likelihood(u, mu, sig)
    a = squash_action(u)
    log_l -= jnp.sum(
        jnp.log((1 - jnp.square(a))), axis=-1
    )
    return a, log_l.reshape(-1, 1)


# @partial(jit, static_argnums=(0, 1, 10, 12))
def get_soft_td_target(
        critic_fn,
        actor_fn,
        next_obs,
        obs,
        reward,
        not_done,
        critic_target_params,
        critic_params,
        actor_params,
        alpha,
        discount,
        rng,
        reward_scale=1.0,
):
    next_action, next_action_log_a = get_squashed_log_prob(
        actor_fn=actor_fn,
        params=actor_params,
        obs=next_obs,
        rng=rng,
    )

    next_q1, next_q2 = critic_fn(critic_target_params, obs=next_obs, action=next_action)
    next_q = jnp.minimum(next_q1, next_q2)
    entropy_term = - alpha * next_action_log_a
    next_q_target = next_q + entropy_term
    target_v = reward_scale * reward + not_done * discount * next_q_target
    target_v_term = target_v.mean()
    return target_v, target_v_term, entropy_term.mean()


def get_action(actor_fn,
               actor_params,
               obs,
               rng=None,
               eval=False):
    mu, sig = actor_fn(actor_params, obs)

    def get_mean(mu, sig, rng):
        return mu

    def sample_action(mu, sig, rng):
        return sample_normal_dist(mu, sig, rng)

    action = jax.lax.cond(
        jnp.logical_or(rng is None, eval),
        get_mean,
        sample_action,
        mu,
        sig,
        rng
    )

    return squash_action(action)


def update_actor(
        actor_fn,
        critic_fn,
        actor_params,
        actor_update_fn,
        actor_opt_state,
        critic_params,
        alpha,
        obs,
        rng
):
    def loss(params):
        action, log_l = get_squashed_log_prob(actor_fn, params, obs, rng)
        q1, q2 = critic_fn(critic_params, obs=obs, action=action)
        min_q = jnp.minimum(q1, q2)
        actor_loss = - min_q + alpha * log_l
        return jnp.mean(actor_loss), log_l

    (loss, log_a), grads = value_and_grad(loss, has_aux=True)(actor_params)
    grads = safe_clip_grads(grads)
    updates, new_actor_opt_state = actor_update_fn(grads, actor_opt_state, params=actor_params)
    new_actor_params = optax.apply_updates(actor_params, updates)
    grad_norm = optax.global_norm(grads)
    return new_actor_params, new_actor_opt_state, loss, log_a, grad_norm


def update_critic(
        critic_fn,
        critic_update_fn,
        critic_params,
        critic_opt_state,
        obs,
        action,
        target_v,
):
    def loss(params):
        q1, q2 = critic_fn(params, obs, action)
        q_loss = jnp.mean(0.5 * (mse(q1, target_v) + mse(q2, target_v)))
        return q_loss

    loss, grads = value_and_grad(loss, has_aux=False)(critic_params)
    grads = safe_clip_grads(grads)
    updates, new_critic_opt_state = critic_update_fn(grads, critic_opt_state, params=critic_params)
    new_critic_params = optax.apply_updates(critic_params, updates)
    grad_norm = optax.global_norm(grads)
    return new_critic_params, new_critic_opt_state, loss, grad_norm


def update_alpha(log_alpha_fn, alpha_params, alpha_opt_state, alpha_update_fn, log_a, target_entropy):
    diff_entropy = jax.lax.stop_gradient(log_a + target_entropy)

    def loss(params):
        def alpha_loss_fn(lp):
            return -(
                    log_alpha_fn(params) * lp
            ).mean()

        return alpha_loss_fn(diff_entropy)

    loss, grads = value_and_grad(loss)(alpha_params)
    grads = safe_clip_grads(grads)
    updates, new_alpha_opt_state = alpha_update_fn(grads, alpha_opt_state, params=alpha_params)
    new_alpha_params = optax.apply_updates(alpha_params, updates)
    grad_norm = optax.global_norm(grads)
    return new_alpha_params, new_alpha_opt_state, loss, grad_norm


@struct.dataclass
class SACModelSummary:
    actor_loss: jnp.array = ZERO
    entropy: jnp.array = ZERO
    critic_loss: jnp.array = ZERO
    alpha_loss: jnp.array = ZERO
    log_alpha: jnp.array = ZERO
    actor_std: jnp.array = ZERO
    critic_grad_norm: jnp.array = ZERO
    actor_grad_norm: jnp.array = ZERO
    alpha_grad_norm: jnp.array = ZERO
    target_v_term: jnp.array = ZERO
    entropy_term: jnp.array = ZERO
    max_reward: jnp.array = ZERO
    min_reward: jnp.array = ZERO

    def dict(self):
        def get_logging_value(x):
            if isinstance(x, jax.Array) or isinstance(x, np.ndarray):
                if x.ndim > 1:
                    return x[-1].item()
                else:
                    return x.item()
            else:
                return x.item()

        return {
            'actor_loss': get_logging_value(self.actor_loss),
            'entropy': get_logging_value(self.entropy),
            'actor_std': get_logging_value(self.actor_std),
            'critic_loss': get_logging_value(self.critic_loss),
            'alpha_loss': get_logging_value(self.alpha_loss),
            'log_alpha': get_logging_value(self.log_alpha),
            'critic_grad_norm': get_logging_value(self.critic_grad_norm),
            'actor_grad_norm': get_logging_value(self.actor_grad_norm),
            'alpha_grad_norm': get_logging_value(self.alpha_grad_norm),
            'target_v_term': get_logging_value(self.target_v_term),
            'entropy_term': get_logging_value(self.entropy_term),
            'max_reward': get_logging_value(self.max_reward),
            'min_reward': get_logging_value(self.min_reward),
        }


class Actor(nn.Module):
    features: Sequence[int]
    action_dim: int
    non_linearity: Callable = nn.relu
    sig_min: float = float(1e-6)
    sig_max: float = float(1e2)

    @nn.compact
    def __call__(self, obs):
        actor_net = MLP(self.features,
                        2 * self.action_dim,
                        self.non_linearity)

        out = actor_net(obs)
        mu, sig = jnp.split(out, 2, axis=-1)
        sig = nn.softplus(sig)
        sig = jnp.clip(sig, self.sig_min, self.sig_max)
        return mu, sig


class Critic(nn.Module):
    features: Sequence[int]
    non_linearity: Callable = nn.relu

    @nn.compact
    def __call__(self, obs, action):
        critic_1 = MLP(
            features=self.features,
            output_dim=1,
            non_linearity=self.non_linearity)

        critic_2 = MLP(
            features=self.features,
            output_dim=1,
            non_linearity=self.non_linearity)
        obs_action = jnp.concatenate((obs, action), -1)
        value_1 = critic_1(obs_action)
        value_2 = critic_2(obs_action)
        return value_1, value_2


class ConstantModule(nn.Module):
    ent_coef_init: float = 1.0

    def setup(self):
        self.log_ent_coef = self.param("log_ent_coef",
                                       init_fn=lambda key: jnp.full((),
                                                                    jnp.log(self.ent_coef_init)))

    @nn.compact
    def __call__(self):
        return self.log_ent_coef


class SACAgent(DummyAgent):

    def __init__(
            self,
            action_space: gym.spaces.box,
            observation_space: gym.spaces.box,
            discount: float = 0.99,
            lr_actor: float = 1e-3,
            weight_decay_actor: float = 1e-5,
            lr_critic: float = 1e-3,
            weight_decay_critic: float = 1e-5,
            lr_alpha: float = 1e-3,
            weight_decay_alpha: float = 0.0,
            actor_features: Sequence[int] = [256, 256],
            critic_features: Sequence[int] = [256, 256, 256, 256],
            target_entropy: Optional[float] = None,
            rng: jax.Array = random.PRNGKey(0),
            q_update_frequency: int = 1,
            scale_reward: float = 1,
            tau: float = 0.005,
            init_ent_coef: float = 1.0,
            tune_entropy_coef: bool = True,
            *args,
            **kwargs
    ):
        super(SACAgent, self).__init__(*args, **kwargs)
        action_dim = np.prod(action_space.shape)
        sample_obs = observation_space.sample()
        sample_act = action_space.sample()
        self.tune_entropy_coef = tune_entropy_coef
        self.obs_sample = sample_obs
        self.act_sample = sample_act
        self.action_dim = action_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_alpha = lr_alpha
        self.discount = discount
        self.target_entropy = -action_dim.astype(np.float32) if target_entropy is None else \
            target_entropy
        action_dim = int(action_dim)
        self.actor_optimizer = \
            optax.apply_if_finite(optax.adamw(learning_rate=lr_actor, weight_decay=weight_decay_actor),
                                  10000000
                                  )
        self.critic_optimizer = optax.apply_if_finite(
            optax.adamw(learning_rate=lr_critic, weight_decay=weight_decay_critic),
            10000000
        )
        self.alpha_optimizer = optax.apply_if_finite(
            optax.adamw(learning_rate=lr_alpha, weight_decay=weight_decay_alpha),
            10000000
        )
        self.actor = Actor(features=actor_features, action_dim=action_dim)
        self.critic = Critic(features=critic_features)
        self.log_alpha = ConstantModule(init_ent_coef)
        self.critic_features = critic_features

        rng, actor_rng, critic_rng, alpha_rng = random.split(rng, 4)
        actor_params = self.actor.init(actor_rng, sample_obs)
        actor_opt_state = self.actor_optimizer.init(actor_params)
        critic_params = self.critic.init(
            critic_rng, sample_obs, sample_act
        )
        target_critic_params = deepcopy(critic_params)
        critic_opt_state = self.critic_optimizer.init(critic_params)

        alpha_params = self.log_alpha.init(alpha_rng)
        alpha_opt_state = self.alpha_optimizer.init(alpha_params)
        self.training_state = SACTrainingState(
            actor_opt_state=actor_opt_state,
            actor_params=actor_params,
            critic_opt_state=critic_opt_state,
            critic_params=critic_params,
            target_critic_params=target_critic_params,
            alpha_opt_state=alpha_opt_state,
            alpha_params=alpha_params,
        )
        self.policy_props = PolicyProperties()

        self.q_update_frequency = q_update_frequency
        self.scale_reward = scale_reward
        self.tau = tau
        self._init_fn()

    def _init_fn(self):
        self.log_alpha_apply = jax.jit(self.log_alpha.apply)
        self.actor_apply = jax.jit(self.actor.apply)
        self.get_eval_action = jax.jit(
            lambda actor_params, obs, rng: get_action(
                actor_fn=self.actor.apply,
                actor_params=actor_params,
                obs=obs,
                rng=rng,
                eval=True
            )
        )
        self.get_action = jax.jit(
            lambda actor_params, obs, rng: get_action(
                actor_fn=self.actor.apply,
                actor_params=actor_params,
                obs=obs,
                rng=rng,
                eval=False,
            )
        )

        self.get_soft_td_target = jax.jit(lambda \
                                                  next_obs, obs, reward, not_done, critic_target_params, critic_params,
                                                  actor_params, alpha, rng: \
                                              get_soft_td_target(
                                                  critic_fn=self.critic.apply,
                                                  actor_fn=self.actor.apply,
                                                  next_obs=next_obs,
                                                  obs=obs,
                                                  reward=reward,
                                                  not_done=not_done,
                                                  critic_target_params=critic_target_params,
                                                  critic_params=critic_params,
                                                  actor_params=actor_params,
                                                  alpha=alpha,
                                                  discount=self.discount,
                                                  rng=rng,
                                                  reward_scale=self.scale_reward,
                                              ))

        self.update_critic = jax.jit(lambda critic_params, critic_opt_state, obs, action, target_v: update_critic(
                                             critic_fn=self.critic.apply,
                                             critic_params=critic_params,
                                             critic_opt_state=critic_opt_state,
                                             critic_update_fn=self.critic_optimizer.update,
                                             obs=obs,
                                             action=action,
                                             target_v=target_v,
                                         ))

        self.soft_update = jax.jit(lambda target_params, online_params:
                                   soft_update(target_params=target_params, online_params=online_params,
                                               tau=self.tau))

        self.update_actor = \
            jax.jit(lambda actor_params, critic_params, alpha, actor_opt_state, obs, rng:
                    update_actor(
                        actor_fn=self.actor.apply,
                        critic_fn=self.critic.apply,
                        actor_params=actor_params,
                        actor_update_fn=self.actor_optimizer.update,
                        critic_params=critic_params,
                        alpha=alpha,
                        actor_opt_state=actor_opt_state,
                        obs=obs,
                        rng=rng,
                    )
                    )
        self.update_alpha = jax.jit(lambda alpha_params, alpha_opt_state, log_a: update_alpha(
            log_alpha_fn=self.log_alpha.apply,
            alpha_params=alpha_params,
            alpha_opt_state=alpha_opt_state,
            alpha_update_fn=self.alpha_optimizer.update,
            log_a=log_a,
            target_entropy=self.target_entropy,
        ))

        def step(carry, ins):
            rng = carry[0]
            training_state = carry[1]
            tran = ins[-1]
            train_rng, rng = jax.random.split(rng, 2)

            (
                new_training_state, summary
            ) = \
                self._train_step_(
                    rng=train_rng,
                    tran=tran,
                    training_state=training_state,
                )
            carry = [
                rng,
                new_training_state
            ]
            outs = [summary]
            return carry, outs

        self.step = jax.jit(step)

    def act_in_jax(self, obs, rng=None, eval=False, eval_idx=0):
        bias_obs = self.policy_props.policy_bias_obs
        bias_scale = self.policy_props.policy_scale_obs
        obs = (obs - bias_obs) / (bias_scale + EPS)
        if eval:
            return self.get_eval_action(
                actor_params=self.training_state.actor_params,
                obs=obs,
                rng=rng,
            )
        else:
            return self.get_action(
                actor_params=self.training_state.actor_params,
                obs=obs,
                rng=rng,
            )

    def _train_step_(self,
                     rng,
                     tran: Transition,
                     training_state: SACTrainingState,
                     ):

        alpha_params = training_state.alpha_params
        alpha_opt_state = training_state.alpha_opt_state
        actor_params = training_state.actor_params
        actor_opt_state = training_state.actor_opt_state
        critic_params = training_state.critic_params
        target_critic_params = training_state.target_critic_params
        critic_opt_state = training_state.critic_opt_state
        rng, actor_rng, td_rng = random.split(rng, 3)
        alpha = jax.lax.stop_gradient(
            jnp.exp(
                self.log_alpha_apply(
                    alpha_params)
            )
        )
        td_rng, target_q_rng = random.split(td_rng, 2)
        target_v, target_v_term, entropy_term = \
            jax.lax.stop_gradient(
                self.get_soft_td_target(
                    next_obs=tran.next_obs,
                    obs=tran.obs,
                    reward=tran.reward,
                    not_done=1.0 - tran.done,
                    critic_target_params=target_critic_params,
                    critic_params=critic_params,
                    actor_params=actor_params,
                    alpha=alpha,
                    rng=target_q_rng,
                )
            )
        new_critic_params, new_critic_opt_state, critic_loss, critic_grad_norm = \
            self.update_critic(
                critic_params=critic_params,
                critic_opt_state=critic_opt_state,
                obs=tran.obs,
                action=tran.action,
                # target_q=target_q,
                target_v=target_v,
            )
        new_target_critic_params = self.soft_update(target_critic_params, new_critic_params)
        new_actor_params, new_actor_opt_state, actor_loss, log_a, actor_grad_norm = self.update_actor(
            actor_params=actor_params,
            critic_params=new_critic_params,
            alpha=alpha,
            actor_opt_state=actor_opt_state,
            obs=tran.obs,
            rng=actor_rng,
        )

        if self.tune_entropy_coef:
            new_alpha_params, new_alpha_opt_state, alpha_loss, alpha_grad_norm = self.update_alpha(
                alpha_params=alpha_params,
                alpha_opt_state=alpha_opt_state,
                log_a=log_a,
            )

        else:
            alpha_loss = jnp.zeros(1)
            alpha_grad_norm = jnp.zeros(1)

        log_alpha = self.log_alpha_apply(new_alpha_params)
        _, std = self.actor_apply(new_actor_params, tran.obs)

        summary = SACModelSummary(
            actor_loss=actor_loss,
            entropy=-log_a.mean(),
            actor_std=std.mean(),
            critic_loss=critic_loss,
            alpha_loss=alpha_loss,
            log_alpha=log_alpha,
            actor_grad_norm=actor_grad_norm,
            alpha_grad_norm=alpha_grad_norm,
            critic_grad_norm=critic_grad_norm,
            target_v_term=target_v_term,
            entropy_term=entropy_term,
            max_reward=jnp.max(tran.reward),
            min_reward=jnp.min(tran.reward),
        )

        new_training_state = SACTrainingState(
            actor_opt_state=new_actor_opt_state,
            actor_params=new_actor_params,
            critic_opt_state=new_critic_opt_state,
            critic_params=new_critic_params,
            target_critic_params=new_target_critic_params,
            alpha_opt_state=new_alpha_opt_state,
            alpha_params=new_alpha_params,
        )

        return (
            new_training_state,
            summary,
        )

    def train_step(self,
                   rng,
                   buffer: ReplayBuffer,
                   validate: bool = True,
                   log_results: bool = True,
                   ):

        # @partial(jit, static_argnums=(0, 2))
        # def sample_data(data_buffer, rng, batch_size):
        #    tran = data_buffer.sample(rng, batch_size=batch_size)
        #    return tran
        transitions = buffer.sample(rng, batch_size=int(self.batch_size * self.train_steps))
        transitions = transitions.reshape(self.train_steps, self.batch_size)

        carry = [
            rng,
            self.training_state,
        ]
        ins = [transitions]
        carry, outs = jax.lax.scan(self.step, carry, ins, length=self.train_steps)
        self.training_state = carry[1]
        bias_obs = convert_to_jax(buffer.state_normalizer.mean)
        scale_obs = convert_to_jax(buffer.state_normalizer.std)
        self.policy_props = PolicyProperties(
            policy_bias_obs=bias_obs,
            policy_scale_obs=scale_obs,
        )
        summaries = outs[-1]
        if log_results:
            for idx in range(self.train_steps):
                summary = get_idx(summaries, idx)
                wandb.log(summary.dict())
        return self.train_steps
