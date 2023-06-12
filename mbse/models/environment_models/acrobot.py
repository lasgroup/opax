import jax

from mbse.models.reward_model import RewardModel
from mbse.models.dynamics_model import DynamicsModel
from mbse.utils.type_aliases import ModelProperties
import jax.numpy as jnp
from functools import partial

from gym.envs.classic_control.acrobot import AcrobotEnv


class AcrobotReward(RewardModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    @jax.jit
    def predict(obs, action, next_obs=None, rng=None):
        cos_theta_1, sin_theta_1 = obs[..., 0], obs[..., 1]
        theta_1 = jnp.arctan2(sin_theta_1, cos_theta_1)
        cos_theta_2, sin_theta_2 = obs[..., 2], obs[..., 3]
        theta_2 = jnp.arctan2(sin_theta_2, cos_theta_2)
        at_top = (-jnp.cos(theta_1) - jnp.cos(theta_1 + theta_2)) > 1.0
        reward = -1.0 * (1 - at_top)
        return reward.reshape(-1).squeeze()


class AcrobotDynamics(DynamicsModel):
    def __init__(self, base_env: AcrobotEnv = AcrobotEnv(), *args, **kwargs):
        self.env = base_env
        self.reward_model = AcrobotReward()
        self.condensed_obs_size = 4
        self.full_obs_size = 6

    @partial(jax.jit, static_argnums=(0,))
    def predict(self, obs, action, rng=None, *args, **kwargs):
        obs = jnp.atleast_2d(obs).reshape(-1, self.full_obs_size)
        s = self.get_condensed_obs(obs)
        torque = action  # (between -1 and 1)

        # Add noise to the force action
        if self.env.torque_noise_max > 0:
            rng, sampling_rng = jax.random.split(rng, 2)
            torque += jax.random.uniform(
                key=rng,
                minval=-self.env.torque_noise_max, maxval=self.env.torque_noise_max
            )
        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        s_augmented = jnp.append(s, torque)

        ns = self.rk4(self._dsdt, s_augmented, self.env.dt)

        ns = ns.at[0:2].set(self.normalize_angle(ns[..., 0:2]))
        bounds = jnp.asarray([self.env.MAX_VEL_1, self.env.MAX_VEL_2])
        ns = ns.at[2:4].set(jnp.clip(ns[2:4], a_max=bounds, a_min=-bounds))
        next_obs = self.get_full_obs(ns)
        return next_obs

    @partial(jax.jit, static_argnums=(0,))
    def evaluate(self,
                 obs,
                 action,
                 parameters=None,
                 rng=None,
                 sampling_idx=None,
                 model_props: ModelProperties = ModelProperties()):
        next_state = self.predict(obs=obs, action=action, rng=rng)
        reward = self.reward_model.predict(obs=obs, action=action, next_obs=next_state)
        return next_state, reward

    @partial(jax.jit, static_argnums=(0, ))
    def get_condensed_obs(self, obs):
        obs = jnp.atleast_2d(obs).reshape(-1, self.full_obs_size)
        cos_theta_1, sin_theta_1 = obs[..., 0], obs[..., 1]
        theta_1 = jnp.arctan2(sin_theta_1, cos_theta_1)
        cos_theta_2, sin_theta_2 = obs[..., 2], obs[..., 3]
        theta_2 = jnp.arctan2(sin_theta_2, cos_theta_2)
        return jnp.concatenate([theta_1, theta_2, obs[..., 2], obs[..., 3]], axis=-1)

    @partial(jax.jit, static_argnums=(0, ))
    def get_full_obs(self, obs):
        obs = jnp.atleast_2d(obs).reshape(-1, self.condensed_obs_size)
        theta_1, theta_2 = obs[..., 0], obs[..., 1]
        return jnp.concatenate(
            [jnp.cos(theta_1), jnp.sin(theta_1), jnp.cos(theta_2), jnp.sin(theta_2), obs[..., 2], obs[..., 3]],
            axis=-1
        )

    @partial(jax.jit, static_argnums=(0,))
    def _dsdt(self, s_augmented):
        m1 = self.env.LINK_MASS_1
        m2 = self.env.LINK_MASS_2
        l1 = self.env.LINK_LENGTH_1
        lc1 = self.env.LINK_COM_POS_1
        lc2 = self.env.LINK_COM_POS_2
        I1 = self.env.LINK_MOI
        I2 = self.env.LINK_MOI
        g = 9.8
        a = jnp.atleast_2d(s_augmented[..., -1]).reshape(-1, 1)
        s = s_augmented[..., :-1]
        theta1 = jnp.atleast_2d(s[..., 0]).reshape(-1, 1)
        theta2 = jnp.atleast_2d(s[..., 1]).reshape(-1, 1)
        dtheta1 = jnp.atleast_2d(s[..., 2]).reshape(-1, 1)
        dtheta2 = jnp.atleast_2d(s[..., 3]).reshape(-1, 1)
        d1 = (
                m1 * lc1 ** 2
                + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * jnp.cos(theta2))
                + I1
                + I2
        )
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * jnp.cos(theta2)) + I2
        phi2 = m2 * lc2 * g * jnp.cos(theta1 + theta2 - jnp.pi / 2.0)
        phi1 = (
                -m2 * l1 * lc2 * dtheta2 ** 2 * jnp.sin(theta2)
                - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * jnp.sin(theta2)
                + (m1 * lc1 + m2 * l1) * g * jnp.cos(theta1 - jnp.pi / 2)
                + phi2
        )
        if self.env.book_or_nips == "nips":
            # the following line is consistent with the description in the
            # paper
            ddtheta2 = (a + d2 / d1 * phi1 - phi2) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        else:
            # the following line is consistent with the java implementation and the
            # book
            ddtheta2 = (
                               a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * jnp.sin(theta2) - phi2
                       ) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1

        dx = jnp.concatenate([dtheta1, dtheta2, ddtheta1, ddtheta2, jnp.zeros_like(dtheta1)], axis=-1)
        return dx.squeeze()

    @staticmethod
    def rk4(derivs, y0, dt):
        # for i in jnp.arange(len(t) - 1):
        dt2 = dt / 2.0
        k1 = derivs(y0)
        k2 = derivs(y0 + dt2 * k1)
        k3 = derivs(y0 + dt2 * k2)
        k4 = derivs(y0 + dt * k3)
        yout = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return yout[..., :4]

    @staticmethod
    @jax.jit
    def normalize_angle(x):
        """Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
        truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
        For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.
        Args:
            x: a scalar
            m: minimum possible value in range
            M: maximum possible value in range
        Returns:
            x: a scalar, wrapped
        """
        sin_x, cos_x = jnp.sin(x), jnp.cos(x)
        return jnp.arctan2(sin_x, cos_x)
