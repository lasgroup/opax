from mbse.models.environment_models.swimmer_reward import SwimmerRewardModel
from mbse.envs.dm_control_env import DeepMindBridge
from dm_control.suite.swimmer import Swimmer, _DEFAULT_TIME_LIMIT, get_model_and_assets, Physics, _CONTROL_TIMESTEP
from dm_control.rl.control import Environment
import collections
from dm_control.utils import containers
from dm_control.suite.utils import randomizers

SUITE = containers.TaggedTasks()


@SUITE.add('benchmarking')
def _make_swimmer(n_joints=6, time_limit=_DEFAULT_TIME_LIMIT, random=None,
                  environment_kwargs=None):
    """Returns a swimmer control environment."""
    model_string, assets = get_model_and_assets(n_joints)
    physics = Physics.from_xml_string(model_string, assets=assets)
    task = CustomSwimmer(random=random)
    environment_kwargs = environment_kwargs or {}
    return Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)


class CustomSwimmer(Swimmer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_observation(self, physics):
        """Returns an observation of joint angles, body velocities and target."""
        obs = collections.OrderedDict()
        obs['joints'] = physics.joints()
        obs['body_velocities'] = physics.body_velocities()
        obs['to_target'] = physics.nose_to_target()
        return obs

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        Initializes the swimmer orientation to [-pi, pi) and the relative joint
        angle of each joint uniformly within its range.

        Args:
          physics: An instance of `Physics`.
        """
        # Random joint angles:
        randomizers.randomize_limited_and_rotational_joints(physics, self.random)
        # Random target position.
        close_target = self.random.rand() < .2  # Probability of a close target.
        target_box = .3 if close_target else 1.0
        xpos, ypos = self.random.uniform(-target_box, target_box, size=2)
        physics.named.model.geom_pos['target', 'x'] = xpos
        physics.named.model.geom_pos['target', 'y'] = ypos
        physics.named.model.light_pos['target_light', 'x'] = xpos
        physics.named.model.light_pos['target_light', 'y'] = ypos


class SwimmerEnvDM(DeepMindBridge):
    def __init__(self, reward_model: SwimmerRewardModel = SwimmerRewardModel(), *args, **kwargs):
        self.reward_model = reward_model
        env = _make_swimmer(time_limit=float('inf'), environment_kwargs={'flat_observation': True})
        super().__init__(env=env, *args, **kwargs)
        self.env = env

    def step(self, action):
        obs, reward, terminate, truncate, info = super().step(action)
        reward = self.reward_model.predict(obs=obs, action=action, next_obs=obs)
        reward = reward.astype(float).item()
        return obs, reward, terminate, truncate, info


if __name__ == "__main__":
    from gym.wrappers.time_limit import TimeLimit

    env = SwimmerEnvDM(reward_model=SwimmerRewardModel(), render_mode='human')
    env = TimeLimit(env, max_episode_steps=1000)
    obs, _ = env.reset(seed=10)
    for i in range(1999):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
