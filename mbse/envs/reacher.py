from mbse.models.environment_models.reacher_reward_model import ReacherRewardModel
from mbse.envs.dm_control_env import DeepMindBridge
from dm_control.suite.reacher import Reacher, _DEFAULT_TIME_LIMIT, get_model_and_assets, Physics
from dm_control.rl.control import Environment
import collections
from dm_control.utils import containers

SUITE = containers.TaggedTasks()


@SUITE.add('benchmarking')
def run(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the run task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = CustomReacher(random=random)
    environment_kwargs = environment_kwargs or {}
    return Environment(physics, task, time_limit=time_limit,
                       **environment_kwargs)


class CustomReacher(Reacher):
    def __init__(self, *args, **kwargs):
        super().__init__(target_size=0.03, *args, **kwargs)

    def get_observation(self, physics):
        """Returns an observation of the state, ignoring horizontal position."""
        obs = collections.OrderedDict()
        obs['position'] = physics.position()
        obs['velocity'] = physics.velocity()
        obs['to_target'] = physics.finger_to_target()
        return obs


class ReacherEnvDM(DeepMindBridge):
    def __init__(self, reward_model: ReacherRewardModel = ReacherRewardModel(), *args, **kwargs):
        self.reward_model = reward_model
        env = run(time_limit=float('inf'), environment_kwargs={'flat_observation': True})
        super().__init__(env=env, *args, **kwargs)
        self.env = env

    def step(self, action):
        obs, reward, terminate, truncate, info = super().step(action)
        reward = self.reward_model.predict(obs=obs, action=action, next_obs=obs)
        reward = reward.astype(float).item()
        return obs, reward, terminate, truncate, info


if __name__ == "__main__":
    from gym.wrappers.time_limit import TimeLimit

    env = ReacherEnvDM(reward_model=ReacherRewardModel())
    env = TimeLimit(env, max_episode_steps=1000)
    obs, _ = env.reset(seed=0)
    for i in range(1999):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()