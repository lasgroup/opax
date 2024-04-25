from opax.models.environment_models.cartpole_reward_model import CartPoleRewardModel
from opax.envs.dm_control_env import DeepMindBridge
from dm_control.suite.cartpole import Balance, _DEFAULT_TIME_LIMIT, get_model_and_assets, Physics
from dm_control.rl.control import Environment
from dm_control.utils import containers

SUITE = containers.TaggedTasks()


@SUITE.add('benchmarking')
def run(time_limit=_DEFAULT_TIME_LIMIT, num_poles: int = 3, random=None, environment_kwargs=None):
    """Returns the run task."""
    physics = Physics.from_xml_string(*get_model_and_assets(num_poles=num_poles))
    task = Balance(swing_up=True, sparse=False, random=random)
    environment_kwargs = environment_kwargs or {}
    return Environment(
        physics, task, time_limit=time_limit, **environment_kwargs)


class CartPoleDM(DeepMindBridge):
    def __init__(self, reward_model: CartPoleRewardModel = CartPoleRewardModel(), *args, **kwargs):
        self.reward_model = reward_model
        env = run(time_limit=float('inf'), num_poles=reward_model.num_poles,
                  environment_kwargs={'flat_observation': True})
        super().__init__(env=env, *args, **kwargs)
        self.env = env

    def step(self, action):
        obs, reward, terminate, truncate, info = super().step(action)
        reward = self.reward_model.predict(obs=obs, action=action, next_obs=obs)
        reward = reward.astype(float).item()
        return obs, reward, terminate, truncate, info


if __name__ == "__main__":
    from gym.wrappers.time_limit import TimeLimit

    env = CartPoleDM(reward_model=CartPoleRewardModel())
    env = TimeLimit(env, max_episode_steps=1000)
    obs, _ = env.reset(seed=0)
    for i in range(1999):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()