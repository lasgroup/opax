from gym import Wrapper


class ActionRepeat(Wrapper):
    def __init__(self, env, repeat):
        assert repeat >= 1, 'Expects at least one repeat.'
        super(ActionRepeat, self).__init__(env)
        self.repeat = repeat

    def step(self, action):
        done = False
        current_step = 0
        while current_step < self.repeat and not done:
            obs, reward, terminate, truncate, info = self.env.step(action)
            done = terminate or truncate
            current_step += 1
        return obs, reward, terminate, truncate, info