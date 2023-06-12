from mbse.utils.replay_buffer import Transition


class RewardModel(object):
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, obs, action, next_obs=None, rng=None):
        pass

    def train_step(self, tran: Transition):
        pass

    def set_bounds(self, max_action, min_action=None):
        pass
