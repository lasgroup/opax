import jax
from opax.utils.replay_buffer import Transition, ReplayBuffer
from opax.agents.dummy_agent import DummyAgent
from typing import Callable, Optional, Union
import wandb
from copy import deepcopy
import numpy as np
from opax.utils.vec_env import VecEnv
from tqdm import tqdm
from gym.wrappers.record_video import RecordVideo
from gym import Env
import glob
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import cloudpickle

@dataclass
class DummyTrainerState: # state of ongoing training to be used for resuming
    buffer: ReplayBuffer
    agent: DummyAgent
    rng: jax.random.PRNGKey
    eval_episodes: int
    agent_name: str
    total_train_steps: int
    train_freq: int
    eval_freq: int
    exploration_steps: int
    rollout_steps: int
    validate: bool
    record_test_video: bool
    video_dir_name: str


# TODO learn deltas, validate_agent should only be used by model based agent.
class DummyTrainer(object):
    """Dummy Trainer for training a RL agent

    env: VecEnv, Vectorized environment for RL
    agent: DummyAgent, RL Agent
    buffer_size: int, Replay buffer size
    total_train_steps: int, Total number of training steps for the RL agent.
    train_freq: int, frequency of training wrt rollouts in the real environment
    eval_freq: int, frequency of evaluation wrt rollouts in the real environment.
    seed: int
    exploration_steps: int, number of random data points collected before agent update.
    rollout_steps: int, Number of steps per rollout in the environment
    eval_episodes: int, Number of episodes used for evaluation
    agent_name: str, Name of the agent
    use_wandb: bool, boolean to indicated logging in wandb
    validate: bool, boolean to indicate policy validation (generally only used for model based RL)
    normalize: bool, boolean to indicate if data is normalized or not
    action_normalize: bool, boolean to indicate if actions should be normalized
    learn_deltas: bool, boolean to indicate if model learns a delta
    record_test_video: bool, boolean to indicate if video should be recorded for model evaluation
    video_prefix: str, prefix of the recorded video
    video_folder: str, folder where the videos are stored
    test_env: Optional[Union[VecEnv, Env]], environment on which the agent is tested.
    """

    def __init__(self,
                 env: VecEnv,
                 agent: DummyAgent,
                 buffer_size: int = int(1e6),
                 total_train_steps: int = int(1e6),
                 train_freq: int = 100,
                 eval_freq: int = 1000,
                 seed: int = 0,
                 exploration_steps: int = int(1e4),
                 rollout_steps: int = 200,
                 eval_episodes: int = 100,
                 agent_name: str = "DummyAgent",
                 use_wandb: bool = True,
                 validate: bool = False,
                 normalize: bool = False,
                 action_normalize: bool = False,
                 learn_deltas: bool = False,
                 record_test_video: bool = False,
                 checkpoint_folder: str = "./",
                 video_prefix: str = "",
                 video_folder: str = "./",
                 test_env: Optional[Union[VecEnv, Env]] = None,
                 ):
        self.env = env
        self.num_envs = max(env.num_envs, 1)
        self.buffer = ReplayBuffer(
            obs_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            max_size=buffer_size,
            normalize=normalize,
            action_normalize=action_normalize,
            learn_deltas=learn_deltas
        )
        self.buffer_size = buffer_size

        self.eval_episodes = eval_episodes
        self.agent_name = agent_name
        self.rng = jax.random.PRNGKey(seed)
        self.use_wandb = use_wandb
        self.validate = validate
        self.total_train_steps = total_train_steps
        self.train_freq = train_freq
        self.eval_freq = eval_freq
        self.exploration_steps = exploration_steps
        self.rollout_steps = rollout_steps
        self.agent = agent
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        self.checkpoint_folder = checkpoint_folder
        self.video_dir_name = video_folder + 'video' + str(seed) + video_prefix + dt_string
        self.record_test_video = record_test_video

        # wraps test environment with recordvideo wraper if record_test_video is true
        def get_test_env_wrapper(env_num):
            video_dir_name = self.video_dir_name
            if self.record_test_video:
                video_dir_name = self.video_dir_name + '/env_id_' + str(env_num)
                test_env_wrapper = lambda x: RecordVideo(x, video_folder=video_dir_name,
                                                         episode_trigger=lambda x: True)
            else:
                test_env_wrapper = lambda x: x
            return test_env_wrapper, video_dir_name

        # Loop over test envs and store them.
        if test_env is None:
            test_env_wrapper, video_dir_name = get_test_env_wrapper(env_num=0)
            self.test_env = [
                test_env_wrapper(deepcopy(self.env.envs[0]))
            ]
            self.video_dirs = [video_dir_name]
        else:
            self.test_env = []
            self.video_dirs = []
            if isinstance(test_env, list):
                for env_id, env in enumerate(test_env):
                    test_env_wrapper, video_dir_name = get_test_env_wrapper(env_num=env_id)
                    self.video_dirs.append(video_dir_name)
                    if isinstance(env, VecEnv):
                        for e in env.envs:
                            self.test_env.append(
                                test_env_wrapper(e)
                            )
                    else:
                        self.test_env.append(
                            test_env_wrapper(env)
                        )
            else:
                test_env_wrapper, video_dir_name = get_test_env_wrapper(env_num=0)
                self.video_dirs = [video_dir_name]
                if isinstance(test_env, VecEnv):
                    for e in test_env.envs:
                        self.test_env.append(
                            test_env_wrapper(e)
                        )
                else:
                    self.test_env.append(
                        test_env_wrapper(test_env)
                    )
        assert len(self.test_env) == self.num_reward_models, "number of reward models must be the same as number of " \
                                                             "envs"

    def train(self):
        pass

    def _get_state(self) -> DummyTrainerState:
        return DummyTrainerState(
            buffer=self.buffer,
            agent=self.agent,
            rng=self.rng,
            eval_episodes=self.eval_episodes,
            agent_name=self.agent_name,
            total_train_steps=self.total_train_steps,
            train_freq=self.train_freq,
            eval_freq=self.eval_freq,
            exploration_steps=self.exploration_steps,
            rollout_steps=self.rollout_steps,
            validate=self.validate,
            record_test_video=self.record_test_video,
            video_dir_name=self.video_dir_name
        )
    
    def _set_state(self, state: DummyTrainerState):
        # NOTE: does not set env, this must be passed at initialization
        self.buffer = state.buffer
        self.agent = state.agent
        self.rng = state.rng
        self.eval_episodes = state.eval_episodes
        self.agent_name = state.agent_name
        self.total_train_steps = state.total_train_steps
        self.train_freq = state.train_freq
        self.eval_freq = state.eval_freq
        self.exploration_steps = state.exploration_steps
        self.rollout_steps = state.rollout_steps
        self.eval_episodes = state.eval_episodes
        self.validate = state.validate
        self.record_test_video = state.record_test_video
        self.video_dir_name = state.video_dir_name

    def save_agent(self, step=0, agent_name=None):
        training_state = self._get_state()
        if agent_name is None:
            agent_name = "agent"
        file_name = f"{agent_name}_step_{step}.pkl"
        file_path = str(Path(self.checkpoint_folder)/file_name)
        cloudpickle.dump(training_state, open(file_path, 'wb'))

    def load_agent(self, agent_directory: str):
        training_state = cloudpickle.load(open(agent_directory, 'rb'))
        self._set_state(training_state)

    def set_total_trainer_steps(self, n: int):
        self.total_train_steps = n

    def step_env(self, obs: Union[jax.Array, np.ndarray], policy: Callable, num_steps: int,
                 rng: jax.random.PRNGKeyArray) -> [Transition, Union[jax.Array, np.ndarray],
                                                   Union[jax.Array, np.ndarray, bool]]:
        """
        Step in the environment starting at initial state obs. Env is not reset.
        :param obs: Last observation in the environment
        :param policy: Policy function
        :param num_steps: Num steps for rollout
        :param rng: random key for rollout
        :return: transitions, last_obs, last_done
        """
        self.agent.prepare_agent_for_rollout()
        rng, reset_rng = jax.random.split(rng, 2)
        num_points = int(num_steps * self.num_envs)
        obs_shape = (num_points,) + self.env.observation_space.shape
        action_space = (num_points,) + self.env.action_space.shape
        obs_vec = np.zeros(obs_shape)
        action_vec = np.zeros(action_space)
        reward_vec = np.zeros((num_points,))
        next_obs_vec = np.zeros(obs_shape)
        done_vec = np.zeros((num_points,))
        next_rng = rng
        last_obs = obs
        last_done = False
        for step in range(num_steps):
            next_rng, actor_rng = jax.random.split(next_rng, 2)
            action = policy(obs, actor_rng)
            next_obs, reward, terminate, truncate, info = self.env.step(action)

            obs_vec[step * self.num_envs: (step + 1) * self.num_envs] = obs
            action_vec[step * self.num_envs: (step + 1) * self.num_envs] = action
            reward_vec[step * self.num_envs: (step + 1) * self.num_envs] = reward.reshape(-1)
            next_obs_vec[step * self.num_envs: (step + 1) * self.num_envs] = next_obs
            done_vec[step * self.num_envs: (step + 1) * self.num_envs] = terminate.reshape(-1)
            obs = np.concatenate([x['current_env_state'].reshape(1, -1) for x in info], axis=0)
            dones = np.concatenate([np.asarray(x['last_done']).reshape(1, -1) for x in info], axis=0)

            last_obs = obs
            last_done = dones
        transitions = Transition(
            obs=obs_vec,
            action=action_vec,
            reward=reward_vec,
            next_obs=next_obs_vec,
            done=done_vec,
        )
        return transitions, last_obs, last_done

    def rollout_policy(self, num_steps: int, policy: Callable, rng: jax.random.PRNGKeyArray) -> Transition:
        """
        rollout a policy in the env. Env is reset before and after the rollout.
        :param num_steps: int, number of steps for rollout
        :param policy: Callable
        :param rng: jax.random.PRNGKeyArray
        :return: transition: Transition from rollout
        """
        self.agent.prepare_agent_for_rollout()
        rng, reset_rng = jax.random.split(rng, 2)
        reset_seed = jax.random.randint(
            reset_rng,
            (1,),
            minval=0,
            maxval=num_steps).item()
        obs, _ = self.env.reset(seed=reset_seed)
        num_points = int(num_steps * self.num_envs)
        obs_shape = (num_points,) + self.env.observation_space.shape
        action_space = (num_points,) + self.env.action_space.shape
        obs_vec = np.zeros(obs_shape)
        action_vec = np.zeros(action_space)
        reward_vec = np.zeros((num_points,))
        next_obs_vec = np.zeros(obs_shape)
        done_vec = np.zeros((num_points,))
        next_rng = rng
        for step in range(num_steps):
            next_rng, actor_rng = jax.random.split(next_rng, 2)
            action = policy(obs, actor_rng)
            next_obs, reward, terminate, truncate, info = self.env.step(action)

            obs_vec[step * self.num_envs: (step + 1) * self.num_envs] = obs
            action_vec[step * self.num_envs: (step + 1) * self.num_envs] = action
            reward_vec[step * self.num_envs: (step + 1) * self.num_envs] = reward.reshape(-1)
            next_obs_vec[step * self.num_envs: (step + 1) * self.num_envs] = next_obs
            done_vec[step * self.num_envs: (step + 1) * self.num_envs] = terminate.reshape(-1)
            # obs_vec = obs_vec.at[step].set(jnp.asarray(obs))
            # action_vec = action_vec.at[step].set(jnp.asarray(action))
            # reward_vec = reward_vec.at[step].set(jnp.asarray(reward))
            # next_obs_vec = next_obs_vec.at[step].set(jnp.asarray(next_obs))
            # done_vec = done_vec.at[step].set(jnp.asarray(terminate))
            obs = np.concatenate([x['current_env_state'].reshape(1, -1) for x in info], axis=0)
            # for idx, done in enumerate(dones):
            #    if done:
            #        reset_rng, next_reset_rng = jax.random.split(reset_rng, 2)
            #        reset_seed = jax.random.randint(
            #            reset_rng,
            #            (1,),
            #            minval=0,
            #            maxval=num_steps).item()
            #        obs[idx], _ = self.env.reset(seed=reset_seed)

        transitions = Transition(
            obs=obs_vec,
            action=action_vec,
            reward=reward_vec,
            next_obs=next_obs_vec,
            done=done_vec,
        )
        reset_seed = jax.random.randint(
            reset_rng,
            (1,),
            minval=0,
            maxval=num_steps).item()
        obs, _ = self.env.reset(seed=reset_seed)
        return transitions

    def eval_policy(self, step: int = 0, rng: Optional[jax.random.PRNGKeyArray] = None) -> dict:
        """
        Evaluates policy in the environment till it is done.

        :param step: Current step for evaluation
        :param rng: key
        :return reward_log: results log from evaluation
        """
        reward_log = {

        }
        self.agent.prepare_agent_for_rollout()
        for i, env in enumerate(self.test_env):
            pbar = tqdm()
            avg_reward = 0.0
            for e in range(self.eval_episodes):
                obs, _ = env.reset(seed=e)
                done = False
                steps = 0
                while not done:
                    action = self.agent.act(obs, rng=rng, eval=True, eval_idx=i)
                    next_obs, reward, terminate, truncate, info = env.step(action)
                    done = terminate or truncate
                    avg_reward += reward
                    obs = next_obs
                    steps += 1
                    pbar.update(1)
                    # print(steps)
                    if done:
                        obs, _ = env.reset(seed=e)
            avg_reward /= self.eval_episodes
            reward_log['reward_task_' + str(i)] = avg_reward
            pbar.close()
            if self.use_wandb and self.record_test_video:
                mp4list = glob.glob(self.video_dirs[i] + '/*.mp4')
                if len(mp4list) > 0:
                    mp4 = mp4list[-1]
                    # log gameplay video in wandb
                    wandb.log({"gameplays":
                                   wandb.Video(mp4, caption='episode/step: ' + str(step) + '/task:' + str(i),
                                               fps=4, format="gif"),
                               "step": step})
        return reward_log

    @property
    def num_reward_models(self):
        return 1
