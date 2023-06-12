from jax import random
import wandb
from tqdm import tqdm
from mbse.trainer.dummy_trainer import DummyTrainer
from mbse.agents.actor_critic.sac import SACAgent
import numpy as np


class OffPolicyTrainer(DummyTrainer):
    def __init__(self,
                 agent_name: str = "OffPolicyAgent",
                 *args,
                 **kwargs
                 ):

        super(OffPolicyTrainer, self).__init__(agent_name=agent_name, *args, **kwargs)
        assert isinstance(self.agent, SACAgent), "Only Off policy agents are allowed"

    def train(self):
        if self.use_wandb:
            wandb.define_metric('env_steps')
        self.rng, eval_rng = random.split(self.rng, 2)
        eval_rng, curr_eval = random.split(eval_rng, 2)
        reward_log = self.eval_policy(rng=curr_eval)
        best_performance = reward_log['reward_task_0']
        reward_log['env_steps'] = 0
        reward_log['learning_step'] = 0
        reward_log['train_steps'] = 0
        train_steps = 0
        self.save_agent(0)
        if self.use_wandb:
            wandb.define_metric("env_steps")
            wandb.define_metric("train_steps")
            wandb.log(reward_log)
        policy = lambda x, y: np.concatenate([self.env.action_space.sample().reshape(1, -1)
                                              for s in range(self.num_envs)], axis=0)
        transitions = self.rollout_policy(self.exploration_steps, policy, self.rng)
        self.buffer.add(transition=transitions)
        rng_keys = random.split(self.rng, self.total_train_steps + 1)
        self.rng = rng_keys[0]
        rng_keys = rng_keys[1:]
        learning_steps = int(self.total_train_steps/(self.rollout_steps*self.num_envs))
        rng_key, reset_rng = random.split(rng_keys[0], 2)
        rng_keys = rng_keys.at[0].set(rng_key)
        reset_seed = random.randint(
            reset_rng,
            (1,),
            minval=0,
            maxval=int(learning_steps*self.rollout_steps)).item()
        obs, _ = self.env.reset(seed=reset_seed)
        for step in tqdm(range(learning_steps)):
            actor_rng, train_rng = random.split(rng_keys[step], 2)
            policy = self.agent.act_in_train
            transitions, obs, done = self.step_env(obs, policy, self.rollout_steps, actor_rng)
            self.buffer.add(transitions)
            #    reset_rng, next_reset_rng = random.split(reset_rng, 2)
            #    reset_seed = random.randint(
            #        next_reset_rng,
            #        (1,),
            #        minval=0,
            #        maxval=int(learning_steps * self.rollout_steps)).item()
            #    obs, _ = self.env.reset(seed=reset_seed)
            # transitions = self.rollout_policy(self.rollout_steps, policy, actor_rng)
            if self.use_wandb:
                wandb.log({'learning_steps':  step,
                           'env_steps': step * self.rollout_steps * self.num_envs,
                           })
            if step % self.train_freq == 0:
                train_rng, agent_rng = random.split(train_rng, 2)
                total_train_steps = self.agent.train_step(
                    rng=agent_rng,
                    buffer=self.buffer,
                    validate=self.validate,
                    log_results=self.use_wandb,
                )
                train_steps += total_train_steps
            # Evaluate episode
            if train_steps % self.eval_freq == 0:
                eval_rng, curr_eval = random.split(eval_rng, 2)
                reward_log = self.eval_policy(rng=curr_eval, step=train_steps)
                if self.use_wandb:
                    wandb.log(reward_log)
                if reward_log['reward_task_0'] > best_performance:
                    best_performance = reward_log['reward_task_0']
                    self.save_agent(step)

            step += 1
        self.save_agent(step, agent_name="final_agent")










