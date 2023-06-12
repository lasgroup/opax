from jax import random
import wandb
from tqdm import tqdm
from mbse.trainer.dummy_trainer import DummyTrainer


class ModelFreeTrainer(DummyTrainer):
    def __init__(self,
                 agent_name: str = "ModelFreeAgent",
                 *args,
                 **kwargs
                 ):
        
        super(ModelFreeTrainer, self).__init__(agent_name=agent_name, *args, **kwargs)

    def train(self):
        if self.use_wandb:
            wandb.define_metric('env_steps')
            wandb.define_metric('train_steps')
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
        policy = lambda x, y: self.env.action_space.sample()
        transitions = self.rollout_policy(self.exploration_steps, policy, self.rng)
        self.buffer.add(transition=transitions)
        rng_keys = random.split(self.rng, self.total_train_steps+1)
        self.rng = rng_keys[0]
        rng_keys = rng_keys[1:]
        learning_steps = int(self.total_train_steps/(self.rollout_steps*self.num_envs))
        for step in tqdm(range(learning_steps)):
            actor_rng, train_rng = random.split(rng_keys[step], 2)
            policy = self.agent.act_in_train
            transitions = self.rollout_policy(self.rollout_steps, policy, actor_rng)
            self.buffer.add(transitions)

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










