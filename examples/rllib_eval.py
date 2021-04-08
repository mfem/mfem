

import gym
from gym import spaces
import ray
import ray.rllib.agents.ppo as ppo
import tensorflow as tf

import numpy as np

# rllib requires you to give the policy an env with the same action
# and observation spaces as used in training. The rest of it can be
# "fake" if you provide your own observation data some other way.

class DummyEnv(gym.Env):

    def __init__(self, config): # the config param is required by rllib

        # image size is 42x42 (a size which uses CNN by default in rllib)
        self.obsx = 42
        self.obsy = 42

        # Either do nothing (0) or refine (1)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-1.0, 2.0, shape=(self.obsx,self.obsy,1))

        self.state = None
        
    def step(self, action):
        pass
    def reset(self):
        pass
    def render(self):
        pass

class Evaluator():

    def __init__(self):

        print("starting ray...")
        ray.shutdown()
        ray.init()
        print("ray up...")

        config = ppo.DEFAULT_CONFIG.copy()
        config["log_level"] = "WARN"

        # Create agent from checkpoint
        self.agent = ppo.PPOTrainer(config,env=DummyEnv)
#        self.agent.restore("DRLRefinePolicy/checkpoint_115/checkpoint-115")
        self.agent.restore("PPO_5x5_H1_thresh1.e-4/checkpoint_485/checkpoint-485")

        self.env = DummyEnv({})

    def eval(self,obs):
        pick = self.agent.compute_action(obs, explore = False)
        return pick

#evaluator = Evaluator()
#obs = np.ones((42,42,1))
#ref = evaluator.eval(obs)
#evaluator.show_logits(obs)
# evaluator.eval(np.random.rand(8))
# evaluator.eval(np.random.rand(8))

