
import commentjson
from ray.rllib.agents.registry import get_agent_class
import amr_env
import gym
from gym import spaces
import ray
import ray.rllib.agents.ppo as ppo
import tensorflow as tf

import numpy as np


import argparse
import random
from amr import models
import pytest
import os
import copy
from math import sqrt, nan, inf, isnan
from ray.rllib.models import ModelCatalog
from amr.models.cnn import CNNSmall

# rllib requires you to give the policy an env with the same action
# and observation spaces as used in training. The rest of it can be
# "fake" if you provide your own observation data some other way.

# USER INPUT
checkpoint_folder = "/p/lustre1/mittal3/local_deref/PPO/PPO_LocalAMR-v0_5b99d_00000_0_2021-06-07_12-12-52/"
checkpoint_number = 900
# END OF USER INPUT

# Read info from json file used for training.
full_checkpoint_path = checkpoint_folder + '/checkpoint_' + str(checkpoint_number)+ '/checkpoint-' + str(checkpoint_number)
path_env_config_file = checkpoint_folder + '/params.json'
with open(path_env_config_file) as json_file:
    env_trainer_config = commentjson.load(json_file)

trainer_config = env_trainer_config
trainer_config['env_config']['mesh_params']['nx'] = 1
trainer_config['env_config']['mesh_params']['ny'] = 1
local_sample = env_trainer_config['env_config']['local_sample']
local_context = env_trainer_config['env_config']['local_context']
reward_params = env_trainer_config['env_config']['reward_function_params']

#set some default params
observe_error = False
observe_values = True

#get observing quantities
observe_depth = env_trainer_config['env_config']['observe_depth']
observe_jacobian = env_trainer_config['env_config']['observe_jacobian']
observe_ar = env_trainer_config['env_config']['observe_ar']
observe_grads = env_trainer_config['env_config']['observe_grads']
normalization = env_trainer_config['env_config']['normalization']

#get reward_params
reward_params = env_trainer_config['env_config']['reward_function_params']
reward_params_name = reward_params['name']
if reward_params_name == "random_penalized_norm_diff":
    observe_error = True
if reward_params_name == "random_binary":
    observe_error = True

class DummyEnv(gym.Env):

    def __init__(self, config): # the config param is required by rllib

        # image size is 42x42 (a size which uses CNN by default in rllib)
        self.obsx = local_sample+2*local_context
        self.obsy = self.obsx

        # Either do nothing (0) or refine (1)
        self.action_space = spaces.Discrete(2)

        n_channels = 1
        n_channels = observe_values + observe_depth + observe_grads

        low = -np.inf
        high = np.inf
        self.observation_space = spaces.Dict({
                "scalar_info": spaces.Box(low=low, high=high, shape=(1 + observe_jacobian + observe_error, ), dtype=np.float32),
                "obs_data":    spaces.Box(low=low, high=high,
                shape=(self.obsx,
                       self.obsy,
                       n_channels), dtype=np.float32)
                })

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

        ModelCatalog.register_custom_model("cnn_small", CNNSmall)
        trainer = ppo.PPOTrainer(config=trainer_config,env=DummyEnv)
        trainer.restore(full_checkpoint_path)

        trainer_config['evaluation_num_workers'] = 1
        trainer_config['evaluation_interval'] = 0
        trainer_config['num_workers'] = 0
        trainer_config['num_envs_per_worker'] = 1

        self.agent = ppo.PPOTrainer(config=trainer_config,env=DummyEnv)
        self.agent.restore(full_checkpoint_path)
        
        self.env = DummyEnv({})

    def eval(self,obso,scalar):
        obso -= np.mean(obso)
        obs = {
            "obs_data" : obso,
            "scalar_info" : scalar
            }   
        pick = self.agent.compute_action(obs, explore=False)
        return pick

    def get_local_sample(self):
        return local_sample

    def get_local_context(self):
        return local_context

    def get_observe_error(self):
        return observe_error

    def get_observe_jacobian(self):
        return observe_jacobian