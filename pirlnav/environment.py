import pdb
from typing import Optional

import habitat
import numpy as np
import torch
import torch.nn as nn
from habitat import Config, Dataset
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from torchvision import transforms


@habitat.registry.register_env(name="SimpleRLEnv")
class SimpleRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config, dataset)

    def reset(self):
        observations = super().reset()
        return observations

    def step(self, *args, **kwargs):
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        return (-np.inf, np.inf)

    def get_reward(self, observations):
        return self._env.get_metrics()[self.config.TASK.SUCCESS_MEASURE]

    def get_done(self, observations):
        if self._env.episode_over:
            return True
        if self._env.get_metrics()[self.config.TASK.SUCCESS_MEASURE]:
            return True
        return False

    def get_info(self, observations):
        return self._env.get_metrics()
