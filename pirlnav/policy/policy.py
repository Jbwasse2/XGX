#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
import os
import pickle
import sys

import cv2
import numpy as np
import open3d as o3d
import pudb
import torch
from habitat import logger
from habitat.tasks.nav.object_nav_task import (
    mapping_mpcat40_to_goal21,
    task_cat2mpcat40,
)
from habitat_baselines.il.env_based.policy.rednet import load_rednet
from habitat_baselines.il.env_based.validity_func.local_nav import (
    LocalAgent,
    loop_nav,
    loop_nav_action,
)
from habitat_baselines.rl.ppo import Policy
from habitat_baselines.utils.common import CategoricalNet
from torch import nn as nn

try:
    sys.path.insert(0, "Detic/third_party/CenterNet2/")
    from centernet.config import add_centernet_config
    from detectron2.config import get_cfg
    from detectron2.engine.defaults import DefaultPredictor
    from detic import (
        get_semantics_for_img,
        get_semantics_for_imgs,
        semantics_results_to_image,
    )
    from detic_folder.config import add_detic_config
    from detic_folder.predictor import VisualizationDemo
except Exception as e:
    pass


def setup_cfg_detic(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE = "cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
        args.confidence_threshold)
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"  # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


class ILPolicy(nn.Module, Policy):
    def __init__(
        self,
        net,
        dim_actions,
        no_critic=False,
        mlp_critic=False,
        critic_hidden_dim=512,
    ):
        super().__init__()
        self.config = None
        self.rednet = None
        self.debug_count = 0
        self.net = net
        self.dim_actions = dim_actions
        self.no_critic = no_critic
        try:
            # f = open("detic_args.pkl", "rb")
            # args = pickle.load(f)
            # f.close()
            # args.cpu = False
            cfg = setup_cfg_detic(args)
            self.detic = VisualizationDemo(cfg, args)
            self.detic.predictor.model.eval()
        except Exception as e:
            print("no detic")

        self.action_distribution = CategoricalNet(self.net.output_size,
                                                  self.dim_actions)
        if self.no_critic:
            self.critic = None
        else:
            if not mlp_critic:
                self.critic = CriticHead(self.net.output_size)
            else:
                self.critic = MLPCriticHead(
                    self.net.output_size,
                    critic_hidden_dim,
                )

    def forward(self, *x):
        features, rnn_hidden_states = self.net(*x)
        distribution = self.action_distribution(features)
        distribution_entropy = distribution.entropy().mean()

        return distribution.logits, rnn_hidden_states, distribution_entropy

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
        return_distribution=False,
        LMN=False,
        LMN_LOSS_IGNORE=False,
    ):
        device = observations["rgb"].device
        self.device = device
        features, rnn_hidden_states = self.net(observations, rnn_hidden_states,
                                               prev_actions, masks)
        distribution = self.action_distribution(features)
        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()
        if self.config.EXPLORE_ONLY:
            action = torch.argmax(distribution.probs[:, 1:], dim=1) + 1
            action = action.unsqueeze(-1)
            LMN = False
        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        if self.no_critic:
            return action, rnn_hidden_states

        value = self.critic(features)
        if LMN:
            (
                value,
                action,
                action_log_probs,
                rnn_hidden_states,
            ) = self.last_mile_navigation(
                value,
                action,
                action_log_probs,
                rnn_hidden_states,
                observations,
                LMN_LOSS_IGNORE,
            )
        if return_distribution:
            return (
                value,
                action,
                action_log_probs,
                rnn_hidden_states,
            )
        return (
            value,
            action,
            action_log_probs,
            rnn_hidden_states,
        )

    def localnav(self, rho, phi, observation, steps_left, env):
        curr_pos = observation["gps"].cpu().numpy()
        curr_rot = observation["compass"].cpu().numpy()
        try:
            local_agent = LocalAgent(
                curr_pos,
                curr_rot,
                map_size_cm=1200,
                map_resolution=5,
            )
            action, terminate = loop_nav_action(
                env,
                local_agent,
                curr_pos,
                curr_rot,
                rho,
                phi,
                1,
                #    min(1, 499 - steps_left.item()),
                observation,
                self.config.LMN.PHI,
                self.config.LMN.RESTART_LOOP_POSE,
            )
            if terminate:
                return 0
            else:
                return action
        except:
            return None

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(observations, rnn_hidden_states, prev_actions,
                               masks)
        return self.critic(features)

    def evaluate_actions(self, observations, rnn_hidden_states, prev_actions,
                         masks, action):
        features, rnn_hidden_states = self.net(observations, rnn_hidden_states,
                                               prev_actions, masks)
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy()  # .mean()

        return (
            value,
            action_log_probs,
            distribution_entropy,
            rnn_hidden_states,
        )

    def get_semantic_mask_detic(self, rgbs):
        imgs = []
        semantics, shape = get_semantics_for_imgs(self.detic, rgbs)
        img = semantics_results_to_image(semantics,
                                         self.config.LMN.THRESHOLD_CUTOFF,
                                         shape,
                                         habitat=True)
        return img

    def last_mile_navigation(
        self,
        value,
        actions,
        action_log_probs,
        rnn_hidden_states,
        observations,
        LMN_LOSS_IGNORE,
    ):
        if isinstance(self.rednet, type(None)):
            print("REDNET")
            print(self.device)
            self.rednet = load_rednet(
                self.device,
                "models/hm3d_rednet.pt",
                resize=True,
                num_classes=7,
            )
            self.rednet.eval()
        if (self.config.LMN.PREDICT_SEMANTICS_REDNET
                and not "semantic" in observations[0]):
            #            pu.db
            temp = self.rednet(
                observations["rgb"],
                observations["depth"],
                self.config.LMN.REDNET_THRESHOLD,
            )
            temp = temp.unsqueeze(-1)
            replacements = {key: 29 for key in range(31)}
            replacements.update({
                0: 0,
                1: 1,
                2: 2,
                3: 3,
                4: 4,
                5: 5,
            })
            #            replacements.update({
            #                2: 0,
            #                8: 1,
            #                10: 2,
            #                12: 3,
            #                15: 4,
            #                7: 5,
            #            })
            for key, v in replacements.items():
                temp[temp == key] = v
            observations["semantic"] = temp
        if not hasattr(self, "lmn"):
            self.lmn = [0] * observations["semantic"].shape[0]

            self.lmn_debug = [
                (0, 0) for _ in range(observations["semantic"].shape[0])
            ]
            self.lmn_dataset = [(
                None,
                None,
                None,
            ) for _ in range(observations["semantic"].shape[0])]
            self.save_start = [
                False for _ in range(observations["semantic"].shape[0])
            ]
        for i in range(len(self.lmn)):
            if self.config.LMN.SKIP_STOP_EXPLORE:
                try:
                    if actions[i] == 0:
                        actions[i] = 2
                except Exception as e:
                    continue
            self.save_start[i] = False
            self.lmn_dataset[i] = (None, None, None)
            if self.lmn[i] == 1:
                continue
            try:
                observation = observations[i]
            except Exception as e:
                continue
            semantic_obs = observation["semantic"]

            #            goal = task_cat2mpcat40[observation['objectgoal']]
            #            goal = mapping_mpcat40_to_goal21[observation['objectgoal'].item()]
            goal = observation["objectgoal"].item()
            #            print(goal)
            semantic_mask = torch.isin(semantic_obs, goal)
            fraction_of_semantics_is_goal = torch.sum(semantic_mask) / (
                semantic_obs.shape[0] * semantic_obs.shape[1])
            # Is there enough of the object in the view?
            # If No, keep explore
            # If Yes, LMN
            if (fraction_of_semantics_is_goal >=
                    self.config.LMN.FRACTION_THRESHOLD).item():
                self.lmn[i] = 1
                self.save_start[i] = True
                if self.config.COLLECT_DATASET:
                    self.lmn_dataset[i] = (
                        observation["rgb"],
                        observation["depth"],
                        observation["semantic"],
                        observation["objectgoal"],
                    )
        # LMN code
        for i in range(len(self.lmn)):
            if self.lmn[i] == 0:
                continue
            # Last Mile Navigaiton!
            try:
                observation = observations[i]
            except Exception as e:
                continue
            semantic_obs = observation["semantic"]
            #            goal = task_cat2mpcat40[observation['objectgoal']]
            #            goal = mapping_mpcat40_to_goal21[observation['objectgoal'].item()]
            goal = observation["objectgoal"].item()
            semantic_mask = torch.isin(semantic_obs, goal)
            fraction_of_semantics_is_goal = torch.sum(semantic_mask) / (
                semantic_obs.shape[0] * semantic_obs.shape[1])
            if (fraction_of_semantics_is_goal <= 0.0001).item():
                if self.config.COLLECT_DATASET:
                    actions[i] = 0
                    self.lmn[i] = 0
                else:
                    self.lmn[i] = 0
                continue
            # Depth is normalized in habitat to unnormalize it multiply by 10!
            depth_image = observation["depth"] * 10
            depth_image[~semantic_mask] = 9001  # big number wow
            depth_image[depth_image ==
                        0.0] = 9001  # 0 distance is NaN basically
            flattened_tensor = depth_image.flatten()
            mask = flattened_tensor != 9001
            sorted_tensor, _ = torch.sort(flattened_tensor[mask])
            if self.config.LMN.INDEX:
                try:
                    index = int(self.config.LMN.INDEX_FRAC *
                                (len(sorted_tensor)))
                except Exception as e:
                    index = 0
            else:
                index = 0
            try:
                rho = sorted_tensor[index]
            except Exception as e:
                self.lmn[i] = 0
                if self.config.COLLECT_DATASET:
                    actions[i] = 0
                continue
            image_waypoint = divmod(depth_image.argmin().item(),
                                    depth_image.shape[1])
            depth_image = observation["depth"] * 10
            depth = depth_image[image_waypoint[0], image_waypoint[1]][0]
            # width, height, fx, fy, cx, cy
            # 79 degree hfov
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                semantic_mask.shape[1],
                semantic_mask.shape[0],
                388,
                291,
                semantic_mask.shape[1] / 2,
                semantic_mask.shape[0] / 2,
            )
            # TOP LEFT CORNER IS 0,0
            x = ((image_waypoint[1] - intrinsic.get_principal_point()[0]) *
                 depth / intrinsic.get_focal_length()[0])
            y = ((image_waypoint[0] - intrinsic.get_principal_point()[1]) *
                 depth / intrinsic.get_focal_length()[1])
            z = depth
            forward_vector = np.array([0, 1])  # x, z
            point_vector = np.array([x.cpu().item(), z.cpu().item()])
            point_vector = point_vector / np.linalg.norm(point_vector)
            phi = np.arctan2(forward_vector[1],
                             forward_vector[0]) - np.arctan2(
                                 point_vector[1], point_vector[0])
            sign = np.sign(np.cross(point_vector, forward_vector))
            phi = sign * np.arccos(
                np.dot(point_vector, forward_vector) /
                (np.linalg.norm(point_vector) *
                 np.linalg.norm(forward_vector)))
            if rho > self.config.LMN.MAX_DISTANCE:
                if self.config.COLLECT_DATASET:
                    actions[i] = 0
                    self.lmn[i] = 0
                else:
                    self.lmn[i] = 0
                continue
            if rho < self.config.LMN.STOP_DISTANCE and rho > self.config.LMN.STOP_MIN:
                actions[i] = 0
                if self.config.COLLECT_DATASET:
                    actions[i] = 0
                    self.lmn[i] = 0
                else:
                    self.lmn[i] = 0
                continue
            if self.config.LMN.FLIP_PHI:
                phi = -phi
            else:
                phi = phi
            rho = max(rho - self.config.LMN.SHAVE_DISTANCE, 0)
            action = self.localnav(rho, phi, observation, 100, None)
            if isinstance(rho, torch.Tensor):
                rho = rho.item()
            #            print(self.lmn)
            #            print(rho,phi)
            #            print(action)
            #            if self.debug_count == 0:
            #            if 1 == 1:
            #                f = open("./debug/semantic.pkl", "wb")
            #                pickle.dump(observation['semantic'].cpu().numpy(), f)
            #                f.close()
            #                f = open("./debug/semanticGoal.pkl", "wb")
            #                pickle.dump(observation['objectgoal'].cpu().item(), f)
            #                f.close()
            #            loc = "./debug/" + str(self.debug_count).zfill(5) + ".png"
            #            img = observations['rgb'].cpu().numpy()[i]
            #            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #            cv2.imwrite(loc, img)
            if isinstance(action, type(None)):
                if self.config.COLLECT_DATASET:
                    actions[i] = 0
                    self.lmn[i] = 0
                continue
            else:
                actions[i] = action
            self.debug_count += 1
            if action == 0:
                self.lmn[i] = 0
        if LMN_LOSS_IGNORE:
            for i in range(len(self.lmn)):
                if self.lmn[i] != 0:
                    value[i] = -123.0
        return value, actions, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(observations, rnn_hidden_states, prev_actions,
                               masks)
        return self.critic(features)

    def evaluate_actions(self, observations, rnn_hidden_states, prev_actions,
                         masks, action):
        features, rnn_hidden_states = self.net(observations, rnn_hidden_states,
                                               prev_actions, masks)
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy()  # .mean()

        return (
            value,
            action_log_probs,
            distribution_entropy,
            rnn_hidden_states,
        )

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config, observation_space, action_space):
        pass


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class MLPCriticHead(nn.Module):
    def __init__(self, input_size, hidden_dim=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.orthogonal_(self.fc[0].weight)
        nn.init.constant_(self.fc[0].bias, 0)

        nn.init.orthogonal_(self.fc[2].weight)
        nn.init.constant_(self.fc[2].bias, 0)

    def forward(self, x):
        return self.fc(x.detach())
