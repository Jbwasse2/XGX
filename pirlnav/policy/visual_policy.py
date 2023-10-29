import torch
from gym.spaces import Box
from typing import Dict
import time
import numpy as np
import pickle
import torch.nn as nn
from gym import Space
from habitat import Config, logger
from habitat.tasks.nav.nav import EpisodicCompassSensor, EpisodicGPSSensor
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import build_rnn_state_encoder
from habitat_baselines.rl.ppo import Net

from pirlnav.policy.policy import ILPolicy
from pirlnav.policy.transforms import get_transform
from pirlnav.policy.visual_encoder import VisualEncoder
from pirlnav.utils.utils import load_encoder
import sys                                                                                          
try:
    sys.path.insert(0, 'Detic/third_party/CenterNet2/')                                                 
    from detic import get_semantics_for_img, semantics_results_to_image, get_semantics_for_imgs
    from detic_folder.config import add_detic_config
    from detic_folder.predictor import VisualizationDemo
    from detectron2.config import get_cfg
    from detectron2.engine.defaults import DefaultPredictor
    from centernet.config import add_centernet_config
except Exception as e:
    pass
from habitat_baselines.il.common.encoders.resnet_encoders import (
    VlnResnetDepthEncoder,
    ResnetRGBEncoder,
    ResnetSemSeqEncoder,
)
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
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold          
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'  # load later                              
    if not args.pred_all_class:                                                                     
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True                                           
    cfg.freeze()
    return cfg     



class ObjectNavILMAENet(Net):
    r"""A baseline sequence to sequence network that concatenates instruction,
    RGB, and depth encodings before decoding an action distribution with an RNN.
    Modules:
        Instruction encoder
        Depth encoder
        RGB encoder
        RNN state encoder
    """

    def __init__(
        self,
        observation_space: Space,
        policy_config: Config,
        num_actions: int,
        run_type: str,
        hidden_size: int,
        rnn_type: str,
        num_recurrent_layers: int,
    ):
        super().__init__()
        self.policy_config = policy_config
        rnn_input_size = 0

        rgb_config = policy_config.RGB_ENCODER
        name = "resize"
        if rgb_config.use_augmentations and run_type == "train":
            name = rgb_config.augmentations_name
        if rgb_config.use_augmentations_test_time and run_type == "eval":
            name = rgb_config.augmentations_name
        self.visual_transform = get_transform(name, size=rgb_config.image_size)
        self.visual_transform.randomize_environments = (
            rgb_config.randomize_augmentations_over_envs
        )
        self.visual_encoder = VisualEncoder(
            image_size=rgb_config.image_size,
            backbone=rgb_config.backbone,
            input_channels=3,
            resnet_baseplanes=rgb_config.resnet_baseplanes,
            resnet_ngroups=rgb_config.resnet_baseplanes // 2,
            avgpooled_image=rgb_config.avgpooled_image,
            drop_path_rate=rgb_config.drop_path_rate,
        )

        self.visual_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self.visual_encoder.output_size,
                policy_config.RGB_ENCODER.hidden_size,
            ),
            nn.ReLU(True),
        )

        rnn_input_size += policy_config.RGB_ENCODER.hidden_size
        logger.info(
            "RGB encoder is {}".format(policy_config.RGB_ENCODER.backbone)
        )

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[
                EpisodicGPSSensor.cls_uuid
            ].shape[0]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32
            logger.info("\n\nSetting up GPS sensor")

        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[
                    0
                ]
                == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding_dim = 32
            self.compass_embedding = nn.Linear(
                input_compass_dim, self.compass_embedding_dim
            )
            rnn_input_size += 32
            logger.info("\n\nSetting up Compass sensor")

        if ObjectGoalSensor.cls_uuid in observation_space.spaces:
            self._n_object_categories = (
                int(
                    observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]
                )
                + 1
            )
            logger.info(
                "Object categories: {}".format(self._n_object_categories)
            )
            self.obj_categories_embedding = nn.Embedding(
                self._n_object_categories, 32
            )
            rnn_input_size += 32
            logger.info("\n\nSetting up Object Goal sensor")

        if policy_config.SEQ2SEQ.use_prev_action:
            self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)
            rnn_input_size += self.prev_action_embedding.embedding_dim

        self.rnn_input_size = rnn_input_size

        # load pretrained weights
        if rgb_config.pretrained_encoder is not None:
            msg = load_encoder(
                self.visual_encoder, rgb_config.pretrained_encoder
            )
            logger.info(
                "Using weights from {}: {}".format(
                    rgb_config.pretrained_encoder, msg
                )
            )

        # freeze backbone
        if rgb_config.freeze_backbone:
            for p in self.visual_encoder.backbone.parameters():
                p.requires_grad = False

        logger.info(
            "State enc: {} - {} - {} - {}".format(
                rnn_input_size, hidden_size, rnn_type, num_recurrent_layers
            )
        )

        self.state_encoder = build_rnn_state_encoder(
            rnn_input_size,
            hidden_size=hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )
        self._hidden_size = hidden_size
        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind and self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """
#        observations['gps'] = observations['gps'] - observations['gps']
#        observations['compass'] = observations['compass'] - observations['compass']
        rgb_obs = observations["rgb"]

        N = rnn_hidden_states.size(1)

        x = []

        if self.visual_encoder is not None:
            if len(rgb_obs.size()) == 5:
                observations["rgb"] = rgb_obs.contiguous().view(
                    -1, rgb_obs.size(2), rgb_obs.size(3), rgb_obs.size(4)
                )
            # visual encoder
            rgb = observations["rgb"]

            rgb = self.visual_transform(rgb, N)
            rgb = self.visual_encoder(rgb)
            rgb = self.visual_fc(rgb)
            x.append(rgb)

        if EpisodicGPSSensor.cls_uuid in observations:
            obs_gps = observations[EpisodicGPSSensor.cls_uuid]
            if len(obs_gps.size()) == 3:
                obs_gps = obs_gps.contiguous().view(-1, obs_gps.size(2))
            x.append(self.gps_embedding(obs_gps))

        if EpisodicCompassSensor.cls_uuid in observations:
            obs_compass = observations["compass"]
            if len(obs_compass.size()) == 3:
                obs_compass = obs_compass.contiguous().view(
                    -1, obs_compass.size(2)
                )
            compass_observations = torch.stack(
                [
                    torch.cos(obs_compass),
                    torch.sin(obs_compass),
                ],
                -1,
            )
            compass_embedding = self.compass_embedding(
                compass_observations.float().squeeze(dim=1)
            )
            x.append(compass_embedding)

        if ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            if len(object_goal.size()) == 3:
                object_goal = object_goal.contiguous().view(
                    -1, object_goal.size(2)
                )
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        if self.policy_config.SEQ2SEQ.use_prev_action:
            prev_actions_embedding = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().view(-1)
            )
            x.append(prev_actions_embedding)

        x = torch.cat(x, dim=1)

        rnn_hidden_states = rnn_hidden_states.contiguous()
        x, rnn_hidden_states = self.state_encoder(
            x, rnn_hidden_states.contiguous(), masks
        )
        return x, rnn_hidden_states


@baseline_registry.register_policy
class ObjectNavILMAEPolicy(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        policy_config: Config,
        run_type: str,
        hidden_size: int,
        rnn_type: str,
        num_recurrent_layers: int,
    ):
        super().__init__(
            ObjectNavILMAENet(
                observation_space=observation_space,
                policy_config=policy_config,
                num_actions=action_space.n,
                run_type=run_type,
                hidden_size=hidden_size,
                rnn_type=rnn_type,
                num_recurrent_layers=num_recurrent_layers,
            ),
            action_space.n,
            no_critic=policy_config.CRITIC.no_critic,
            mlp_critic=policy_config.CRITIC.mlp_critic,
            critic_hidden_dim=policy_config.CRITIC.hidden_dim,
        )

    @classmethod
    def from_config(cls, config: Config, observation_space, action_space):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            policy_config=config.POLICY,
            run_type=config.RUN_TYPE,
            hidden_size=config.POLICY.STATE_ENCODER.hidden_size,
            rnn_type=config.POLICY.STATE_ENCODER.rnn_type,
            num_recurrent_layers=config.POLICY.STATE_ENCODER.num_recurrent_layers,
        )

    @property
    def num_recurrent_layers(self):
        return self.net.num_recurrent_layers

    def freeze_visual_encoders(self):
        for param in self.net.visual_encoder.parameters():
            param.requires_grad_(False)

    def unfreeze_visual_encoders(self):
        for param in self.net.visual_encoder.parameters():
            param.requires_grad_(True)

    def freeze_state_encoder(self):
        for param in self.net.state_encoder.parameters():
            param.requires_grad_(False)

    def unfreeze_state_encoder(self):
        for param in self.net.state_encoder.parameters():
            param.requires_grad_(True)

    def freeze_actor(self):
        for param in self.action_distribution.parameters():
            param.requires_grad_(False)

    def unfreeze_actor(self):
        for param in self.action_distribution.parameters():
            param.requires_grad_(True)



######################################
#############J########################

class ObjectNavILMAENetSemantic(Net):
    r"""A baseline sequence to sequence network that concatenates instruction,
    RGB, and depth encodings before decoding an action distribution with an RNN.
    Modules:
        Instruction encoder
        Depth encoder
        RGB encoder
        RNN state encoder
    """
    def get_semantic_mask_detic(self, rgbs):                                                         
        imgs = []
        semantics, shape = get_semantics_for_imgs(self.detic_demo, rgbs)                              
        img = semantics_results_to_image(semantics, 0.5, shape, habitat=True)                       
        return img

    def __init__(
        self,
        observation_space: Space,
        policy_config: Config,
        num_actions: int,
        run_type: str,
        hidden_size: int,
        rnn_type: str,
        num_recurrent_layers: int,
    ):
        super().__init__()
        self.policy_config = policy_config
        rnn_input_size = 0

        f = open("detic_args.pkl", "rb")
        args = pickle.load(f)                                                                           
        f.close()                                                                                       
        args.cpu = False
        #cfg = setup_cfg_detic(args)                                                                     
        #self.detic_demo = VisualizationDemo(cfg, args)
        #self.detic_demo.predictor.model.eval()
        #cfg = get_cfg()
        #cfg.merge_from_file(
        #    "configs/detectron/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        #)
#        cfg.MODEL.DEVICE = "cpu"                                                                                                                                                                                                                              
        #cfg.MODEL.WEIGHTS = "./model_final_f10217.pkl"
        #self.maskrcnn = DefaultPredictor(cfg)     
        rgb_config = policy_config.RGB_ENCODER
        name = "resize"
        if rgb_config.use_augmentations and run_type == "train":
            name = rgb_config.augmentations_name
        if rgb_config.use_augmentations_test_time and run_type == "eval":
            name = rgb_config.augmentations_name
        self.visual_transform = get_transform(name, size=rgb_config.image_size)
        self.visual_transform.randomize_environments = (
            rgb_config.randomize_augmentations_over_envs
        )

        self.visual_encoderSEM = VisualEncoder(
            image_size=rgb_config.image_size,
            backbone=rgb_config.backbone,
            input_channels=1,
            resnet_baseplanes=rgb_config.resnet_baseplanes,
            resnet_ngroups=rgb_config.resnet_baseplanes // 2,
            avgpooled_image=rgb_config.avgpooled_image,
            drop_path_rate=rgb_config.drop_path_rate,
        )
        self.visual_encoderDEPTH = VisualEncoder(
            image_size=rgb_config.image_size,
            backbone=rgb_config.backbone,
            input_channels=1,
            resnet_baseplanes=rgb_config.resnet_baseplanes,
            resnet_ngroups=rgb_config.resnet_baseplanes // 2,
            avgpooled_image=rgb_config.avgpooled_image,
            drop_path_rate=rgb_config.drop_path_rate,
        )
        self.visual_encoderRGB = VisualEncoder(
            image_size=rgb_config.image_size,
            backbone=rgb_config.backbone,
            input_channels=3,
            resnet_baseplanes=rgb_config.resnet_baseplanes,
            resnet_ngroups=rgb_config.resnet_baseplanes // 2,
            avgpooled_image=rgb_config.avgpooled_image,
            drop_path_rate=rgb_config.drop_path_rate,
        )

        self.visual_fcRGB = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self.visual_encoderRGB.output_size,
                policy_config.RGB_ENCODER.hidden_size,
            ),
            nn.ReLU(True),
        )
        self.visual_fcDEPTH = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self.visual_encoderRGB.output_size,
                policy_config.RGB_ENCODER.hidden_size,
            ),
            nn.ReLU(True),
        )
        self.visual_fcSEM = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self.visual_encoderRGB.output_size,
                policy_config.RGB_ENCODER.hidden_size,
            ),
            nn.ReLU(True),
        )
        rnn_input_size += 3*policy_config.RGB_ENCODER.hidden_size
        logger.info(
            "RGB encoder is {}".format(policy_config.RGB_ENCODER.backbone)
        )

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[
                EpisodicGPSSensor.cls_uuid
            ].shape[0]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32
            logger.info("\n\nSetting up GPS sensor")

        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[
                    0
                ]
                == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding_dim = 32
            self.compass_embedding = nn.Linear(
                input_compass_dim, self.compass_embedding_dim
            )
            rnn_input_size += 32
            logger.info("\n\nSetting up Compass sensor")

        if ObjectGoalSensor.cls_uuid in observation_space.spaces:
            self._n_object_categories = (
                int(
                    observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]
                )
                + 1
            )
            logger.info(
                "Object categories: {}".format(self._n_object_categories)
            )
            self.obj_categories_embedding = nn.Embedding(
                self._n_object_categories, 32
            )
            rnn_input_size += 32
            logger.info("\n\nSetting up Object Goal sensor")

        if policy_config.SEQ2SEQ.use_prev_action:
            self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)
            rnn_input_size += self.prev_action_embedding.embedding_dim

        self.rnn_input_size = rnn_input_size

        # load pretrained weights
        if rgb_config.pretrained_encoder is not None:
            msg = load_encoder(
                self.visual_encoder, rgb_config.pretrained_encoder
            )
            logger.info(
                "Using weights from {}: {}".format(
                    rgb_config.pretrained_encoder, msg
                )
            )

        # freeze backbone
        if rgb_config.freeze_backbone:
            for p in self.visual_encoder.backbone.parameters():
                p.requires_grad = False

        logger.info(
            "State enc: {} - {} - {} - {}".format(
                rnn_input_size, hidden_size, rnn_type, num_recurrent_layers
            )
        )

        self.state_encoder = build_rnn_state_encoder(
            rnn_input_size,
            hidden_size=hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )
        self._hidden_size = hidden_size
        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind and self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_semantic_mask_maskrcnn(self, rgb):
        #im = np.zeros((480, 640)).astype(np.uint8)
        # Bed = 11, chair = 3, couch = 10, plant = 14, toilet = 18, tv = 22
        update_label = {0: 11, 1: 3, 2: 10, 3: 14, 4: 18, 5: 22}
        res = self.maskrcnn(rgb)
        val = list(self.pred_to_score(res))
        for i in range(6):
            if isinstance(val[i][0], type(None)):
                continue
            label_new = update_label[i]
            box = val[i][0]
            value = val[i][1]
            mask = val[i][2]
            for m, v in zip(mask, value):
                if v > 0.00:
                    im[m] = label_new
        return im


    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """
#        observations['gps'] = observations['gps'] - observations['gps']
#        observations['compass'] = observations['compass'] - observations['compass']
        start = time.time() 
        rgb_obs = observations["rgb"]

        N = rnn_hidden_states.size(1)

        x = []

        if len(rgb_obs.size()) == 5:
            observations["rgb"] = rgb_obs.contiguous().view(
                -1, rgb_obs.size(2), rgb_obs.size(3), rgb_obs.size(4)
            )
        # visual encoder
        rgb = observations["rgb"]
        depth = observations['depth']

#        with torch.no_grad():
#            self.detic_demo.predictor.model.eval()
#            sem =  self.get_semantic_mask_detic(rgb).unsqueeze(-1).to(depth.device)
        sem = observations['semantic']
        #print("SEMANTIC RUNTIME = " +str(sem_end-sem_start))
        rgb = self.visual_transform(rgb, N)
        depth = self.visual_transform(depth, N)
        sem = self.visual_transform(sem, N)
        rgb = self.visual_encoderRGB(rgb)
        depth = self.visual_encoderDEPTH(depth)
        sem = self.visual_encoderSEM(sem)
        embedRGB = self.visual_fcRGB(rgb)
        embedDEPTH = self.visual_fcDEPTH(depth)
        embedSEM = self.visual_fcSEM(sem)
        x.append(embedRGB)
        x.append(embedDEPTH)
        x.append(embedSEM)

        if EpisodicGPSSensor.cls_uuid in observations:
            obs_gps = observations[EpisodicGPSSensor.cls_uuid]
            if len(obs_gps.size()) == 3:
                obs_gps = obs_gps.contiguous().view(-1, obs_gps.size(2))
            x.append(self.gps_embedding(obs_gps))

        if EpisodicCompassSensor.cls_uuid in observations:
            obs_compass = observations["compass"]
            if len(obs_compass.size()) == 3:
                obs_compass = obs_compass.contiguous().view(
                    -1, obs_compass.size(2)
                )
            compass_observations = torch.stack(
                [
                    torch.cos(obs_compass),
                    torch.sin(obs_compass),
                ],
                -1,
            )
            compass_embedding = self.compass_embedding(
                compass_observations.float().squeeze(dim=1)
            )
            x.append(compass_embedding)

        if ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            if len(object_goal.size()) == 3:
                object_goal = object_goal.contiguous().view(
                    -1, object_goal.size(2)
                )
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        if self.policy_config.SEQ2SEQ.use_prev_action:
            prev_actions_embedding = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().view(-1)
            )
            x.append(prev_actions_embedding)

        x = torch.cat(x, dim=1)

        rnn_hidden_states = rnn_hidden_states.contiguous()
        x, rnn_hidden_states = self.state_encoder(
            x, rnn_hidden_states.contiguous(), masks
        )
        return x, rnn_hidden_states


@baseline_registry.register_policy
class ObjectNavILMAEPolicySemantic(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        policy_config: Config,
        run_type: str,
        hidden_size: int,
        rnn_type: str,
        num_recurrent_layers: int,
    ):
        super().__init__(
            ObjectNavILMAENetSemantic(
                observation_space=observation_space,
                policy_config=policy_config,
                num_actions=action_space.n,
                run_type=run_type,
                hidden_size=hidden_size,
                rnn_type=rnn_type,
                num_recurrent_layers=num_recurrent_layers,
            ),
            action_space.n,
            no_critic=policy_config.CRITIC.no_critic,
            mlp_critic=policy_config.CRITIC.mlp_critic,
            critic_hidden_dim=policy_config.CRITIC.hidden_dim,
        )

    @classmethod
    def from_config(cls, config: Config, observation_space, action_space):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            policy_config=config.POLICY,
            run_type=config.RUN_TYPE,
            hidden_size=config.POLICY.STATE_ENCODER.hidden_size,
            rnn_type=config.POLICY.STATE_ENCODER.rnn_type,
            num_recurrent_layers=config.POLICY.STATE_ENCODER.num_recurrent_layers,
        )

    @property
    def num_recurrent_layers(self):
        return self.net.num_recurrent_layers

    def freeze_visual_encoders(self):
        for param in self.net.visual_encoderRGB.parameters():
            param.requires_grad_(False)
        for param in self.net.visual_encoderDEPTH.parameters():
            param.requires_grad_(False)
        for param in self.net.visual_encoderSEM.parameters():
            param.requires_grad_(False)

    def unfreeze_visual_encoders(self):
        for param in self.net.visual_encoderRGB.parameters():
            param.requires_grad_(True)
        for param in self.net.visual_encoderDEPTH.parameters():
            param.requires_grad_(True)
        for param in self.net.visual_encoderSEM.parameters():
            param.requires_grad_(True)

    def freeze_state_encoder(self):
        for param in self.net.state_encoder.parameters():
            param.requires_grad_(False)

    def unfreeze_state_encoder(self):
        for param in self.net.state_encoder.parameters():
            param.requires_grad_(True)

    def freeze_actor(self):
        for param in self.action_distribution.parameters():
            param.requires_grad_(False)

    def unfreeze_actor(self):
        for param in self.action_distribution.parameters():
            param.requires_grad_(True)



class ObjectNavILNet(Net):
    r"""A baseline sequence to sequence network that concatenates instruction,
    RGB, and depth encodings before decoding an action distribution with an RNN.
    Modules:
        Instruction encoder
        Depth encoder
        RGB encoder
        RNN state encoder
    """

    def __init__(self, observation_space: Space, model_config: Config, num_actions, device=None):
        super().__init__()
        self.model_config = model_config
        rnn_input_size = 0

        # Init the depth encoder
        self.depth_encoder = VlnResnetDepthEncoder(
            observation_space,
            output_size=128,
#            checkpoint="data/ddppo-models/gibson-2plus-resnet50.pth",
            backbone="resnet50",
            trainable=False #26? Not 128
        )
        rnn_input_size += 128
        self.rgb_encoder = ResnetRGBEncoder(
            observation_space,
            output_size=256,
            backbone="resnet18",
            trainable=True,
            normalize_visual_inputs=False,
        )
        rnn_input_size += 256

        sem_seg_output_size = 0
        self.semantic_predictor = None
        self.is_thda = False
        sem_embedding_size = 256

        rgb_shape = observation_space.spaces["rgb"].shape
        spaces = {
            "semantic": Box(
                low=0,
                high=255,
                shape=(rgb_shape[0], rgb_shape[1], sem_embedding_size),
                dtype=np.uint8,
            ),
        }
        class MyClass:
            def __init__(self, spaces):
                self.spaces = spaces

        sem_obs_space = MyClass(spaces)
        self.sem_seg_encoder = ResnetSemSeqEncoder(
            sem_obs_space,
            output_size=256,
            backbone="resnet18",
            trainable=True,
            semantic_embedding_size=sem_embedding_size,
            is_thda=False
        )
        sem_seg_output_size = 256
        logger.info("Setting up Sem Seg model")
        rnn_input_size += sem_seg_output_size

#        self.embed_sge = model_config.embed_sge
#        if self.embed_sge:
#            self.task_cat2mpcat40 = torch.tensor(task_cat2mpcat40, device=device)
#            self.mapping_mpcat40_to_goal = np.zeros(
#                max(
#                    max(mapping_mpcat40_to_goal21.keys()) + 1,
#                    50,
#                ),
#                dtype=np.int8,
#            )
#
#            for key, value in mapping_mpcat40_to_goal21.items():
#                self.mapping_mpcat40_to_goal[key] = value
#            self.mapping_mpcat40_to_goal = torch.tensor(self.mapping_mpcat40_to_goal, device=device)
#            rnn_input_size += 1

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[
                EpisodicGPSSensor.cls_uuid
            ].shape[0]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32
            logger.info("\n\nSetting up GPS sensor")
        
        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[
                    0
                ]
                == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding_dim = 32
            self.compass_embedding = nn.Linear(input_compass_dim, self.compass_embedding_dim)
            rnn_input_size += 32
            logger.info("\n\nSetting up Compass sensor")

        if ObjectGoalSensor.cls_uuid in observation_space.spaces:
            self._n_object_categories = (
                int(
                    observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]
                )
                + 1
            )
            self.obj_categories_embedding = nn.Embedding(
                self._n_object_categories, 6
            )
            rnn_input_size += 6
            logger.info("\n\nSetting up Object Goal sensor")

        self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)
        rnn_input_size += self.prev_action_embedding.embedding_dim

        self.rnn_input_size = rnn_input_size

        self.state_encoder = RNNStateEncoder(
            input_size=rnn_input_size,
            hidden_size=2048,
            num_layers=2,
            rnn_type='GRU',
        )

        self.train()

    @property
    def output_size(self):
        return 2048

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind and self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def _extract_sge(self, observations):
        # recalculating to keep this self-contained instead of depending on training infra
        if "semantic" in observations and "objectgoal" in observations:
            obj_semantic = observations["semantic"].contiguous().flatten(start_dim=1)
            
            if len(observations["objectgoal"].size()) == 3:
                observations["objectgoal"] = observations["objectgoal"].contiguous().view(
                    -1, observations["objectgoal"].size(2)
                )

            idx = self.task_cat2mpcat40[
                observations["objectgoal"].long()
            ]
            if self.is_thda:
                idx = self.mapping_mpcat40_to_goal[idx].long()
            idx = idx.to(obj_semantic.device)

            if len(idx.size()) == 3:
                idx = idx.squeeze(1)

            goal_visible_pixels = (obj_semantic == idx).sum(dim=1)
            goal_visible_area = torch.true_divide(goal_visible_pixels, obj_semantic.size(-1)).float()
            return goal_visible_area.unsqueeze(-1)

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """
        rgb_obs = observations["rgb"]
        depth_obs = observations["depth"]

        x = []

        if self.depth_encoder is not None:
            if len(depth_obs.size()) == 5:
                observations["depth"] = depth_obs.contiguous().view(
                    -1, depth_obs.size(2), depth_obs.size(3), depth_obs.size(4)
                )

            depth_embedding = self.depth_encoder(observations)
            x.append(depth_embedding)

        if self.rgb_encoder is not None:
            if len(rgb_obs.size()) == 5:
                observations["rgb"] = rgb_obs.contiguous().view(
                    -1, rgb_obs.size(2), rgb_obs.size(3), rgb_obs.size(4)
                )

            rgb_embedding = self.rgb_encoder(observations)
            x.append(rgb_embedding)

        semantic_obs = observations["semantic"]
        if len(semantic_obs.size()) == 4:
            observations["semantic"] = semantic_obs.contiguous().view(
                -1, semantic_obs.size(2), semantic_obs.size(3)
            )
        if self.embed_sge:
            sge_embedding = self._extract_sge(observations)
            x.append(sge_embedding)

        sem_seg_embedding = self.sem_seg_encoder(observations)
        x.append(sem_seg_embedding)

        if EpisodicGPSSensor.cls_uuid in observations:
            obs_gps = observations[EpisodicGPSSensor.cls_uuid]
            if len(obs_gps.size()) == 3:
                obs_gps = obs_gps.contiguous().view(-1, obs_gps.size(2))
            x.append(self.gps_embedding(obs_gps))
        
        if EpisodicCompassSensor.cls_uuid in observations:
            obs_compass = observations["compass"]
            if len(obs_compass.size()) == 3:
                obs_compass = obs_compass.contiguous().view(-1, obs_compass.size(2))
            compass_observations = torch.stack(
                [
                    torch.cos(obs_compass),
                    torch.sin(obs_compass),
                ],
                -1,
            )
            compass_embedding = self.compass_embedding(compass_observations.squeeze(dim=1))
            x.append(compass_embedding)

        if ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            if len(object_goal.size()) == 3:
                object_goal = object_goal.contiguous().view(-1, object_goal.size(2))
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        prev_actions_embedding = self.prev_action_embedding(
            ((prev_actions.float() + 1) * masks).long().view(-1)
        )
        x.append(prev_actions_embedding)
        
        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states


@baseline_registry.register_policy
class ObjectNavILPolicy(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        policy_config: Config,
        run_type: str,
        hidden_size: int,
        rnn_type: str,
        num_recurrent_layers: int,
    ):
        super().__init__(
#            ObjectNavILMAENetSemantic(
#    def __init__(self, observation_space: Space, model_config: Config, num_actions, device=None):
            ObjectNavILNet(
                observation_space=observation_space,
                model_config=None,
                num_actions=None,
            ),
            action_space.n,
            no_critic=policy_config.CRITIC.no_critic,
            mlp_critic=policy_config.CRITIC.mlp_critic,
            critic_hidden_dim=policy_config.CRITIC.hidden_dim,
        )

    @classmethod
    def from_config(cls, config: Config, observation_space, action_space):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            policy_config=config.POLICY,
            run_type=config.RUN_TYPE,
            hidden_size=config.POLICY.STATE_ENCODER.hidden_size,
            rnn_type=config.POLICY.STATE_ENCODER.rnn_type,
            num_recurrent_layers=config.POLICY.STATE_ENCODER.num_recurrent_layers,
        )

    @property
    def num_recurrent_layers(self):
        return self.net.num_recurrent_layers

    def freeze_visual_encoders(self):
        for param in self.net.visual_encoder.parameters():
            param.requires_grad_(False)

    def unfreeze_visual_encoders(self):
        for param in self.net.visual_encoder.parameters():
            param.requires_grad_(True)

    def freeze_state_encoder(self):
        for param in self.net.state_encoder.parameters():
            param.requires_grad_(False)

    def unfreeze_state_encoder(self):
        for param in self.net.state_encoder.parameters():
            param.requires_grad_(True)

    def freeze_actor(self):
        for param in self.action_distribution.parameters():
            param.requires_grad_(False)

    def unfreeze_actor(self):
        for param in self.action_distribution.parameters():
            param.requires_grad_(True)
