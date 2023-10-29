import contextlib
import os
import pickle
import random
import string
import time
from collections import OrderedDict, defaultdict, deque
from typing import Any, Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm
from arguments import get_args
from gym import spaces
from habitat import Config, logger
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils import profiling_wrapper
from habitat.utils.render_wrapper import overlay_frame
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensorboard_utils import TensorboardWriter, get_writer
from habitat_baselines.il.env_based.policy.rednet import load_rednet
from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    add_signal_handlers,
    init_distrib_slurm,
    is_slurm_batch_job,
    load_resume_state,
    rank0_only,
    requeue_job,
    save_resume_state,
)
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_baselines.utils.common import (
    action_array_to_dict,
    batch_obs,
    generate_video,
    get_num_actions,
    is_continuous_action_space,
)
from PIL import Image
from torchvision import transforms

from pirlnav.algos.ppo import DDPPO, PPO
from pirlnav.utils.lr_scheduler import PIRLNavLRScheduler


def replace_value(value):
    value_dict = {0: 0, 3: 1, 2: 2, 4: 3, 5: 4, 1: 5}
    return value_dict.get(
        value, value
    )  # Use get() to return the value if it's in the dictionary, else return the original value


def replace_value_rednet(value):
    value_dict = {2: 0, 10: 1, 13: 2, 17: 3, 21: 4, 9: 5}


def visualize_semantic(ob):
    rgb_image = ob["rgb"]
    semantic_mask = ob["semantic"].squeeze()
    colormaps = [
        (0, (255, 0, 0)),  # Red
        (1, (0, 255, 0)),  # Green
        (2, (0, 0, 255)),  # Blue
        (3, (255, 255, 0)),  # Yellow
        (4, (255, 0, 255)),  # Magenta
        (5, (0, 255, 255)),  # Cyan
        (6, (255, 128, 0)),  # Orange
        (7, (128, 255, 0)),  # Lime Green
        (8, (0, 255, 128)),  # Teal
        (9, (0, 128, 255)),  # Sky Blue
        (10, (255, 0, 128)),  # Magenta
        (11, (128, 0, 255)),  # Deep Purple
        (12, (255, 128, 128)),  # Light Pink
        (13, (128, 255, 128)),  # Light Green
        (14, (128, 128, 255)),  # Light Blue
        (15, (255, 255, 128)),  # Light Yellow
        (16, (255, 128, 255)),  # Light Magenta
        (17, (128, 255, 255)),  # Light Cyan
        (18, (192, 192, 192)),  # Light Grey
        (19, (128, 128, 128)),  # Grey
        (20, (255, 192, 128)),  # Light Orange
        (21, (255, 128, 192)),  # Light Pinkish Red
        (22, (192, 255, 128)),  # Pale Green
        (23, (128, 255, 192)),  # Pale Cyan
        (24, (128, 192, 255)),  # Pale Blue
        (25, (192, 128, 255)),  # Pale Purple
        (26, (192, 255, 255)),  # Pale Turquoise
        (27, (255, 192, 255)),  # Pale Magenta
        (28, (192, 192, 255)),  # Pale Lavender
        (29, (255, 255, 192)),  # Pale Yellow
    ]

    assert (rgb_image.shape[:2] == semantic_mask.shape
            ), "Dimensions of RGB image and mask must match."
    overlay_image = np.copy(rgb_image)
    alpha = 0.5
    for cmap in colormaps:
        #        if cmap[0] not in ob["objectgoal"]:
        #            continue
        mask_region = semantic_mask == cmap[0]
        overlay_color = np.array(cmap[1], dtype=np.uint8)

        overlay_image[mask_region] = (
            alpha * overlay_color +
            (1 - alpha) * overlay_image[mask_region]).astype(np.uint8)
    return overlay_image


@baseline_registry.register_trainer(name="pirlnav-ddppo")
@baseline_registry.register_trainer(name="pirlnav-ppo")
class PIRLNavPPOTrainer(PPOTrainer):
    def __init__(self, config=None):
        self.rednet = None
        # f = open("args_mask.pkl", "rb")
        # args = pickle.load(f)
        # f.close()
        # args.sem_gpu_id = 0
        # self.maskrcnn = SemanticPredMaskRCNN(args)
        super().__init__(config)

    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.
        Args:
            ppo_cfg: config node with relevant params
        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        policy = baseline_registry.get_policy(self.config.RL.POLICY.name)
        observation_space = self.obs_space
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms)

        logger.info("Setting up policy in PIRLNav trainer..........")

        self.actor_critic = policy.from_config(self.config, observation_space,
                                               self.policy_action_space)
        self.obs_space = observation_space
        self.actor_critic.to(self.device)

        if self.config.RL.DDPPO.pretrained_encoder or self.config.RL.DDPPO.pretrained:
            pretrained_state = torch.load(
                self.config.RL.DDPPO.pretrained_weights, map_location="cpu")
        if self.config.RL.DDPPO.pretrained:
            if ("model.net.visual_encoder.running_mean_and_var._mean"
                    in pretrained_state["state_dict"]):
                new_dict = OrderedDict()
                for key, value in pretrained_state["state_dict"].items():
                    new_key = key.replace("model", "actor_critic")
                    new_dict[new_key] = value

                self.actor_critic.load_state_dict(
                    {k[len("actor_critic."):]: v
                     for k, v in new_dict.items()},
                    strict=False,
                )
            else:
                self.actor_critic.load_state_dict(
                    {  # type: ignore
                        k[len("actor_critic."):]: v
                        for k, v in pretrained_state["state_dict"].items()
                    },
                    strict=False,
                )
        elif self.config.RL.DDPPO.pretrained_encoder:
            if ("model.net.visual_encoder.running_mean_and_var._mean"
                    in pretrained_state["state_dict"]):
                new_dict = OrderedDict()
                for key, value in pretrained_state["state_dict"].items():
                    new_key = key.replace("model", "actor_critic")
                    new_dict[new_key] = value

                prefix = "actor_critic.net.visual_encoder."
                self.actor_critic.net.visual_encoder.load_state_dict({
                    k[len(prefix):]: v
                    for k, v in new_dict.items() if k.startswith(prefix)
                })
            else:
                prefix = "actor_critic.net.visual_encoder."
                self.actor_critic.net.visual_encoder.load_state_dict({
                    k[len(prefix):]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(prefix)
                })

        if not self.config.RL.DDPPO.train_encoder:
            self._static_encoder = True
            for param in self.actor_critic.net.visual_encoder.parameters():
                param.requires_grad_(False)

        if self.config.RL.DDPPO.reset_critic:
            nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
            nn.init.constant_(self.actor_critic.critic.fc.bias, 0)

        self.agent = (DDPPO if self._is_distributed else PPO)(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training DD/PPO.
        Returns:
            None
        """
        if self.config.PRETRAIN_TRAINING_CONFIG:
            checkpoint_path = self.config.PRETRAIN_TRAINING
            ckpt_dict = self.load_checkpoint(checkpoint_path,
                                             map_location="cpu")
            self.config = self._setup_eval_config(ckpt_dict["config"])
        self._init_train()

        count_checkpoints = 0
        prev_time = 0

        lr_scheduler = PIRLNavLRScheduler(
            optimizer=self.agent.optimizer,
            agent=self.agent,
            num_updates=self.config.NUM_UPDATES,
            base_lr=self.config.RL.PPO.lr,
            finetuning_lr=self.config.RL.Finetune.lr,
            ppo_eps=self.config.RL.PPO.eps,
            start_actor_update_at=self.config.RL.Finetune.
            start_actor_update_at,
            start_actor_warmup_at=self.config.RL.Finetune.
            start_actor_warmup_at,
            start_critic_update_at=self.config.RL.Finetune.
            start_critic_update_at,
            start_critic_warmup_at=self.config.RL.Finetune.
            start_critic_warmup_at,
        )

        resume_state = load_resume_state(self.config)
        if resume_state is not None:
            self.agent.load_state_dict(resume_state["state_dict"])
            self.agent.optimizer.load_state_dict(resume_state["optim_state"])
            lr_scheduler.load_state_dict(resume_state["lr_sched_state"])

            requeue_stats = resume_state["requeue_stats"]
            self.env_time = requeue_stats["env_time"]
            self.pth_time = requeue_stats["pth_time"]
            self.num_steps_done = requeue_stats["num_steps_done"]
            self.num_updates_done = requeue_stats["num_updates_done"]
            self._last_checkpoint_percent = requeue_stats[
                "_last_checkpoint_percent"]
            count_checkpoints = requeue_stats["count_checkpoints"]
            prev_time = requeue_stats["prev_time"]

            self.running_episode_stats = requeue_stats["running_episode_stats"]
            self.window_episode_stats.update(
                requeue_stats["window_episode_stats"])
        if self.config.PRETRAIN_TRAINING:
            checkpoint_path = self.config.PRETRAIN_TRAINING
            ckpt_dict = self.load_checkpoint(checkpoint_path,
                                             map_location="cpu")
            self.agent.load_state_dict(ckpt_dict["state_dict"])
        ppo_cfg = self.config.RL.PPO
        self.actor_critic.config = self.config

        with (get_writer(self.config, flush_secs=self.flush_secs)
              if rank0_only() else contextlib.suppress()) as writer:
            while not self.is_done():
                profiling_wrapper.on_start_step()
                profiling_wrapper.range_push("train update")

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * (
                        1 - self.percent_done())

                if rank0_only() and self._should_save_resume_state():
                    requeue_stats = dict(
                        env_time=self.env_time,
                        pth_time=self.pth_time,
                        count_checkpoints=count_checkpoints,
                        num_steps_done=self.num_steps_done,
                        num_updates_done=self.num_updates_done,
                        _last_checkpoint_percent=self._last_checkpoint_percent,
                        prev_time=(time.time() - self.t_start) + prev_time,
                        running_episode_stats=self.running_episode_stats,
                        window_episode_stats=dict(self.window_episode_stats),
                    )

                    save_resume_state(
                        dict(
                            state_dict=self.agent.state_dict(),
                            optim_state=self.agent.optimizer.state_dict(),
                            lr_sched_state=lr_scheduler.state_dict(),
                            config=self.config,
                            requeue_stats=requeue_stats,
                        ),
                        self.config,
                    )

                if EXIT.is_set():
                    profiling_wrapper.range_pop()  # train update

                    self.envs.close()

                    requeue_job()

                    return

                self.agent.eval()
                count_steps_delta = 0
                profiling_wrapper.range_push("rollouts loop")

                profiling_wrapper.range_push("_collect_rollout_step")
                for buffer_index in range(self._nbuffers):
                    self._compute_actions_and_step_envs(buffer_index)

                for step in range(ppo_cfg.num_steps):
                    is_last_step = (self.should_end_early(step + 1)
                                    or (step + 1) == ppo_cfg.num_steps)

                    for buffer_index in range(self._nbuffers):
                        count_steps_delta += self._collect_environment_result(
                            buffer_index)

                        if (buffer_index + 1) == self._nbuffers:
                            profiling_wrapper.range_pop(
                            )  # _collect_rollout_step

                        if not is_last_step:
                            if (buffer_index + 1) == self._nbuffers:
                                profiling_wrapper.range_push(
                                    "_collect_rollout_step")
                            self._compute_actions_and_step_envs(buffer_index)

                    if is_last_step:
                        break

                profiling_wrapper.range_pop()  # rollouts loop

                if self._is_distributed:
                    self.num_rollouts_done_store.add("num_done", 1)

                (
                    value_loss,
                    action_loss,
                    dist_entropy,
                ) = self._update_agent()

                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()  # type: ignore

                self.num_updates_done += 1
                losses = self._coalesce_post_step(
                    dict(
                        value_loss=value_loss,
                        action_loss=action_loss,
                        entropy=dist_entropy,
                    ),
                    count_steps_delta,
                )

                self._training_log(writer, losses, prev_time)

                # checkpoint model
                if rank0_only() and self.should_checkpoint():
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth",
                        dict(
                            step=self.num_steps_done,
                            wall_time=(time.time() - self.t_start) + prev_time,
                        ),
                    )
                    count_checkpoints += 1

                profiling_wrapper.range_pop()  # train update

            self.envs.close()

    def _init_train(self):
        resume_state = load_resume_state(self.config)
        if resume_state is not None:
            self.config: Config = resume_state["config"]
            self.using_velocity_ctrl = (
                self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS) == [
                    "VELOCITY_CONTROL"
            ]

        if self.config.RL.DDPPO.force_distributed:
            self._is_distributed = True

        if is_slurm_batch_job():
            add_signal_handlers()

        if self._is_distributed:
            local_rank, tcp_store = init_distrib_slurm(
                self.config.RL.DDPPO.distrib_backend)
            if rank0_only():
                logger.info("Initialized DD-PPO with {} workers".format(
                    torch.distributed.get_world_size()))

            self.config.defrost()
            self.config.TORCH_GPU_ID = local_rank
            self.config.SIMULATOR_GPU_ID = local_rank
            # Multiply by the number of simulators to make sure they also get unique seeds
            self.config.TASK_CONFIG.SEED += (torch.distributed.get_rank() *
                                             self.config.NUM_ENVIRONMENTS)
            self.config.freeze()

            random.seed(self.config.TASK_CONFIG.SEED)
            np.random.seed(self.config.TASK_CONFIG.SEED)
            torch.manual_seed(self.config.TASK_CONFIG.SEED)
            self.num_rollouts_done_store = torch.distributed.PrefixStore(
                "rollout_tracker", tcp_store)
            self.num_rollouts_done_store.set("num_done", "0")

        if rank0_only() and self.config.VERBOSE:
            logger.info(f"config: {self.config}")

        profiling_wrapper.configure(
            capture_start_step=self.config.PROFILING.CAPTURE_START_STEP,
            num_steps_to_capture=self.config.PROFILING.NUM_STEPS_TO_CAPTURE,
        )

        self._init_envs()

        action_space = self.envs.action_spaces[0]
        if self.using_velocity_ctrl:
            # For navigation using a continuous action space for a task that
            # may be asking for discrete actions
            self.policy_action_space = action_space["VELOCITY_CONTROL"]
            action_shape = (2, )
            discrete_actions = False
        else:
            self.policy_action_space = action_space
            if is_continuous_action_space(action_space):
                # Assume ALL actions are NOT discrete
                action_shape = (get_num_actions(action_space), )
                discrete_actions = False
            else:
                # For discrete pointnav
                action_shape = None
                discrete_actions = True

        ppo_cfg = self.config.RL.PPO
        policy_cfg = self.config.POLICY
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.config.TORCH_GPU_ID)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if rank0_only() and not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_actor_critic_agent(ppo_cfg)
        if self._is_distributed:
            self.agent.init_distributed(
                find_unused_params=True)  # type: ignore

        logger.info("agent number of parameters: {}".format(
            sum(param.numel() for param in self.agent.parameters())))

        obs_space = self.obs_space
        if self._static_encoder:
            self._encoder = self.actor_critic.net.visual_encoder
            obs_space = spaces.Dict({
                "visual_features":
                spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=self._encoder.output_shape,
                    dtype=np.float32,
                ),
                **obs_space.spaces,
            })

        self._nbuffers = 2 if ppo_cfg.use_double_buffered_sampler else 1

        self.rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            obs_space,
            self.policy_action_space,
            policy_cfg.STATE_ENCODER.hidden_size,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
            is_double_buffered=ppo_cfg.use_double_buffered_sampler,
            action_shape=action_shape,
            discrete_actions=discrete_actions,
        )
        self.rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(observations,
                          device=self.device,
                          cache=self._obs_batching_cache)
        batch = apply_obs_transforms_batch(batch,
                                           self.obs_transforms)  # type: ignore

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        self.rollouts.buffers["observations"][0] = batch  # type: ignore

        self.current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        self.running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size))

        self.env_time = 0.0
        self.pth_time = 0.0
        self.t_start = time.time()

    @rank0_only
    def _training_log(self,
                      writer,
                      losses: Dict[str, float],
                      prev_time: int = 0):
        deltas = {
            k:
            ((v[-1] - v[0]).sum().item() if len(v) > 1 else v[0].sum().item())
            for k, v in self.window_episode_stats.items()
        }
        deltas["count"] = max(deltas["count"], 1.0)

        writer.add_scalar(
            "reward",
            deltas["reward"] / deltas["count"],
            self.num_steps_done,
        )

        # Check to see if there are any metrics
        # that haven't been logged yet
        metrics = {
            k: v / deltas["count"]
            for k, v in deltas.items() if k not in {"reward", "count"}
        }

        for k, v in metrics.items():
            writer.add_scalar(f"metrics/{k}", v, self.num_steps_done)
        for k, v in losses.items():
            writer.add_scalar(f"losses/{k}", v, self.num_steps_done)

        fps = self.num_steps_done / ((time.time() - self.t_start) + prev_time)
        writer.add_scalar("metrics/fps", fps, self.num_steps_done)

        lrs = {}
        for i, param_group in enumerate(self.agent.optimizer.param_groups):
            lrs["pg_{}".format(i)] = param_group["lr"]
        writer.add_scalars("learning_rate", lrs, self.num_steps_done)

        # log stats
        if self.num_updates_done % self.config.LOG_INTERVAL == 0:
            logger.info("update: {}\tfps: {:.3f}\t".format(
                self.num_updates_done,
                fps,
            ))

            logger.info("update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                        "frames: {}".format(
                            self.num_updates_done,
                            self.env_time,
                            self.pth_time,
                            self.num_steps_done,
                        ))

            logger.info("Average window size: {}  {}".format(
                len(self.window_episode_stats["count"]),
                "  ".join("{}: {:.3f}".format(k, v / deltas["count"])
                          for k, v in deltas.items() if k != "count"),
            ))

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        print(checkpoint_path)
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # Map location CPU is almost always better than mapping to a CUDA device.
        if self.config.EVAL.SHOULD_LOAD_CKPT:
            ckpt_dict = self.load_checkpoint(checkpoint_path,
                                             map_location="cpu")
        else:
            ckpt_dict = {}
        config = self.config.clone()

        ppo_cfg = config.RL.PPO
        policy_cfg = config.POLICY

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if len(self.config.VIDEO_OPTION
               ) > 0 and self.config.VIDEO_RENDER_TOP_DOWN:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        if config.VERBOSE:
            logger.info(f"env config: {config}")

        self._init_envs(config)

        action_space = self.envs.action_spaces[0]
        if self.using_velocity_ctrl:
            # For navigation using a continuous action space for a task that
            # may be asking for discrete actions
            self.policy_action_space = action_space["VELOCITY_CONTROL"]
            action_shape = (2, )
            discrete_actions = False
        else:
            self.policy_action_space = action_space
            if is_continuous_action_space(action_space):
                # Assume NONE of the actions are discrete
                action_shape = (get_num_actions(action_space), )
                discrete_actions = False
            else:
                # For discrete pointnav
                action_shape = (1, )
                discrete_actions = True

        self._setup_actor_critic_agent(ppo_cfg)

        if self.agent.actor_critic.should_load_agent_state:
            if ("model.net.visual_encoder.running_mean_and_var._mean"
                    in ckpt_dict["state_dict"]):
                new_dict = OrderedDict()
                for key, value in ckpt_dict["state_dict"].items():
                    new_key = key.replace("model", "actor_critic")
                    new_dict[new_key] = value
                self.agent.load_state_dict(new_dict, strict=False)
            else:
                self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic
        self.actor_critic.config = self.config

        observations = self.envs.reset()
        batch = batch_obs(observations,
                          device=self.device,
                          cache=self._obs_batching_cache)
        batch = apply_obs_transforms_batch(batch,
                                           self.obs_transforms)  # type: ignore

        current_episode_reward = torch.zeros(self.envs.num_envs,
                                             1,
                                             device="cpu")

        test_recurrent_hidden_states = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            self.actor_critic.num_recurrent_layers,
            policy_cfg.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            *action_shape,
            device=self.device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[Any, Any] = {
        }  # dict of dicts that stores stats per episode

        rgb_frames = [[] for _ in range(self.config.NUM_ENVIRONMENTS)
                      ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}.")
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        pbar = tqdm.tqdm(total=number_of_eval_episodes)
        self.actor_critic.eval()
        self.first_images = [None] * self.config.NUM_ENVIRONMENTS
        while len(stats_episodes
                  ) < number_of_eval_episodes and self.envs.num_envs > 0:
            current_episodes = self.envs.current_episodes()
            n_envs = self.envs.num_envs
            if (self.config.LMN.PREDICT_SEMANTICS_MASKRCNN
                    and not "semantic" in observations[0]):
                semantic_pred, rgb_vis = self.maskrcnn.get_prediction(
                    observations[0]["rgb"])
                new_layer = np.full((480, 640, 1),
                                    self.config.LMN.THRESHOLD_CUTOFF)
                semantic_pred = np.concatenate((semantic_pred, new_layer),
                                               axis=2)
                temp = np.argmax(semantic_pred, axis=2)
                replace_func = np.vectorize(replace_value)
                temp = replace_func(temp)
                observations[0]["semantic"] = temp
                # observations[0]["semantic"] = visualize_semantic(
                #    observations[0])
                batch["semantic"] = torch.Tensor(temp).unsqueeze(0).unsqueeze(
                    -1)
            if not self.config.COLLECT_DATASET:
                with torch.no_grad():
                    (
                        _,
                        actions,
                        _,
                        test_recurrent_hidden_states,
                    ) = self.actor_critic.act(
                        batch,
                        test_recurrent_hidden_states,
                        prev_actions,
                        not_done_masks,
                        deterministic=False,
                        LMN=self.config.LAST_MILE_NAVIGATION,
                        LMN_LOSS_IGNORE=self.config.LMN_LOSS_IGNORE,
                    )
                    prev_actions.copy_(actions)  # type: ignore
            else:
                actions = torch.tensor([[0 for _ in range(n_envs)]]).T
            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            # For backwards compatibility, we also call .item() to convert to
            # an int
            if actions[0].shape[0] > 1:
                step_data = [
                    action_array_to_dict(self.policy_action_space, a)
                    for a in actions.to(device="cpu")
                ]
            else:
                step_data = [a.item() for a in actions.to(device="cpu")]

            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            if self.config.COLLECT_DATASET:
                for i in range(n_envs):
                    rgb = observations[i]["rgb"]
                    # rgb = self.actor_critic.lmn_dataset[i][0].cpu().numpy()
                    depth = observations[i]["depth"]
                    # depth = self.actor_critic.lmn_dataset[i][1].cpu().numpy()
                    sem = observations[i]["semantic"]
                    # sem = self.actor_critic.lmn_dataset[i][2].cpu().numpy()
                    # mask = np.where(sem != self.actor_critic.lmn_dataset[i]
                    #                [3].cpu().numpy()[0])
                    color_dict = {
                        0: (255, 0, 0),  # Red
                        1: (0, 255, 0),  # Green
                        2: (0, 0, 255),  # Blue
                        3: (255, 255, 0),  # Yellow
                        4: (255, 0, 255),  # Magenta
                        5: (0, 255, 255),  # Cyan
                        255: (255, 255, 255),  # White
                    }
                    self.first_images[i] = (rgb, depth, sem)
            for info in infos:
                if "top_down_map" in info:
                    fog_of_war_cov = np.sum(
                        info["top_down_map"]["fog_of_war_mask"] == 1)
                    map_size = np.sum(info["top_down_map"]["map"] == 1)
                    info["exploration_coverage_m2"] = (
                        fog_of_war_cov *
                        info["top_down_map"]["meters_per_pixel"]**2
                    )  # 0.01 meters per pixel
                    info["exploration_coverage"] = fog_of_war_cov / map_size
            if (self.config.LMN.PREDICT_SEMANTICS_REDNET
                    and not "semantic" in observations[0]):
                pass
            if (self.config.LMN.PREDICT_SEMANTICS_MASKRCNN
                    and not "semantic" in observations[0]):
                semantic_pred, rgb_vis = self.maskrcnn.get_prediction(
                    observations[0]["rgb"])
                new_layer = np.full((480, 640, 1), 0.9)
                semantic_pred = np.concatenate((semantic_pred, new_layer),
                                               axis=2)
                temp = np.argmax(semantic_pred, axis=2)
                replace_func = np.vectorize(replace_value)
                temp = replace_func(temp)
                observations[0]["semantic"] = temp
                # observations[0]["semantic"] = visualize_semantic(
                #    observations[0])
                batch["semantic"] = torch.Tensor(temp).unsqueeze(0).unsqueeze(
                    -1)
            batch = batch_obs(  # type: ignore
                observations,
                device=self.device,
                cache=self._obs_batching_cache,
            )
            batch = apply_obs_transforms_batch(
                batch, self.obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            rewards = torch.tensor(rewards_l, dtype=torch.float,
                                   device="cpu").unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            for i in range(n_envs):
                if (
                        next_episodes[i].scene_id,
                        next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not not_done_masks[i].item():
                    pbar.update()
                    episode_stats = {
                        "reward": current_episode_reward[i].item()
                    }
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(
                        current_episodes[i].scene_id,
                        current_episodes[i].episode_id,
                    )] = episode_stats

                    num_episodes = len(stats_episodes)
                    aggregated_stats = {}
                    for stat_key in next(iter(stats_episodes.values())).keys():
                        aggregated_stats[stat_key] = (
                            sum(v[stat_key] for v in stats_episodes.values()) /
                            num_episodes)
                    print("**********************")
                    print(num_episodes)
                    print(aggregated_stats)
                    print("**********************")
                    if self.config.COLLECT_DATASET:
                        # MAKE DATAPOINT
                        # note previously the ob goal was index at 0 not i and was causing issues.
                        name = (current_episodes[i].episode_id + "_" +
                                current_episodes[i].object_category + "_")
                        #    + str(observations[i]["objectgoal"][i]) + "_")
                        metrics = infos[i]
                        metric_strs = []
                        keys_to_include_in_name = (
                            self.config.EVAL_KEYS_TO_INCLUDE_IN_NAME)
                        if (keys_to_include_in_name is not None
                                and len(keys_to_include_in_name) > 0):
                            use_metrics_k = [
                                k for k in metrics
                                if any(to_include_k in k for to_include_k in
                                       keys_to_include_in_name)
                            ]
                        else:
                            use_metrics_k = list(metrics.keys())
                        #                        if metrics["num_steps"] == 1:
                        #                            continue
                        for k in use_metrics_k:
                            metric_strs.append(f"{k}={metrics[k]:.2f}")
                        random_string = "".join(
                            random.choice(string.ascii_letters)
                            for _ in range(10))
                        image_name = (
                            f"episode={name}-ckpt={checkpoint_index}-string-{random_string}"
                            + "-".join(metric_strs))
                        try:
                            rgb, depth, sem = self.first_images[i]
                        except Exception as e:
                            self.first_images[i] = None
                            continue
                        depth = depth * 10000
                        depth = depth.astype(np.uint16)
                        if isinstance(rgb, type(None)):
                            continue
                        if random.random() < 0.1:
                            cv2.imwrite(
                                "./switch_dataset2_eval/" + image_name +
                                ".png", depth)
                            cv2.imwrite(
                                "./switch_dataset2_eval/" + image_name +
                                ".jpg",
                                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                            )
                            cv2.imwrite(
                                "./switch_dataset2_eval/" + image_name +
                                "_sem.png", sem)
                        self.first_images[i] = None

                    if len(self.config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id +
                            current_episodes[i].object_category,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            fps=self.config.VIDEO_FPS,
                            tb_writer=writer,
                            keys_to_include_in_name=self.config.
                            EVAL_KEYS_TO_INCLUDE_IN_NAME,
                        )

                        rgb_frames[i] = []

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image(
                        {k: v[i]
                         for k, v in batch.items()}, infos[i])
                    if self.config.VIDEO_RENDER_ALL_INFO:
                        frame = overlay_frame(frame, infos[i])
                    # if hasattr(self.actor_critic, "lmn"):
                    #    font = cv2.FONT_HERSHEY_SIMPLEX
                    #    font_scale = 1
                    #    font_color = (255, 255, 255)  # White color
                    #    line_thickness = 2
                    #    x = 100
                    #    y = 100
                    #    if self.actor_critic.lmn[i]:
                    #        frame[0:200, 0:200, 0] = 255
                    #        frame[0:200, 0:200, 1] = 0
                    #        frame[0:200, 0:200, 2] = 0
                    #        frame = cv2.putText(
                    #            frame,
                    #            str(self.actor_critic.lmn_debug[i]),
                    #            (x, y),
                    #            font,
                    #            font_scale,
                    #            font_color,
                    #            line_thickness,
                    #        )
                    rgb_frames[i].append(frame)

            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        num_episodes = len(stats_episodes)
        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key]
                    for v in stats_episodes.values()) / num_episodes)

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalar("eval_reward/average_reward",
                          aggregated_stats["reward"], step_id)

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        self.envs.close()
