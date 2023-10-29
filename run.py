#!/usr/bin/env python3
import argparse
import os
import random
import subprocess
from datetime import datetime

import habitat
import numpy as np
import pudb
import quaternion
import torch
from habitat import logger
from habitat.config import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from pirlnav.config import get_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def main_submit(run_type, exp_config):
    script = "hostname 2>&1 | tail -n1"
    proc = subprocess.run(script, shell=True, capture_output=True)
    main_addr = proc.stdout.decode().strip("\n")
    parser = argparse.ArgumentParser()
    assert run_type in ["train", "eval"]
    run_exp(exp_config, run_type)


def execute_exp(config: Config, run_type: str) -> None:
    r"""This function runs the specified config with the specified runtype
    Args:
    config: Habitat.config
    runtype: str {train or eval}
    """
    # set a random seed (from detectron2)
    seed = (os.getpid() + int(datetime.now().strftime("%S%f")) +
            int.from_bytes(os.urandom(2), "big"))
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    logger.info("Using a generated random seed {}".format(seed))
    config.defrost()
    config.RUN_TYPE = run_type
    config.TASK_CONFIG.SEED = seed
    config.freeze()
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    if config.FORCE_TORCH_SINGLE_THREADED and torch.cuda.is_available():
        torch.set_num_threads(1)

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()


def run_exp(exp_config: str, run_type: str, opts=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    print("*" * 15)
    print(exp_config, opts)
    print("*" * 15)
    config = get_config(exp_config, opts)
    execute_exp(config, run_type)


if __name__ == "__main__":
    main()
