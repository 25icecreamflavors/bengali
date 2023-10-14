import logging
import os
import random

import numpy as np
import torch
import wandb
import yaml


def set_seed(seed=1771):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def read_config(file_path):
    """Open and read yaml config.

    Args:
        file_path (str): path to config file

    Returns:
        dict: config file
    """
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def setup_logging(say_my_name="debug"):
    """Set up logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        handlers=[
            logging.StreamHandler(),  # Output log messages to console
            logging.FileHandler(
                f"logs/{say_my_name}.log"
            ),  # Save log messages to a file
        ],
    )


def set_wandb(config):
    # Set up wandb logging
    if config["debug"] == 1:
        project_name = "debug_bengali"
    else:
        project_name = "bengali"

    wandb.init(
        # Set the project where this run will be logged
        project=project_name,
        # Set up the run name
        name=(
            f"{config['name']}_lr{config['learning_rate']}_"
            f"epochs{config['epochs']}"
        ),
        # Track hyperparameters and run metadata
        config={
            "learning_rate": config["learning_rate"],
            "epochs": config["epochs"],
        },
    )
