"""
This module provides functions to get a task operator and model adapter.
"""

import os
import sys
from logging import Logger

from omegaconf import DictConfig
from omegaconf.omegaconf import open_dict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

sys.path.append(ROOT_DIR)

from pipeline.modeladapter import ModelAdapter
from pipeline.taskoperator import TaskOperator


def get_task_operator(
    model_config: DictConfig,
    operator_config: DictConfig,
    out_dir: str,
    dataset: dict,
    logger: Logger,
) -> TaskOperator:
    """Build model adapter object and task operator obejct.

    Args:
        model_config (DictConfig): configuration to build model adapter.
        operator_config (DictConfig): configuration to build task operator.
        out_dir (str): parent directory for result directory.
        dataset (dict): dataset for train and test model.
        logger (Logger): logger.
        device (int): device to run model.
        world_size (int, optional): number of process. Defaults to None.

    Returns:
        TaskOperator: task operator which contain model.
    """
    with open_dict(model_config):
        if isinstance(dataset["test"]["labels"][0], float):
            model_config.num_labels = 1
        else:
            model_config.num_labels = len(dataset["test"]["labels"][0])

    logger.info(
        f"add number of labels: {model_config.num_labels}, "
        + f"following {dataset['test']['labels'][0]}",
    )

    model_adapter = ModelAdapter(model_config, model_config.device)

    task_operator = TaskOperator(
        out_dir=out_dir,
        model_adapter=model_adapter,
        config=operator_config,
        dataset=dataset,
        logger=logger,
        device=model_config.device,
    )

    return task_operator
