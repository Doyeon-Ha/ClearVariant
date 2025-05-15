"""
Main script to process raw data, construct datasets, and execute downstream tasks
(such as model training or evaluation) using configurable pipelines.

This script performs the following high-level steps:
1. Sets up the output directory and logging.
2. Loads or processes raw data to build a dataset.
3. Initializes a task operator (e.g., trainer or evaluator) based on configuration.
4. Executes the specified task.

Configuration is managed using Hydra and OmegaConf.

Modules used:
- `RawDataProcessor`: Processes raw input data into structured format.
- `DatasetBuilder`: Constructs or loads datasets.
- `get_task_operator`: Retrieves the operator to perform the final task
    (e.g., training).
- `utils.get_logger`: Initializes and configures logging.

To run:
    python main.py

Make sure the required configuration files are present under the `config/` directory.
"""

import os
import sys
from datetime import datetime
from logging import Logger

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from pipeline.datasetbuilder import DatasetBuilder
from pipeline.gettaskoperator import get_task_operator
from pipeline.rawdataprocessor import RawDataProcessor
from utils.utils import get_logger


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """
    Process raw data, build datasets, and perform tasks using the specified
    configurations.

    Args:
        cfg (DictConfig): Configuration object containing various settings.
    """
    out_dir, logger = _set_up(cfg)
    dataset = _load_dataset(cfg, out_dir, logger)

    task_operator = get_task_operator(
        cfg.model_config,
        cfg.task_operator_config,
        out_dir,
        dataset,
        logger,
    )
    task_operator.do_task()

    return


def _set_up(config: DictConfig) -> tuple[str, Logger]:
    """
    Set up the output directory and logger based on the provided configuration.

    Args:
        config (DictConfig): Configuration object containing various settings.

    Returns:
        tuple[str, Logger]: A tuple containing the output directory path and the logger
        instance.
    """
    out_dir = os.path.join(
        ROOT_DIR,
        config.constants.output_root,
        config.task_operator_config.task,
        config.raw_data_processor_config.db_name,
        ".".join(
            [
                config.dataset_builder_config.db_processing,
                config.raw_data_processor_config.db_option.split("/")[-1].replace(
                    ".csv", ""
                ),
            ]
        ),
        config.model_config.pretrained_model.split("/")[1],
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "model_param"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "attn_out"), exist_ok=True)

    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(config, resolve=True))

    logger = get_logger(out_dir, "log")
    logger.info(f"Model Output directory is {out_dir}")

    return out_dir, logger


def _load_dataset(cfg: DictConfig, out_dir: str, logger: Logger) -> dict:
    """
    Load or build dataset based on configuration.

    This function either loads a preprocessed dataset from a given path or processes
    raw data and builds the dataset from scratch using the configurations provided.
    The final dataset is saved as a CSV file to the specified output directory.

    Args:
        cfg (DictConfig): Configuration object containing dataset builder and raw data
        processor settings.
        out_dir (str): Directory where the resulting dataset CSV will be saved.
        logger (Logger): Logger instance for logging progress and messages.

    Returns:
        dict: A dictionary representing the processed dataset.
    """
    if cfg.dataset_builder_config.processed_dataset:
        logger.info(f"load data from {cfg.dataset_builder_config.processed_dataset}")
        dataset_builder = DatasetBuilder(cfg.dataset_builder_config, None, logger)
        dataset_builder.load_variant_data(cfg.dataset_builder_config.processed_dataset)
        dataset = dataset_builder.return_dataset()
    else:
        raw_data_processor = RawDataProcessor(cfg.raw_data_processor_config, logger)
        variant_df = raw_data_processor.process_raw_data()

        dataset_builder = DatasetBuilder(
            cfg.dataset_builder_config, variant_df.copy(), logger
        )
        dataset = dataset_builder.build_dataset()

    dataset_builder.write_dataset(os.path.join(out_dir, "dataset.csv"))

    return dataset


if __name__ == "__main__":
    main()
