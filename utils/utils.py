"""
This module provides utility functions for logging.

Functions:
    get_logger(log_dir_path: str, version: str) -> logging.Logger:
        Get logger object.
    write_config(config: dict, config_path: str) -> None:
        Write config information into a file.
"""

import logging
import os
import sys

UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(UTILS_DIR)
sys.path.append(ROOT_DIR)


def get_logger(log_dir_path: str, version: str) -> logging.Logger:
    """
    Get logger object

    Note:
        if log_dir_path does not exist, it makes the directory in that path.

    Args:
        log_dir_path (str): log directory path
        version (str): version of execution

    Returns:
        logger (logging.Logger): logger object

    Examples:
        >>> log_dir_path = os.path.join(
                ROOT_DIR,
                "logs",
                os.path.splitext(os.path.basename(__file__))[0]
            )
        >>> get_logger(log_dir_path)
        "logs/{module}/{timestamp}.log"
    """
    os.makedirs(log_dir_path, exist_ok=True)

    log_file_path = os.path.join(log_dir_path, f"{version}.log")
    logger_formatter = logging.Formatter(
        fmt="{asctime}\t{name}\t{filename}:{lineno}\t{levelname}\t{message}",
        datefmt="%Y-%m-%dT%H:%M:%S",
        style="{",
    )

    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path, exist_ok=True)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logger_formatter)
    stream_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(filename=log_file_path)
    file_handler.setFormatter(logger_formatter)
    file_handler.setLevel(logging.DEBUG)

    logger = logging.getLogger(version)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger
