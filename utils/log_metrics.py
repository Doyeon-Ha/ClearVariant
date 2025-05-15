"""This module is used to log metrics. Remove dependency on file, or mlflow etc."""

import csv
import inspect
import os

import mlflow

MLFLOW_URI = os.environ.get("MLFLOW_URI", "ENV_NOT_EXIST")

TEAN_NAME = "ClearVariant"
TRACKING_TOOL = None
_RUN_LOG_FILE_PATH = None


def start_run(
    exp_name: str, run_name: str, params: dict, tracking_tool: str = "file"
) -> None:
    """Start a run with the specified tracking tool.

    Args:
        tracking_tool (str): name of the tracking tool
        exp_name (str): Experiment name which can contain several runs.
            recommended to specify the dataset. e.g. LOFGOF_variant:2021_5fold
        run_name (str): Run name. recommended to specify the model name and timestamp.
            e.g. ESM2:doubleattn_2025:01:01:00:00:00
        params (dict): Dictionary of parameters to log.

    Raises:
        ValueError: If the tracking tool is not supported.
    """
    global TRACKING_TOOL

    exp_name_with_team = f"{TEAN_NAME}.{exp_name}"

    try:
        start_run_func = globals()[f"_start_run_{tracking_tool}"]
        start_run_func(exp_name_with_team, run_name, params)
        TRACKING_TOOL = tracking_tool
    except KeyError:
        raise ValueError(
            f"Unsupported tracking tool: {tracking_tool}\n"
            + f"Supported tracking tools: {_get_possible_tools('_start_run_')}"
        )

    return


def _get_possible_tools(prefix: str) -> list:
    """Get possible tools that can be used.

    Note:
        This function is used to get possible tools by checking the prefix
        of the function name.

    Args:
        prefix (str): Prefix of the function name.

    Returns:
        list: List of possible tools.
    """
    current_module = inspect.getmodule(_get_possible_tools)
    all_functions = inspect.getmembers(current_module, inspect.isfunction)

    possible_tools = [
        name.replace(prefix, "") for name, _ in all_functions if name.startswith(prefix)
    ]

    return possible_tools


def log_metrics(metrics: dict, step: int) -> None:
    """Log metrics with the specified tracking tool.

    Note:
        You must call start_run before calling this function.

    Args:
        metrics (dict): Dictionary of metrics to log.
        step (int): Step number.

    Raises:
        ValueError: If start_run is not called before this function.
    """

    try:
        log_metrics_func = globals()[f"_log_metrics_{TRACKING_TOOL}"]
        log_metrics_func(metrics, step)
    except KeyError:
        raise ValueError("You must call start_run before log_metrics")

    return


def end_run() -> None:
    """End the run with the specified tracking tool.

    Note:
        You must call start_run before calling this function.

    Raises:
        ValueError: If start_run is not called before this function.
    """
    global TRACKING_TOOL
    try:
        end_run_func = globals()[f"_end_run_{TRACKING_TOOL}"]
        end_run_func()
        TRACKING_TOOL = None
    except KeyError:
        raise ValueError("You must call start_run before end_run")

    return


def _start_run_file(exp_name: str, run_name: str, params: dict) -> None:
    """Start a run with file logging.

    Creates a directory for the experiment and a file for the run.
    Writes parameters at the top of the file.

    Args:
        exp_name (str): Experiment name (with team).
        run_name (str): Run name.
        params (dict): Parameters to log.
    """
    global _RUN_LOG_FILE_PATH

    output_dir = os.path.join("logged_metric", exp_name)
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{run_name}.tsv")
    _RUN_LOG_FILE_PATH = file_path

    with open(file_path, mode="w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["type", "key", "value"])
        for key, value in params.items():
            writer.writerow(["param", key, value])


def _log_metrics_file(metrics: dict, step: int) -> None:
    """Append metrics to the run file.

    Args:
        metrics (dict): Metrics to log.
        step (int): Step number.
    """
    if _RUN_LOG_FILE_PATH is None:
        raise ValueError("File path not initialized. Did you call start_run()?")

    with open(_RUN_LOG_FILE_PATH, mode="a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for key, value in metrics.items():
            writer.writerow(["metric", f"{key}@step{step}", value])


def _end_run_file() -> None:
    """End the run for file logging. Nothing to do here."""
    return


def _start_run_mlflow(exp_name: str, run_name: str, params: dict) -> None:
    """Start a run with mlflow.

    Args:
        exp_name (str): Experiment name which can contain several runs.
            recommended to specify the dataset. e.g. LOFGOF_variant:2021_5fold
        run_name (str): Run name. recommended to specify the model name and timestamp.
            e.g. ESM2:doubleattn_2025:01:01:00:00:00
        params (dict): Dictionary of parameters to log.

    """
    mlflow.set_tracking_uri(MLFLOW_URI)

    client = mlflow.MlflowClient(MLFLOW_URI)
    experiment = client.get_experiment_by_name(exp_name)
    if experiment is None:
        _return_code = client.create_experiment(exp_name)
        experiment = client.get_experiment_by_name(exp_name)

    mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name)
    mlflow.log_params(params)

    return


def _log_metrics_mlflow(metrics: dict, step: int) -> None:
    """Log metrics with mlflow.

    Note:
        You must call start_run before calling this function.

    Args:
        metrics (dict): Dictionary of metrics to log.
        step (int): Step number.
    """
    mlflow.log_metrics(metrics, step=step)

    return


def _end_run_mlflow() -> None:
    """End the run with mlflow."""
    mlflow.end_run()

    return
