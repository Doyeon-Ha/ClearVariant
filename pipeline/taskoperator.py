"""
This module defines the TaskOperator class which handles training, inference,
and evaluation tasks for a machine learning model.
"""

import os
import sys
import time
from logging import Logger

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler
from tqdm.auto import tqdm
from transformers import get_scheduler, modeling_outputs

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

sys.path.append(ROOT_DIR)

from pipeline.model import compute_loss
from pipeline.modeladapter import ModelAdapter
from utils import log_metrics


class TaskOperator:
    """
    Handles training, inference, and evaluation tasks for a machine learning model.
    """

    def __init__(
        self,
        out_dir: str,
        model_adapter: ModelAdapter,
        config: dict,
        dataset: dict,
        logger: Logger,
        device: int,
    ):
        self.out_dir = out_dir
        self.model_adapter = model_adapter
        self.config = config
        self.task = self.config.task
        self.dataset = DatasetDict(
            {
                "train": Dataset.from_dict(dataset["train"]),
                "test": Dataset.from_dict(dataset["test"]),
            }
        )
        self.logger = logger

        self.dataloader = dict()
        self.optimizer = None
        self.scheduler = None

        self.file_write_index = 0
        self.step = 0

        self.device = device

        self._build_dataloader()

    def _build_dataloader(self) -> None:
        """Build dataloader with dataset."""
        for data_split in self.dataset:
            if len(self.dataset[data_split]) > 0:
                args = self._get_dataloader_argument(data_split)
                self.dataloader[data_split] = DataLoader(
                    self.dataset[data_split], **args
                )

        return

    def _get_dataloader_argument(self, data_split: str):
        """Generate and return arguments for dataloader.

        Note:
            data_split infers whether this is train or test.
            if it is train and classification, datasampler will work.
            if it is train and regression, only random shuffeling
        """
        args = {
            "batch_size": self.config.batch_size,
        }
        if data_split == "train":
            if isinstance(self.dataset["train"]["labels"][0], float):
                args["shuffle"] = True
            else:
                args["sampler"] = ImbalancedDatasetSampler(
                    self.dataset["train"],
                    labels=[str(x) for x in self.dataset["train"]["labels"]],
                )
        else:
            args["shuffle"] = False

        return args

    def do_task(self) -> None:
        """train, inference, and write context vector will be done."""
        if self.config.task == "train":
            self.train()
        elif self.config.task == "inference":
            self.inference()
        elif self.config.task == "write":
            self.write_attn_vector()

    def train(self) -> None:
        """For each epoch, train and evaluate"""
        self._set_up_train()
        self.logger.info("Start Training")
        for epoch in range(self.config.epochs):
            start_time = time.time()
            self._run_batch_train(epoch)
            self._run_batch_eval(epoch)
            end_time = time.time()
            log_metrics.log_metrics(
                {"epoch_time": (end_time - start_time), "epoch": epoch},
                step=epoch,
            )
        log_metrics.end_run()

        return

    def _set_up_train(self) -> None:
        """Set up optimizer and learning rate scheduler for train task."""
        num_training_steps = self.config.epochs * len(self.dataloader["train"])
        self.optimizer = AdamW(
            self.model_adapter.parameters(), lr=self.config.learning_rate
        )
        self.scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        exp_name, run_name, params = self._get_run_info()
        log_metrics.start_run(exp_name, run_name, params)

        return

    def _get_run_info(self) -> tuple:
        """Get run information for logging."""
        params = dict()
        for key, value in self.config.items():
            params[key] = value

        sub_dir_list = self.out_dir.split("/")

        db_name = sub_dir_list[-4]
        db_option = sub_dir_list[-3]
        model = sub_dir_list[-2]
        date = sub_dir_list[-1]

        params["db_name"] = db_name
        params["db_split"] = db_option.split(".")[0]
        params["db_option"] = db_option.split(".")[1]
        params["model"] = model
        params["date"] = date

        exp_name = f"{db_name}"
        run_name = f"{db_option}.{model}.{date}"

        return exp_name, run_name, params

    def _run_batch_train(self, epoch: int) -> None:
        """Per each batch, do forward and backward with gradient update"""
        self.logger.info(f"Train Epoch {epoch}")

        loss_list = list()
        self.model_adapter.set_train_mode()
        for batch in tqdm(self.dataloader["train"]):
            outputs = self.model_adapter.forward(batch)
            self._backward(outputs)
            loss_list.append(outputs.loss.item())
            log_metrics.log_metrics(
                {"batch train loss": outputs.loss.item()}, step=self.step
            )
            self.step += 1
        log_metrics.log_metrics(
            {"epoch train loss": np.mean(loss_list), "epoch": epoch}, step=epoch
        )

        return

    def _backward(self, outputs: modeling_outputs) -> None:
        """With outputs of model, calculate loss and update model."""
        outputs.loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        return

    def _run_batch_eval(self, epoch: int) -> None:
        """Without model update, only calculate output and performance"""
        self.logger.info(f"Eval Epoch {epoch}")

        all_preds_list = list()
        all_labels_list = list()
        self.model_adapter.set_eval_mode()
        with torch.no_grad():
            for batch in tqdm(self.dataloader["test"]):
                outputs = self.model_adapter.forward(batch)
                all_preds_list.append(outputs.logits)
                all_labels_list.append(
                    self.model_adapter._process_label(dict(), batch)["labels"]
                )
        if self.config.task == "train":
            self._log_performance(all_preds_list, all_labels_list, epoch)
            self._save_model(epoch)
        elif self.config.task == "inference":
            self._write_performance_file(all_preds_list, all_labels_list)

        return

    def _log_performance(
        self, logits_list: list, labels_list: list, epoch: int
    ) -> None:
        """Log all calculated performances.

        Args:
            logits_list: list of logits
            labels_list: list of labels
            epoch: int, current epoch
        """
        metrics = {
            "epoch": epoch,
        }

        pred_batch = torch.cat(logits_list, dim=0)
        label_batch = torch.cat(labels_list, dim=0)

        metrics["eval loss"] = compute_loss(
            pred_batch,
            label_batch,
            self.model_adapter.config.num_labels,
            self.model_adapter.config.problem_type,
        )
        num_classes = self.model_adapter.config.num_labels

        self.logger.info(f"The number of classes: {num_classes}")

        if num_classes == 1:
            metrics["r2_score"] = r2_score(
                label_batch.cpu().numpy(), pred_batch.cpu().numpy()
            )
            metrics["eval loss"] = compute_loss(
                pred_batch,
                label_batch,
                self.model_adapter.config.num_labels,
                self.model_adapter.config.problem_type,
            )
        else:
            score_batch = F.softmax(pred_batch, dim=1)[:, 1]

            pred_batch_argmax = torch.argmax(pred_batch, axis=1)
            label_batch_argmax = torch.argmax(label_batch, axis=1)

            prediction_list = pred_batch_argmax.cpu().squeeze().tolist()
            label_list = label_batch_argmax.cpu().squeeze().tolist()
            score_list = score_batch.cpu().squeeze().tolist()

            self.logger.info(f"samples: {label_list[:10]}")
            self.logger.info(f"preds: {prediction_list[:10]}")
            label_counts = {
                label.item(): (label_batch_argmax == label).sum().item()
                for label in torch.unique(label_batch_argmax)
            }
            self.logger.info(f"Label counts: {label_counts}")
            pred_counts = {
                pred.item(): (pred_batch_argmax == pred).sum().item()
                for pred in torch.unique(pred_batch_argmax)
            }
            self.logger.info(f"Prediction counts: {pred_counts}")

            for cls in range(num_classes):
                metrics[f"precision_{cls}"] = precision_score(
                    label_list, prediction_list, average=None
                )[cls]
                metrics[f"recall_{cls}"] = recall_score(
                    label_list, prediction_list, average=None
                )[cls]
                metrics[f"f1_{cls}"] = f1_score(
                    label_list, prediction_list, average=None
                )[cls]

                if num_classes == 2:
                    metrics[f"auprc_{cls}"] = average_precision_score(
                        [(1 - x) if cls == 0 else x for x in label_list],
                        [(1 - x) if cls == 0 else x for x in score_list],
                    )

            if num_classes == 2:
                metrics["auroc"] = roc_auc_score(label_list, score_list)

            metrics["accuracy"] = accuracy_score(label_list, prediction_list)

            cm = confusion_matrix(label_list, prediction_list)
            for i in range(self.model_adapter.config.num_labels):
                for j in range(self.model_adapter.config.num_labels):
                    metrics[f"true.{i}/pred.{j}"] = cm[i, j]

        log_metrics.log_metrics(metrics, step=epoch)

        return

    def _save_model(self, epoch: int) -> None:
        """Save Each Epoch's model parameters under
        {self.out_dir}/model_param/{epoch}.
        """
        param_out_dir = os.path.join(self.out_dir, "model_param", str(epoch))
        os.makedirs(param_out_dir, exist_ok=True)
        self.model_adapter.save_model(param_out_dir)

        return

    def _write_performance_file(self, logits_list: list, labels_list: list) -> None:
        """Write label and prediction result as file."""
        column_list = ["labels", "predictions"]
        with open(os.path.join(self.out_dir, "classification_result.tsv"), "w") as f:
            f.write("\t".join(column_list) + "\n")
            for logit, label in zip(logits_list, labels_list):
                logit = logit.cpu().tolist()
                label = label.cpu().tolist()
                for one_pred, one_label in zip(logit, label):
                    f.write(f"{one_label}\t{one_pred}\n")

        return

    def inference(self) -> None:
        """Make Model Result for test data."""
        self._run_batch_eval(0)

        return

    def _write_vector(
        self,
        batch: dict,
        vector: list,
        data_split: str,
        sub_dir: str | None = None,
    ) -> dict:
        """Write vector one sample by one sample into .npz files.

        Args:
            batch: dict, batch data
            vector: list, vector data
            data_split: str, data split
            sub_dir: str, sub directory. Defaults to None.

        Returns:
            dict_to_write: dict, dictionary to write
        """

        # Step 1: Label processing
        dict_to_write = {}
        if self.model_adapter.config.num_labels == 1:
            dict_to_write["labels"] = [x.item() for x in batch["labels"]]
        else:
            dict_to_write["labels"] = [
                x.argmax().item()
                if isinstance(x, torch.Tensor)
                else torch.tensor(x).argmax().item()
                for x in batch["labels"]
            ]

        # Step 2: Copy other arrays to write
        dict_to_write["GeneName"] = batch["GeneName"]
        dict_to_write["refseqID"] = batch["refseqID"]
        dict_to_write["HGVSp"] = batch["HGVSp"]
        dict_to_write["vector"] = vector

        # Step 3: save per-sample (npz)
        for i in range(len(dict_to_write["labels"])):
            per_sample = {k: v[i] for k, v in dict_to_write.items()}

            # output path
            if sub_dir is None:
                output_path = os.path.join(
                    self.context_vector_dir,
                    f"{data_split}_{self.file_write_index:07}.npz",
                    # leading zeros for sorting
                )
            else:
                output_path = os.path.join(
                    self.out_dir,
                    sub_dir,
                    f"{data_split}_{self.file_write_index:07}.npz",
                    # leading zeros for sorting
                )

            # np.savez (in binary)
            np.savez(output_path, **per_sample)
            self.file_write_index += 1

        return dict_to_write

    def write_attn_vector(self) -> None:
        """Write the attention vector, which comes from attn classifier.
        It will be processed for all samples.
        """
        self.logger.info("Start Attn Out")
        self.model_adapter.set_eval_mode()
        self.file_write_index = 0

        for batch in tqdm(self.dataloader["test"]):
            with torch.no_grad():
                attn_vector = self.model_adapter.get_attn_vector(batch)
                self._write_vector(batch, attn_vector, "test", "attn_out")

        return
