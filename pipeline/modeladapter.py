"""
This module provides a template for model adapters, including methods for
loading models, processing inputs, and handling device configurations.
"""

import os
import sys
from typing import Generator

import safetensors
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn.parameter import Parameter
from transformers import AutoTokenizer, modeling_outputs

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

sys.path.append(ROOT_DIR)

from pipeline.model import ClearVariant


class ModelAdapter:
    """
    A template class for model adapters, providing methods for loading models,
    processing inputs, and handling device configurations.
    """

    def __init__(self, config: DictConfig, device: int):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = device

        self._load_model_N_tokenizer()
        if self.config.pretrained_model != self.config.model_checkpoint:
            self._get_model_checkpoint()

    def _load_model_N_tokenizer(self) -> None:
        """Load model and tokenizer"""
        self.model = ClearVariant(
            self.config.pretrained_model,
            self.config.num_labels,
            self.config.problem_type,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_model)
        self.model.to(self.device)

        return

    def _get_model_checkpoint(self) -> None:
        """If we get fine-tunned model, get model safetensors."""
        tensors = {}
        with safetensors.safe_open(self.config.model_checkpoint, framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        self.model.load_state_dict(tensors)

        return

    def forward(self, batch: dict) -> modeling_outputs:
        """For each batch, do the forward function.

        Args:
            batch (dict): Input of DL model. The content may vary by model.

        Returns:
            modeling_outputs: Output of DL model.
        """
        processed_batch = self._process_input(batch)
        outputs = self.model(**processed_batch)

        return outputs

    def _process_input(self, batch: dict) -> dict:
        """Classifier need tokenizing and label transpose.
        Args:
            batch (dict): {"sequence": ["AAA", "BBB"],
                           "RefSeq": ["AAA", "BBB"],
                           "labels": [[0, 1], [1, 0], [0, 0]]}
        Returns:
            dict: {"ref_input_ids": [...],
                   "ref_attention_mask": [...],
                   "var_input_ids": [...],
                   "var_attention_mask": [...],
                   "labels": [[0, 1, 0], [1, 0, 0]], ...}
        """
        processed_batch = dict()

        ref_processed_batch = self._tokenizing(batch["RefSeq"])
        processed_batch["ref_input_ids"] = ref_processed_batch["input_ids"]
        processed_batch["ref_attention_mask"] = ref_processed_batch["attention_mask"]

        var_processed_batch = self._tokenizing(batch["sequence"])
        processed_batch["var_input_ids"] = var_processed_batch["input_ids"]
        processed_batch["var_attention_mask"] = var_processed_batch["attention_mask"]

        processed_batch = self._process_label(processed_batch, batch)

        return processed_batch

    def _tokenizing(self, sequence: list) -> dict:
        """Tokenizing the input sequence. One of the function to process input.

        Note:
            Frequently used on self._process_input. Not all sub class.

        Args:
            sequence (list): list of protein sequence.

        Returns:
            dict: tokenized input. result may vary by tokenizer.
        """
        return self.tokenizer(
            sequence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.config.max_seq_len + 2,
        ).to(self.device)

    def _process_label(self, processed_batch: dict, batch: dict) -> dict:
        """Process label dependes on the types of label.

        Note:
            if label is string, it will still stiring.
            if label is int, it will still int.
            if label is float64, it will be changed to float32
            if label is one hot encoded, it will be transposed.

        Args:
            processed_batch (dict): base for output. add label here.
            batch (dict): data to construct input

        Returns:
            dict: processed_batch with processed labels.
        """
        processed_batch["labels"] = batch["labels"]
        if isinstance(batch["labels"][0], torch.Tensor):
            if len(batch["labels"][0].shape) == 0:
                if batch["labels"].dtype == torch.float64:
                    processed_batch["labels"] = batch["labels"].float()
            elif len(batch["labels"][0].shape) == 1:
                processed_batch["labels"] = torch.stack(batch["labels"]).transpose(0, 1)

        return processed_batch

    def get_attn_vector(self, batch: dict) -> list:
        """Return attn_vector.

        Args:
            batch (dict): {"sequence": ["AAA", "BBB"],
                           "RefSeq": ["AAA", "BBB"],
                           "labels": [[0, 1], [1, 0], [0, 0]]}

        Returns:
            list: list converted attn vector [batch_size, seqlen, seqlen]
        """
        processed_batch = self._process_input(batch)
        self.model(**processed_batch)

        return self.model.classifier.attn_output_weights.tolist()

    def set_train_mode(self) -> None:
        """Set model to training mode."""
        self._set_module_mode(self.model, requires_grad=True, train_mode=True)

        return

    def set_eval_mode(self) -> None:
        """Set model to evaluation mode."""
        self._set_module_mode(self.model, requires_grad=False, train_mode=False)

        return

    def _set_module_mode(
        self,
        module: nn.Module,
        requires_grad: bool,
        train_mode: bool,
    ) -> None:
        """
        Set module's training mode and gradient requirements.

        Args:
            module (nn.Module): module to set training mode.
            requires_grad (bool): True if module requires gradient.
            train_mode (bool): True if module is in training mode.
        """
        module.requires_grad_(requires_grad)
        module.train() if train_mode else module.eval()

        return

    def parameters(self) -> Generator[Parameter, None, None]:
        """Return the parameters of model.

        Returns:
            Generator[Parameter]: Generator iterate tensors which represent
            the parameter of each layer.
        """
        return self.model.parameters()

    def save_model(self, output_dir: str) -> None:
        """Save model parameter to the output path

        Note:
            it will create several files inside the directory.
            model can be loaded with model.from_pretrained() by giving
            directory path as model name

        Args:
            output_dir (str): path for output directory
        """
        self.model.save_pretrained(output_dir)

        return
