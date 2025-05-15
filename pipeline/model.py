"""
This module contains the implementation of the ClearVariant and related classes.
"""

import os
import sys
from typing import Literal

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_utils import PretrainedConfig
from transformers.models.esm.modeling_esm import EsmModel, EsmPreTrainedModel

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

sys.path.append(ROOT_DIR)


class ClearVariant(EsmPreTrainedModel):
    """
    ClearVariant is a model class for sequence classification using ESM models. It
    combines reference and variant sequences for classification tasks.
    """

    def __init__(self, pretrained_model_name: str, num_labels: int, problem_type: str):
        esm_ref = EsmModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name,
            num_labels=num_labels,
            add_pooling_layer=False,
        )
        model_config = esm_ref.config
        model_config.num_labels = num_labels
        super().__init__(model_config)

        self.problem_type = problem_type
        self.model_config = model_config
        self.num_labels = num_labels
        self.esm_ref = esm_ref
        self.esm_var = EsmModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name,
            num_labels=num_labels,
            add_pooling_layer=False,
        )

        self.classifier = AttnConcatHead(
            self.model_config, self.model_config.hidden_size * 2
        )

        self.init_weights()

    def forward(
        self,
        ref_input_ids: torch.LongTensor,
        ref_attention_mask: torch.Tensor,
        var_input_ids: torch.LongTensor,
        var_attention_mask: torch.Tensor,
        labels: torch.LongTensor | None = None,
    ) -> SequenceClassifierOutput:
        """
        Perform a forward pass through the model.

        Args:
            ref_input_ids (torch.LongTensor): Input IDs for the reference sequence.
            ref_attention_mask (torch.Tensor): Attention mask for the reference
                sequence.
            var_input_ids (torch.LongTensor): Input IDs for the variant sequence.
            var_attention_mask (torch.Tensor): Attention mask for the variant sequence.
            labels (torch.LongTensor, optional): Ground truth labels for the input
                sequences. Defaults to None.

        Returns:
            SequenceClassifierOutput: The output containing the loss and logits.
        """
        output_ref = self.esm_ref(ref_input_ids, attention_mask=ref_attention_mask)
        output_var = self.esm_var(var_input_ids, attention_mask=var_attention_mask)

        logits = self.classifier(output_ref[0], output_var[0])

        loss = compute_loss(logits, labels, self.num_labels, self.problem_type)

        return SequenceClassifierOutput(loss=loss, logits=logits)


class AttnConcatHead(nn.Module):
    """
    Attention-based concatenation head for sequence classification.

    This class implements a multi-head attention mechanism to concatenate
    reference and variant vectors for sequence classification tasks.
    """

    def __init__(self, config: PretrainedConfig, hidden_embedding_dim: int):
        """
        Init function for AttnConcatHead.

        Args:
            config (PretrainedConfig): Configuration object containing model parameters.
            hidden_embedding_dim (int): Size of hidden_embedding_dim.
        """
        super().__init__()
        self.mul_attn = nn.MultiheadAttention(
            embed_dim=hidden_embedding_dim,
            num_heads=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_embedding_dim, config.num_labels)

    def forward(
        self, ref_vector: torch.Tensor, var_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform a forward pass through the model.

        Args:
            ref_vector (torch.Tensor): Reference tensor input.
            var_vector (torch.Tensor): Variable tensor input.

        Returns:
            torch.Tensor: Output tensor after processing through the model.
        """

        x1 = var_vector
        x2 = ref_vector
        x = torch.concat([x1, x2], dim=-1)
        x = self.dropout(x)
        x, self.attn_output_weights = self.mul_attn(query=x, key=x, value=x)
        x = x[:, 0, :]  # take [CLS] token
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_labels: int,
    problem_type: (
        Literal[
            "regression",
            "single_label_classification",
            "multi_label_classification",
        ]
        | None
    ) = None,
) -> torch.FloatTensor:
    """
    Compute the loss between the predicted logits and the true labels based on the
        problem type.

    Args:
        logits (torch.Tensor): The predicted logits from the model.
        labels (torch.Tensor): The true labels.
        num_labels (int): The number of labels/classes.
        problem_type (str): The type of problem. If None, it will be inferred based on
            the number of labels and the dtype of the labels.

    Returns:
        torch.FloatTensor: The computed loss.
    """
    loss = None
    if labels is not None:
        labels = labels.to(logits.device)

        if problem_type is None:
            if num_labels == 1:
                problem_type = "regression"
            elif num_labels > 1 and (
                labels.dtype == torch.long or labels.dtype == torch.int
            ):
                problem_type = "single_label_classification"
            else:
                problem_type = "multi_label_classification"

        if problem_type == "regression":
            loss_fct = MSELoss()
            if num_labels == 1:
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(logits, labels)
        elif problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        elif problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

    return loss
