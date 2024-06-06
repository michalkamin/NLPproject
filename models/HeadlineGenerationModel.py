import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from transformers import (
    T5ForConditionalGeneration,
    BartForConditionalGeneration
    )
from transformers.modeling_outputs import Seq2SeqLMOutput
from typing import Dict, Tuple


class HeadlineGenerationModel(pl.LightningModule):
    """
    A PyTorch Lightning Module for headline generation using T5 or BART models.

    Args:
        model_name (str): The name of the model to use
        ("t5-small" or "facebook/bart-base").
    """

    def __init__(self, model_name: str = "t5-small") -> None:
        super().__init__()
        if model_name.startswith("t5"):
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_name, return_dict=True
            )
        elif model_name.startswith("facebook/bart"):
            self.model = BartForConditionalGeneration.from_pretrained(
                model_name, return_dict=True
            )
        else:
            raise ValueError(
                "model_name should be 't5-small' or 'facebook/bart-base'"
            )
        self.val_loss: list = []
        self.val_loss_epoch: list = []
        self.validation_step_outputs: list = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the model.

        Args:
            input_ids (torch.Tensor): Input IDs.
            attention_mask (torch.Tensor): Attention mask for the inputs.
            labels (torch.Tensor): Labels for the inputs.
            decoder_attention_mask (torch.Tensor): Decoder attention mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Loss and logits from the model.
        """
        output: Seq2SeqLMOutput = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )
        return output.loss, output.logits

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Training step for the model.

        Args:
            batch (Dict[str, torch.Tensor]): Batch of data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss for the training step.
        """
        input_ids = batch["article_input_ids"]
        attention_mask = batch["article_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Validation step for the model.

        Args:
            batch (Dict[str, torch.Tensor]): Batch of data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss for the validation step.
        """
        input_ids = batch["article_input_ids"]
        attention_mask = batch["article_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.validation_step_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self) -> None:
        """
        Actions to perform at the end of the validation epoch.
        """
        epoch_average = torch.stack(self.validation_step_outputs).mean()
        self.log("validation_epoch_average", epoch_average)
        self.validation_step_outputs.clear()

    def test_step(self,
                  batch: Dict[str, torch.Tensor],
                  batch_idx: int) -> torch.Tensor:
        """
        Test step for the model.

        Args:
            batch (Dict[str, torch.Tensor]): Batch of data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss for the test step.
        """
        input_ids = batch["article_input_ids"]
        attention_mask = batch["article_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self) -> AdamW:
        """
        Configures the optimizer for the model.

        Returns:
            AdamW: The optimizer.
        """
        return AdamW(self.parameters(), lr=0.0001)
