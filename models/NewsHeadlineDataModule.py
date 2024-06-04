import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
from typing import Optional
from models.NewsHeadlineDataset import NewsHeadlineDataset


class NewsHeadlineDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for handling the
    data loading and preprocessing for the news headline dataset.

    Args:
        train_df (pd.DataFrame): DataFrame containing the training data.
        test_df (pd.DataFrame): DataFrame containing the test data.
        val_df (pd.DataFrame): DataFrame containing the validation data.
        tokenizer (T5Tokenizer): The tokenizer used for tokenizing the text.
        batch_size (int, optional): Batch size for the dataloaders.
        Defaults to 8.
        article_max_token_len (int, optional): Maximum token length for articles.
        Defaults to 512.
        headline_max_token_len (int, optional): Maximum token length for headlines.
        Defaults to 128.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        val_df: pd.DataFrame,
        tokenizer: T5Tokenizer,
        batch_size: int = 8,
        article_max_token_len: int = 512,
        headline_max_token_len: int = 128
    ) -> None:
        super().__init__()

        self.train_df = train_df
        self.test_df = test_df
        self.val_df = val_df
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.article_max_token_len = article_max_token_len
        self.headline_max_token_len = headline_max_token_len

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Sets up the datasets for training, validation, and testing.

        Args:
            stage (Optional[str], optional): The stage for which the setup is being done.
            Defaults to None.
        """
        self.train_dataset = NewsHeadlineDataset(
            self.train_df,
            self.tokenizer,
            self.article_max_token_len,
            self.headline_max_token_len
        )

        self.test_dataset = NewsHeadlineDataset(
            self.test_df,
            self.tokenizer,
            self.article_max_token_len,
            self.headline_max_token_len
        )

        self.val_dataset = NewsHeadlineDataset(
            self.val_df,
            self.tokenizer,
            self.article_max_token_len,
            self.headline_max_token_len
        )

    def train_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the training dataset.

        Returns:
            DataLoader: The DataLoader for the training dataset.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the validation dataset.

        Returns:
            DataLoader: The DataLoader for the validation dataset.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the test dataset.

        Returns:
            DataLoader: The DataLoader for the test dataset.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
