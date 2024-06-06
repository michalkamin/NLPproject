import unittest
import pandas as pd
from transformers import T5Tokenizer
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
from models.NewsHeadlineDataModule import NewsHeadlineDataModule


class TestNewsHeadlineDataModule(unittest.TestCase):
    """
    Unit tests for the NewsHeadlineDataModule class.
    """

    def setUp(self) -> None:
        """
        Set up the test case environment with sample data and a tokenizer.
        """
        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.sample_train_data: pd.DataFrame = pd.DataFrame({
            "Article": ["Train article 1", "Train article 2"],
            "Headline": ["Train headline 1", "Train headline 2"]
        })
        self.sample_test_data: pd.DataFrame = pd.DataFrame({
            "Article": ["Test article 1", "Test article 2"],
            "Headline": ["Test headline 1", "Test headline 2"]
        })
        self.sample_val_data: pd.DataFrame = pd.DataFrame({
            "Article": ["Val article 1", "Val article 2"],
            "Headline": ["Val headline 1", "Val headline 2"]
        })

        self.data_module: NewsHeadlineDataModule = NewsHeadlineDataModule(
            train_df=self.sample_train_data,
            test_df=self.sample_test_data,
            val_df=self.sample_val_data,
            tokenizer=self.tokenizer,
            batch_size=2,
            article_max_token_len=512,
            headline_max_token_len=128
        )

    def test_setup(self) -> None:
        """
        Test the setup method of the data module.
        """
        self.data_module.setup()

        self.assertIsNotNone(self.data_module.train_dataset)
        self.assertIsNotNone(self.data_module.val_dataset)
        self.assertIsNotNone(self.data_module.test_dataset)
        self.assertEqual(len(self.data_module.train_dataset),
                         len(self.sample_train_data))
        self.assertEqual(len(self.data_module.val_dataset),
                         len(self.sample_val_data))
        self.assertEqual(len(self.data_module.test_dataset),
                         len(self.sample_test_data))

    def test_train_dataloader(self) -> None:
        """
        Test the train_dataloader method of the data module.
        """
        self.data_module.setup()
        train_loader: DataLoader = self.data_module.train_dataloader()

        batch: Optional[Dict[str, Any]] = next(iter(train_loader), None)
        self.assertIsNotNone(batch)
        if batch is not None:
            self.assertEqual(len(batch["article_input_ids"]), 2)

    def test_val_dataloader(self) -> None:
        """
        Test the val_dataloader method of the data module.
        """
        self.data_module.setup()
        val_loader: DataLoader = self.data_module.val_dataloader()

        batch: Optional[Dict[str, Any]] = next(iter(val_loader), None)
        self.assertIsNotNone(batch)
        if batch is not None:
            self.assertEqual(len(batch["article_input_ids"]), 2)

    def test_test_dataloader(self) -> None:
        """
        Test the test_dataloader method of the data module.
        """
        self.data_module.setup()
        test_loader: DataLoader = self.data_module.test_dataloader()

        batch: Optional[Dict[str, Any]] = next(iter(test_loader), None)
        self.assertIsNotNone(batch)
        if batch is not None:
            self.assertEqual(len(batch["article_input_ids"]), 2)


if __name__ == '__main__':
    unittest.main()
