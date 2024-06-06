import unittest
import pandas as pd
from transformers import T5Tokenizer
from torch.utils.data import DataLoader
from typing import Any, Dict
from models.NewsHeadlineDataset import NewsHeadlineDataset


class TestNewsHeadlineDataset(unittest.TestCase):
    """
    Unit tests for the NewsHeadlineDataset class.
    """

    def setUp(self) -> None:
        """
        Set up the test case environment with sample data and a tokenizer.
        """
        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.sample_data: pd.DataFrame = pd.DataFrame({
            "Article": ["Sample article 1", "Sample article 2"],
            "Headline": ["Sample headline 1", "Sample headline 2"]
        })
        self.dataset: NewsHeadlineDataset = NewsHeadlineDataset(
            data=self.sample_data,
            tokenizer=self.tokenizer,
            article_max_token_len=512,
            headline_max_token_len=128
        )

    def test_len(self) -> None:
        """
        Test the length of the dataset.
        """
        expected_length: int = len(self.sample_data)
        actual_length: int = len(self.dataset)
        self.assertEqual(actual_length, expected_length)

    def test_getitem(self) -> None:
        """
        Test the __getitem__ method of the dataset.
        """
        sample: Dict[str, Any] = self.dataset[0]
        expected_article: str = self.sample_data.iloc[0]["Article"]
        expected_headline: str = self.sample_data.iloc[0]["Headline"]

        self.assertIn("article", sample)
        self.assertIn("headline", sample)
        self.assertIn("article_input_ids", sample)
        self.assertIn("article_attention_mask", sample)
        self.assertIn("labels", sample)
        self.assertIn("labels_attention_mask", sample)

        self.assertEqual(sample["article"], expected_article)
        self.assertEqual(sample["headline"], expected_headline)
        self.assertEqual(len(sample["article_input_ids"]), 512)
        self.assertEqual(len(sample["article_attention_mask"]), 512)
        self.assertEqual(len(sample["labels"]), 128)
        self.assertEqual(len(sample["labels_attention_mask"]), 128)

    def test_dataloader(self) -> None:
        """
        Test if the dataset can be properly loaded into a DataLoader.
        """
        batch_size: int = 2
        dataloader: DataLoader = DataLoader(self.dataset,
                                            batch_size=batch_size)
        batch: Dict[str, Any] = next(iter(dataloader))

        self.assertEqual(batch["article_input_ids"].shape,
                         (batch_size, 512))
        self.assertEqual(batch["article_attention_mask"].shape,
                         (batch_size, 512))
        self.assertEqual(batch["labels"].shape, (batch_size, 128))


if __name__ == '__main__':
    unittest.main()
