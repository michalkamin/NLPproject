import pandas as pd
from transformers import T5Tokenizer
from torch.utils.data import Dataset


class NewsHeadlineDataset(Dataset):
    """
    A custom dataset class for handling news articles and their corresponding headlines,
    designed to work with the T5 tokenizer and PyTorch's Dataset class.

    Args:
        tokenizer (T5Tokenizer): The tokenizer used to preprocess the articles and headlines.
        data (pd.DataFrame): The dataframe containing the news articles and their corresponding headlines.
        article_max_token_len (int): The maximum length of tokens for the article input.
        headline_max_token_len (int): The maximum length of tokens for the headline output.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 tokenizer: T5Tokenizer,
                 article_max_token_len: int = 512,
                 headline_max_token_len: int = 128):
        """
        Initializes the NewsHeadlineDataset with the provided data and tokenizer.

        Args:
        -----------
            data (pd.DataFrame): The dataframe containing the news articles and their corresponding headlines.
            tokenizer (T5Tokenizer): The tokenizer used to preprocess the articles and headlines.
            article_max_token_len (int, optional): The maximum length of tokens for the article input (default is 512).
            headline_max_token_len (int, optional): The maximum length of tokens for the headline output (default is 128).
        """
        self.tokenizer = tokenizer
        self.data = data
        self.article_max_token_len = article_max_token_len
        self.headline_max_token_len = headline_max_token_len

    def __len__(self) -> int:
        """
        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        """
        Returns a dictionary containing the tokenized article and headline for a given index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            dict
                A dictionary containing the tokenized article and headline with the following keys:
                - article: The original article text.
                - headline: The original headline text.
                - article_input_ids: The token IDs of the article.
                - article_attention_mask: The attention mask for the article.
                - labels: The token IDs of the headline with padding tokens replaced by -100.
                - labels_attention_mask: The attention mask for the headline.
        """
        data_row = self.data.iloc[index]

        article = data_row["Article"]

        article_encoding = self.tokenizer(
            article,
            max_length=self.article_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt")

        headline_encoding = self.tokenizer(
            data_row["Headline"],
            max_length=self.headline_max_token_len,
            padding="max_length", truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt")

        labels = headline_encoding["input_ids"]
        labels[labels == 0] = -100

        return dict(
            article=article,
            headline=data_row["Headline"],
            article_input_ids=article_encoding["input_ids"].flatten(),
            article_attention_mask=article_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=headline_encoding["attention_mask"].flatten()
        )
