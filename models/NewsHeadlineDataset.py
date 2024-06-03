import pandas as pd
from transformers import T5Tokenizer
from torch.utils.data import Dataset


class NewsHeadlineDataset(Dataset):

    def __init__(self, data: pd.DataFrame, tokenizer: T5Tokenizer, article_max_token_len: int = 512, headline_max_token_len: int = 128):

        self.tokenizer = tokenizer
        self.data = data
        self.article_max_token_len = article_max_token_len
        self.headline_max_token_len = headline_max_token_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        data_row = self.data.iloc[index]

        article = data_row["Article"]

        article_encoding = self.tokenizer(article, max_length=self.article_max_token_len, padding="max_length", truncation=True, return_attention_mask=True, add_special_tokens=True, return_tensors="pt")

        headline_encoding = self.tokenizer(data_row["Headline"], max_length=self.headline_max_token_len, padding="max_length", truncation=True, return_attention_mask=True, add_special_tokens=True, return_tensors="pt")

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
