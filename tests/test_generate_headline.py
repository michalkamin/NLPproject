import unittest
import torch
from transformers import (T5Tokenizer,
                          T5ForConditionalGeneration,
                          BartTokenizer,
                          BartForConditionalGeneration)
from evaluation.generate_headline import generate_headline


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestGenerateHeadline(unittest.TestCase):
    """
    Unit tests for the generate_headline function.
    """

    tokenizer_t5: T5Tokenizer
    model_t5: T5ForConditionalGeneration
    tokenizer_bart: BartTokenizer
    model_bart: BartForConditionalGeneration

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up the tokenizer and model for T5
        and BART before any tests are run.
        """
        cls.tokenizer_t5 = T5Tokenizer.from_pretrained('t5-small')
        cls.model_t5 = T5ForConditionalGeneration.from_pretrained(
            't5-small'
        ).to(device)

        cls.tokenizer_bart = BartTokenizer.from_pretrained(
            'facebook/bart-base'
        )

        cls.model_bart = BartForConditionalGeneration.from_pretrained(
            'facebook/bart-base'
        ).to(device)

    def test_generate_headline_t5(self) -> None:
        """
        Test headline generation using the T5 model.
        """
        text = "The quick brown fox jumps over the lazy dog."
        headline = generate_headline(text,
                                     self.tokenizer_t5,
                                     self.model_t5,
                                     prompt="summarize: ")
        self.assertIsInstance(headline, str)
        self.assertTrue(len(headline) > 0)

    def test_generate_headline_bart(self) -> None:
        """
        Test headline generation using the BART model.
        """
        text = "The quick brown fox jumps over the lazy dog."
        headline = generate_headline(text,
                                     self.tokenizer_bart,
                                     self.model_bart)
        self.assertIsInstance(headline, str)
        self.assertTrue(len(headline) > 0)


if __name__ == '__main__':
    unittest.main()
