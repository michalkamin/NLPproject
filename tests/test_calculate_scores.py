import unittest
import pandas as pd
from transformers import (T5Tokenizer,
                          T5ForConditionalGeneration,
                          BartTokenizer,
                          BartForConditionalGeneration)
from evaluation.evaluate import calculate_scores_df, calculate_scores_df_tuned


class TestCalculateScores(unittest.TestCase):
    """
    Unit tests for the calculate_scores_df
    and calculate_scores_df_tuned functions.
    """
    tokenizer_t5: T5Tokenizer
    model_t5: T5ForConditionalGeneration
    trained_model_t5: T5ForConditionalGeneration
    tokenizer_bart: BartTokenizer
    model_bart: BartForConditionalGeneration
    trained_model_bart: BartForConditionalGeneration
    df: pd.DataFrame

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up the tokenizer and model for T5 and BART before any tests are run.
        """
        cls.tokenizer_t5 = T5Tokenizer.from_pretrained('t5-small')
        cls.model_t5 = T5ForConditionalGeneration.from_pretrained('t5-small')
        cls.trained_model_t5 = T5ForConditionalGeneration.from_pretrained('t5-small')

        cls.tokenizer_bart = BartTokenizer.from_pretrained('facebook/bart-base')
        cls.model_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
        cls.trained_model_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

        cls.df = pd.DataFrame({
            'Article': ["The quick brown fox jumps over the lazy dog."],
            'Headline': ["Quick brown fox jumps"],
            'Summary': ["Fox jumps over dog"]
        })

    def test_calculate_scores_df(self) -> None:
        """
        Test the calculate_scores_df function.
        """
        results_df = calculate_scores_df(self.df, self.tokenizer_t5,
                                         self.model_t5,
                                         self.trained_model_t5,
                                         self.tokenizer_bart,
                                         self.model_bart,
                                         self.trained_model_bart,
                                         include_summary=False,
                                         verbose=False)
        self.assertEqual(len(results_df), 1)
        self.assertIn('bleu_finetuned_t5', results_df.columns)
        self.assertIn('bleu_pretrained_t5', results_df.columns)

    def test_calculate_scores_df_tuned(self) -> None:
        """
        Test the calculate_scores_df_tuned function.
        """
        results_df = calculate_scores_df_tuned(self.df,
                                               self.tokenizer_t5,
                                               self.trained_model_t5,
                                               self.tokenizer_bart,
                                               self.trained_model_bart,
                                               include_summary=False,
                                               verbose=False)
        self.assertEqual(len(results_df), 1)
        self.assertIn('bleu_finetuned_t5', results_df.columns)
        self.assertIn('rouge_finetuned_t5', results_df.columns)


if __name__ == '__main__':
    unittest.main()
