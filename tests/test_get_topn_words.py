import unittest
from evaluation.get_topn_words import get_topn_words


class TestUtils(unittest.TestCase):
    """
    Unit tests for utility functions.
    """

    def test_get_topn_words(self) -> None:
        """
        Test the get_topn_words function.
        """
        article = "The quick brown fox jumps over the lazy dog."
        top_words = get_topn_words(article, n=2)
        self.assertEqual(len(top_words), 2)
        self.assertIsInstance(top_words, list)

    def test_get_topn_words_single_word(self) -> None:
        """
        Test the get_topn_words function with a single word article.
        """
        article = "Fox."
        top_words = get_topn_words(article, n=1)
        self.assertEqual(len(top_words), 1)
        self.assertIsInstance(top_words, list)


if __name__ == '__main__':
    unittest.main()
