# -*- coding: utf-8 -*-

import unittest


class TestAlterations(unittest.TestCase):

    def setUp(self):
        pass

    def test_split_n_encode(self):
        """
        Test the Rust implementation of the split and one-hot encode functionality
        """
        from lumberjack import alterations
        raw_texts = ['hello, there', 'hi, there']
        unique_words, one_hot = alterations.split_n_one_hot_encode(raw_texts, sep=',', cutoff=0)

        self.assertEqual(
            'hello' in unique_words, True,
            msg='Expected to find "hello" in unique words, but it was not! Unique words: {}'.format(unique_words)
        )

        self.assertEqual(
            sum(one_hot[0]), 2,
            msg='Expected to find the sum of one_hot index 0 to be 2, but it was not, sum was {}'.format(sum(one_hot))
        )


if __name__ == '__main__':
    unittest.main()
