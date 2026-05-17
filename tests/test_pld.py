"""Unit tests for PLD ngram-matching logic.

These tests only exercise the pure-Python helpers — they do NOT load an
MLX model, so they run fast and don't need mlx-lm at the import path
(beyond what's already imported by daydream.pld at module load).
"""
from __future__ import annotations

import unittest

from daydream.pld import PLDConfig, _find_ngram_match


class PromptLookupTests(unittest.TestCase):
    def test_finds_match_in_prompt(self) -> None:
        haystack = [10, 11, 12, 13, 14, 11, 12]
        needle = [11, 12]
        # First match at position 1, so pos AFTER the match is 3.
        self.assertEqual(_find_ngram_match(haystack, needle, last_emitted_index=4), 3)

    def test_returns_none_when_no_match(self) -> None:
        haystack = [1, 2, 3, 4, 5]
        needle = [9, 9]
        self.assertIsNone(_find_ngram_match(haystack, needle, last_emitted_index=4))

    def test_empty_needle(self) -> None:
        self.assertIsNone(_find_ngram_match([1, 2, 3], [], last_emitted_index=2))

    def test_haystack_too_short(self) -> None:
        # haystack must be longer than needle for there to be a "match"
        # not overlapping with the trailing occurrence.
        self.assertIsNone(_find_ngram_match([1, 2], [1, 2], last_emitted_index=1))

    def test_returns_first_match(self) -> None:
        # When the ngram appears multiple times, take the earliest.
        haystack = [5, 7, 8, 5, 7, 8, 5, 7]
        needle = [5, 7]
        self.assertEqual(_find_ngram_match(haystack, needle, last_emitted_index=7), 2)

    def test_config_defaults(self) -> None:
        cfg = PLDConfig()
        self.assertEqual(cfg.ngram_size, 3)
        self.assertEqual(cfg.min_ngram_size, 2)
        self.assertEqual(cfg.max_speculative, 5)
        self.assertLessEqual(cfg.min_ngram_size, cfg.ngram_size)


if __name__ == "__main__":
    unittest.main()
