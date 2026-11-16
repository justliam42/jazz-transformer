import unittest
from types import SimpleNamespace

from tokenizer import IRealTokenizer


class IRealTokenizerTests(unittest.TestCase):
    def test_tokens_include_duration_position_and_section(self):
        song = SimpleNamespace(
            key='C',
            measures_as_beats=[
                ['C7', ' ', 'G7', ' '],
                ['F7', ' ', ' ', ' '],
            ],
            measure_sections=['A', 'B'],
        )

        tokens = IRealTokenizer().tokenize_song(song)

        self.assertEqual(
            [
                (token.root, token.duration, token.beat_position, token.bar_position, token.section)
                for token in tokens
            ],
            [
                (0, 2, 1, 0, 'A'),
                (7, 2, 3, 0, 'A'),
                (5, 4, 1, 1, 'B'),
            ],
        )

    def test_no_chord_is_a_token_but_empty_beats_are_not(self):
        song = SimpleNamespace(
            key='C',
            measures_as_beats=[['C7', ' ', 'N.C.', ' ']],
            measure_sections=[None],
        )

        tokens = IRealTokenizer().tokenize_song(song)

        self.assertEqual(len(tokens), 2)
        self.assertEqual(tokens[0].duration, 2)
        self.assertEqual(tokens[1].root, 12)
        self.assertEqual(tokens[1].beat_position, 3)
        self.assertIsNone(tokens[1].section)


if __name__ == '__main__':
    unittest.main()
