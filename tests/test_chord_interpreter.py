import unittest

from chord_interpreter import ChordInterpreter, NO_CHORD_ROOT


class ChordInterpreterTests(unittest.TestCase):
    def setUp(self):
        self.interpreter = ChordInterpreter()

    def test_major_seventh_pitch_classes(self):
        chord = self.interpreter.interpret('C^7')

        self.assertEqual(chord.root, 0)
        self.assertEqual(chord.pitch_class_set, (1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1))

    def test_key_relative_root_and_bass_interval(self):
        chord = self.interpreter.interpret('D7/F#', tonic=2)

        self.assertEqual(chord.root, 0)
        self.assertEqual(chord.pitch_class_set, (1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0))
        self.assertEqual(chord.bass_interval, 4)

    def test_no_chord(self):
        chord = self.interpreter.interpret('N.C.')

        self.assertEqual(chord.root, NO_CHORD_ROOT)
        self.assertEqual(chord.pitch_class_set, (0,) * 12)


if __name__ == '__main__':
    unittest.main()
