import math
import tempfile
import unittest
from types import SimpleNamespace

import torch

from ireal_transformer import (
    IRealTransformer,
    TrainingConfig,
    load_ireal_transformer,
    save_ireal_transformer,
    train_ireal_transformer,
)
from tokenizer import IRealTokenizer


def make_song():
    return SimpleNamespace(
        key='C',
        measures_as_beats=[
            ['C7', ' ', ' ', ' '],
            ['F7', ' ', ' ', ' '],
            ['G7', ' ', ' ', ' '],
            ['C7', ' ', ' ', ' '],
        ],
        measure_sections=['A', 'A', 'B', 'A'],
    )


class IRealTransformerTests(unittest.TestCase):
    def test_forward_uses_tokenizer_features_and_predicts_roots(self):
        song = make_song()
        tokens = IRealTokenizer().tokenize_song(song)
        model = IRealTransformer(
            hidden_dim=16,
            attention_heads=4,
            layers=1,
            max_sequence_length=8,
            pitch_class_sets=sorted({token.pitch_class_set for token in tokens}),
        )

        logits = model([tokens])
        hidden, valid_mask = model.encode([tokens])
        song_embedding = model.song_embedding([tokens * 3])

        self.assertEqual(logits['root'].shape, (1, 4, 13))
        self.assertEqual(
            logits['pitch_class_set'].shape,
            (1, 4, len(model.pitch_class_sets)),
        )
        self.assertEqual(logits['duration'].shape, (1, 4, 6))
        self.assertTrue(all(torch.isfinite(output).all() for output in logits.values()))
        self.assertEqual(hidden.shape, (1, 4, 16))
        self.assertTrue(valid_mask.all())
        self.assertEqual(song_embedding.shape, (1, 16))
        self.assertIn(
            model.decode_pitch_class_sets([logits['pitch_class_set'][0, 0].argmax().item()])[0],
            model.pitch_class_sets,
        )

    def test_training_runs_one_epoch(self):
        config = TrainingConfig(
            hidden_dim=16,
            attention_heads=4,
            layers=1,
            sequence_length=8,
            batch_size=2,
            epochs=1,
            max_bars=16,
        )

        model, losses = train_ireal_transformer([make_song()], config, device='cpu')

        self.assertEqual(len(losses), 1)
        self.assertTrue(math.isfinite(losses[0]))
        self.assertEqual(model.output['root'].out_features, 13)
        self.assertEqual(
            set(model.output.keys()),
            {'root', 'pitch_class_set', 'bass_interval', 'duration'},
        )
        self.assertEqual(len(model.pitch_class_sets), 1)

    def test_saves_and_loads_model_with_configuration(self):
        song = make_song()
        config = TrainingConfig(
            hidden_dim=16,
            attention_heads=4,
            layers=1,
            sequence_length=8,
            batch_size=2,
            epochs=1,
            max_bars=16,
        )
        model, _ = train_ireal_transformer([song], config, device='cpu')
        model.eval()
        expected = model([model.tokenize_song(song)])['root']

        with tempfile.TemporaryDirectory() as directory:
            path = f'{directory}/model.pth'
            save_ireal_transformer(model, path)
            loaded = load_ireal_transformer(path, device='cpu')
            actual = loaded([loaded.tokenize_song(song)])['root']
            self.assertTrue(torch.equal(actual, expected))
            self.assertFalse(loaded.training)


if __name__ == '__main__':
    unittest.main()
