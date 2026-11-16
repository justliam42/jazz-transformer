import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ireal_transformer import IRealTransformer
from song_similarity import save_song_embeddings
from tokenizer import IRealTokenizer


def make_song(title, chords):
    return SimpleNamespace(
        title=title,
        composer='Test Composer',
        key='C',
        measures_as_beats=[[chord, ' ', ' ', ' '] for chord in chords],
        measure_sections=['A'] * len(chords),
    )


class SongSimilarityTests(unittest.TestCase):
    def test_exports_chart_metadata_and_embeddings(self):
        songs = [
            make_song('Tune One', ['C7', 'F7']),
            make_song('Tune Two', ['G7', 'C7']),
        ]
        tokenizer = IRealTokenizer()
        tokenized = tokenizer.tokenize_songs(songs)
        pitch_sets = sorted({
            token.pitch_class_set
            for tokens in tokenized
            for token in tokens
        })
        model = IRealTransformer(
            hidden_dim=16,
            attention_heads=4,
            layers=1,
            max_sequence_length=8,
            pitch_class_sets=pitch_sets,
        )
        model.train()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'song_embeddings.json'
            save_song_embeddings(model, songs, path, batch_size=1)
            data = json.loads(path.read_text(encoding='utf-8'))

        self.assertEqual(data['distance'], 'cosine')
        self.assertEqual([song['title'] for song in data['songs']], ['Tune One', 'Tune Two'])
        self.assertEqual(len(data['songs'][0]['embedding']), 16)
        self.assertTrue(all(isinstance(value, float) for value in data['songs'][0]['embedding']))
        self.assertTrue(model.training)


if __name__ == '__main__':
    unittest.main()
