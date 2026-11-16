from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch

from ireal_transformer import IRealTransformer
from parser import Tune


def save_song_embeddings(
    model: IRealTransformer,
    songs: Sequence[Tune],
    output_path: str | Path = 'song_embeddings.json',
    batch_size: int = 16,
) -> None:
    """Export pooled bidirectional song embeddings and chart labels to JSON."""
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')

    was_training = model.training
    model.eval()
    entries = []
    try:
        with torch.inference_mode():
            for start in range(0, len(songs), batch_size):
                song_batch = songs[start:start + batch_size]
                tokens = [model.tokenize_song(song) for song in song_batch]
                for offset, song_tokens in enumerate(tokens):
                    if not song_tokens:
                        raise ValueError(
                            f'Song at index {start + offset} has no chord events'
                        )
                embeddings = model.song_embedding(tokens, causal=False).cpu().tolist()
                entries.extend(
                    {
                        'title': song.title,
                        'composer': song.composer,
                        'key': song.key,
                        'embedding': embedding,
                    }
                    for song, embedding in zip(song_batch, embeddings)
                )
    finally:
        model.train(was_training)

    payload = {'distance': 'cosine', 'songs': entries}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
