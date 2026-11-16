from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import torch
from torch import nn

from tokenizer import IRealTokenizer, Token
from tqdm import tqdm

if TYPE_CHECKING:
    from parser import Tune


SECTION_IDS = {
    None: 0,
    **{chr(letter): letter - ord('A') + 1 for letter in range(ord('A'), ord('Z') + 1)},
}


@dataclass(frozen=True)
class TrainingConfig:
    """Small baseline settings for next-chord training on iReal charts."""

    hidden_dim: int = 128
    attention_heads: int = 4
    layers: int = 2
    dropout: float = 0.1
    sequence_length: int = 128
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_bars: int = 1024
    max_beats_per_bar: int = 32


class IRealTransformer(nn.Module):
    """Causal chord-event model with accessible sequence and song embeddings."""

    def __init__(
        self,
        tokenizer: IRealTokenizer | None = None,
        hidden_dim: int = 128,
        attention_heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
        max_sequence_length: int = 128,
        max_bars: int = 1024,
        max_beats_per_bar: int = 32,
        pitch_class_sets: Sequence[tuple[int, ...]] | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim % attention_heads:
            raise ValueError('hidden_dim must be divisible by attention_heads')
        if max_sequence_length <= 0 or max_bars <= 0 or max_beats_per_bar <= 0:
            raise ValueError('sequence, bar, and beat limits must be positive')

        self.tokenizer = tokenizer or IRealTokenizer()
        self.duration_to_id = {
            duration: index
            for index, duration in enumerate(self.tokenizer.duration_classes)
        }
        self.max_sequence_length = max_sequence_length
        self.max_bars = max_bars
        self.max_beats_per_bar = max_beats_per_bar
        self.hidden_dim = hidden_dim
        self.attention_heads = attention_heads
        self.layers = layers
        self.dropout = dropout
        if pitch_class_sets is None:
            raise ValueError('pitch_class_sets must be provided from the training corpus')
        self.pitch_class_sets = tuple(dict.fromkeys(tuple(values) for values in pitch_class_sets))
        if not self.pitch_class_sets or any(len(values) != 12 for values in self.pitch_class_sets):
            raise ValueError('pitch_class_sets must contain non-empty 12-bit pitch sets')
        self.pitch_set_to_id = {
            pitch_set: index for index, pitch_set in enumerate(self.pitch_class_sets)
        }
        self.root_embedding = nn.Embedding(13, hidden_dim)
        self.bass_embedding = nn.Embedding(12, hidden_dim)
        self.duration_embedding = nn.Embedding(len(self.duration_to_id), hidden_dim)
        self.beat_embedding = nn.Embedding(max_beats_per_bar, hidden_dim)
        self.bar_embedding = nn.Embedding(max_bars, hidden_dim)
        self.section_embedding = nn.Embedding(27, hidden_dim)
        self.position_embedding = nn.Embedding(max_sequence_length, hidden_dim)
        self.pitch_set_projection = nn.Linear(12, hidden_dim, bias=False)
        self.input_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.output = nn.ModuleDict({
            'root': nn.Linear(hidden_dim, 13),
            'pitch_class_set': nn.Linear(hidden_dim, len(self.pitch_class_sets)),
            'bass_interval': nn.Linear(hidden_dim, 12),
            'duration': nn.Linear(hidden_dim, len(self.duration_to_id)),
        })

    def _embed_tokens(
        self, token_batch: Sequence[Sequence[Token]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not token_batch:
            raise ValueError('token_batch must contain at least one sequence')
        lengths = [len(sequence) for sequence in token_batch]
        if any(length == 0 for length in lengths):
            raise ValueError('token sequences must not be empty')
        sequence_length = max(lengths)
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f'sequence length {sequence_length} exceeds configured maximum '
                f'{self.max_sequence_length}'
            )

        batch_size = len(token_batch)
        device = next(self.parameters()).device
        roots = torch.zeros((batch_size, sequence_length), dtype=torch.long, device=device)
        basses = torch.zeros_like(roots)
        durations = torch.zeros_like(roots)
        beats = torch.zeros_like(roots)
        bars = torch.zeros_like(roots)
        sections = torch.zeros_like(roots)
        pitch_sets = torch.zeros(
            (batch_size, sequence_length, 12), dtype=torch.float32, device=device
        )
        padding_mask = torch.ones(
            (batch_size, sequence_length), dtype=torch.bool, device=device
        )

        for batch_index, sequence in enumerate(token_batch):
            for token_index, token in enumerate(sequence):
                if token.duration not in self.duration_to_id:
                    raise ValueError(f'Unsupported token duration: {token.duration}')
                if not 1 <= token.beat_position <= self.max_beats_per_bar:
                    raise ValueError(f'Beat position out of range: {token.beat_position}')
                if not 0 <= token.bar_position < self.max_bars:
                    raise ValueError(f'Bar position out of range: {token.bar_position}')
                try:
                    section_id = SECTION_IDS[token.section.upper() if token.section else None]
                except KeyError as error:
                    raise ValueError(
                        f'Unsupported section label: {token.section!r}; expected a single letter'
                    ) from error
                if len(token.pitch_class_set) != 12:
                    raise ValueError('pitch_class_set must contain 12 values')
                if token.root not in range(13) or token.bass_interval not in range(12):
                    raise ValueError('token root or bass interval is out of range')

                roots[batch_index, token_index] = token.root
                basses[batch_index, token_index] = token.bass_interval
                durations[batch_index, token_index] = self.duration_to_id[token.duration]
                beats[batch_index, token_index] = token.beat_position - 1
                bars[batch_index, token_index] = token.bar_position
                sections[batch_index, token_index] = section_id
                pitch_sets[batch_index, token_index] = torch.tensor(
                    token.pitch_class_set, dtype=torch.float32, device=device
                )
                padding_mask[batch_index, token_index] = False

        positions = torch.arange(sequence_length, device=device).expand(batch_size, -1)
        embeddings = (
            self.root_embedding(roots)
            + self.bass_embedding(basses)
            + self.duration_embedding(durations)
            + self.beat_embedding(beats)
            + self.bar_embedding(bars)
            + self.section_embedding(sections)
            + self.position_embedding(positions)
            + self.pitch_set_projection(pitch_sets)
        )
        return embeddings, padding_mask

    def encode(
        self,
        token_batch: Sequence[Sequence[Token]],
        *,
        causal: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return hidden states and a valid-token mask for each token sequence."""
        embeddings, padding_mask = self._embed_tokens(token_batch)
        sequence_length = embeddings.shape[1]
        causal_mask = None
        if causal:
            causal_mask = torch.triu(
                torch.ones(
                    (sequence_length, sequence_length),
                    dtype=torch.bool,
                    device=embeddings.device,
                ),
                diagonal=1,
            )
        hidden = self.transformer(
            self.input_norm(embeddings),
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        return hidden, ~padding_mask

    def song_embedding(
        self, token_batch: Sequence[Sequence[Token]], *, causal: bool = False
    ) -> torch.Tensor:
        """Mean-pool token states, chunking songs longer than the model context."""
        if not token_batch:
            raise ValueError('token_batch must contain at least one sequence')
        chunks: list[Sequence[Token]] = []
        owners: list[int] = []
        for song_index, tokens in enumerate(token_batch):
            if not tokens:
                raise ValueError('token sequences must not be empty')
            for start in range(0, len(tokens), self.max_sequence_length):
                chunks.append(tokens[start:start + self.max_sequence_length])
                owners.append(song_index)

        hidden, valid_mask = self.encode(chunks, causal=causal)
        weights = valid_mask.unsqueeze(-1).to(hidden.dtype)
        chunk_sums = (hidden * weights).sum(dim=1)
        chunk_lengths = weights.sum(dim=1)
        song_sums = hidden.new_zeros((len(token_batch), hidden.shape[-1]))
        song_lengths = hidden.new_zeros((len(token_batch), 1))
        for chunk_index, song_index in enumerate(owners):
            song_sums[song_index] += chunk_sums[chunk_index]
            song_lengths[song_index] += chunk_lengths[chunk_index]
        return song_sums / song_lengths.clamp_min(1)

    def decode_pitch_class_sets(self, class_ids: Sequence[int]) -> list[tuple[int, ...]]:
        """Map predicted pitch-set class IDs back to valid 12-bit chord sets."""
        if any(
            class_id < 0 or class_id >= len(self.pitch_class_sets)
            for class_id in class_ids
        ):
            raise ValueError('pitch-set class ID is out of range')
        return [self.pitch_class_sets[class_id] for class_id in class_ids]

    def forward(
        self, token_batch: Sequence[Sequence[Token]]
    ) -> dict[str, torch.Tensor]:
        hidden, _ = self.encode(token_batch, causal=True)
        return {name: head(hidden) for name, head in self.output.items()}

    def tokenize_song(self, song: Tune) -> list[Token]:
        """Tokenize a parsed Tune using the model's tokenizer."""
        return self.tokenizer.tokenize_song(song)

    def get_song_embedding(self, song: Tune, *, causal: bool = False) -> torch.Tensor:
        """Return a single embedding vector for a parsed Tune."""
        with torch.no_grad():
            tokens = self.tokenize_song(song)
            return self.song_embedding([tokens], causal=causal)[0]

    def model_config(self) -> dict[str, object]:
        """Return the constructor settings needed to recreate this model."""
        return {
            'hidden_dim': self.hidden_dim,
            'attention_heads': self.attention_heads,
            'layers': self.layers,
            'dropout': self.dropout,
            'max_sequence_length': self.max_sequence_length,
            'max_bars': self.max_bars,
            'max_beats_per_bar': self.max_beats_per_bar,
            'pitch_class_sets': self.pitch_class_sets,
        }


def train_ireal_transformer(
    songs: Sequence[Tune],
    config: TrainingConfig = TrainingConfig(),
    device: str | torch.device | None = None,
) -> tuple[IRealTransformer, list[float]]:
    """Train next-event chord features and return the model and epoch losses."""
    if config.epochs <= 0 or config.batch_size <= 0 or config.sequence_length <= 0:
        raise ValueError('epochs, batch_size, and sequence_length must be positive')

    tokenizer = IRealTokenizer()
    tokenized_songs = tokenizer.tokenize_songs(list(songs))
    pitch_class_sets = tuple(sorted({
        token.pitch_class_set
        for tokens in tokenized_songs
        for token in tokens
    }))
    print("Tokenizing complete. Found", len(pitch_class_sets), "unique pitch class sets.")
    examples: list[tuple[list[Token], list[Token]]] = []
    for tokens in tokenized_songs:
        for start in range(0, len(tokens) - 1, config.sequence_length):
            end = min(start + config.sequence_length, len(tokens) - 1)
            inputs = tokens[start:end]
            targets = tokens[start + 1:end + 1]
            if inputs and targets:
                examples.append((inputs, targets))
    if not examples:
        raise ValueError('training requires at least one song with two chord events')

    model = IRealTransformer(
        tokenizer=tokenizer,
        hidden_dim=config.hidden_dim,
        attention_heads=config.attention_heads,
        layers=config.layers,
        dropout=config.dropout,
        max_sequence_length=config.sequence_length,
        max_bars=config.max_bars,
        max_beats_per_bar=config.max_beats_per_bar,
        pitch_class_sets=pitch_class_sets,
    )
    selected_device = torch.device(
        device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    model.to(selected_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    losses: list[float] = []

    print("Training on", len(examples), "examples for", config.epochs, "epochs...")
    for _ in tqdm(range(config.epochs)):
        random.shuffle(examples)
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        for batch_start in range(0, len(examples), config.batch_size):
            batch = examples[batch_start:batch_start + config.batch_size]
            input_sequences = [example[0] for example in batch]
            max_length = max(len(targets) for _, targets in batch)
            targets = {
                name: torch.full(
                    (len(batch), max_length), -100, dtype=torch.long, device=selected_device
                )
                for name in ('root', 'pitch_class_set', 'bass_interval', 'duration')
            }
            for batch_index, (_, target_tokens) in enumerate(batch):
                for token_index, token in enumerate(target_tokens):
                    if token.duration not in tokenizer.duration_classes:
                        raise ValueError(f'Unsupported token duration: {token.duration}')

                    targets['root'][batch_index, token_index] = token.root
                    targets['pitch_class_set'][batch_index, token_index] = (
                        model.pitch_set_to_id[token.pitch_class_set]
                    )
                    targets['bass_interval'][batch_index, token_index] = token.bass_interval
                    targets['duration'][batch_index, token_index] = (
                        tokenizer.duration_classes.index(token.duration)
                    )

            optimizer.zero_grad(set_to_none=True)
            logits_by_field = model(input_sequences)
            field_losses = [
                nn.functional.cross_entropy(
                    logits_by_field[name].reshape(-1, logits_by_field[name].shape[-1]),
                    target.reshape(-1),
                    ignore_index=-100,
                )
                for name, target in targets.items()
            ]
            valid_mask = targets['root'] != -100
            loss = (
                field_losses[0]
                + field_losses[1]
                + field_losses[2]
                + 0.5 * field_losses[3]
            ) / 3.5
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            valid_tokens = int(valid_mask.sum().item())
            epoch_loss += loss.item() * valid_tokens
            epoch_tokens += valid_tokens
        losses.append(epoch_loss / epoch_tokens)

    return model, losses

def save_ireal_transformer(model: IRealTransformer, model_path: str | Path) -> None:
    """Save model weights and the settings needed to recreate the model."""
    torch.save(
        {'config': model.model_config(), 'state_dict': model.state_dict()},
        model_path,
    )


def load_ireal_transformer(
    model_path: str | Path,
    device: str | torch.device | None = None,
) -> IRealTransformer:
    """Load a model saved with :func:`save_ireal_transformer`."""
    selected_device = torch.device(
        device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    checkpoint = torch.load(model_path, map_location=selected_device, weights_only=True)
    model = IRealTransformer(**checkpoint['config'])
    model.load_state_dict(checkpoint['state_dict'])
    model.to(selected_device)
    model.eval()
    return model