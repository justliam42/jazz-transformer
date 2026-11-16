from dataclasses import dataclass
from typing import TYPE_CHECKING

from chord_interpreter import ChordInterpreter

if TYPE_CHECKING:
    from parser import Tune


DURATION_CLASSES = (1, 2, 3, 4, 6, 8)
ROOT_CATEGORIES = (
    'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'N.C.'
)


@dataclass(frozen=True)
class Token:
    """Categorical chord features and onset location for one chord event.

    Root values index ROOT_CATEGORIES; pitch-class bits are semitone intervals
    above that root. Beat positions are one-based and bar positions are zero-based.
    """

    root: int
    pitch_class_set: tuple[int, ...]
    bass_interval: int
    duration: int
    beat_position: int
    bar_position: int
    section: str | None


class IRealTokenizer:
    """Turn a parsed iReal Tune into key-normalized chord-event tokens."""

    def __init__(self, duration_classes: tuple[int, ...] = DURATION_CLASSES):
        if not duration_classes or any(duration <= 0 for duration in duration_classes):
            raise ValueError('duration_classes must contain positive beat lengths')
        self.duration_classes = tuple(sorted(set(duration_classes)))
        self.chord_interpreter = ChordInterpreter()

    def _duration_class(self, beats: int) -> int:
        return next(
            (duration for duration in self.duration_classes if duration >= beats),
            self.duration_classes[-1],
        )

    def tokenize_song(self, song: 'Tune') -> list[Token]:
        if len(song.measures_as_beats) != len(song.measure_sections):
            raise ValueError('Tune measure beats and section labels are not aligned')

        tonic = self.chord_interpreter.key_pitch_class(song.key)
        events: list[tuple[int, int, str, str | None]] = []
        measure_starts: list[int] = []
        beat_offset = 0

        for bar_position, (beats, section) in enumerate(
            zip(song.measures_as_beats, song.measure_sections)
        ):
            measure_starts.append(beat_offset)
            for beat_index, beat_chords in enumerate(beats):
                if not beat_chords.strip():
                    continue
                for chord in beat_chords.split(','):
                    chord = chord.strip()
                    if chord:
                        events.append((beat_offset + beat_index, bar_position, chord, section))
            beat_offset += len(beats)

        tokens: list[Token] = []
        previous_root: int | None = None
        previous_intervals: frozenset[int] | None = None
        event_index = 0
        while event_index < len(events):
            position, bar_position, chord, section = events[event_index]
            interpreted = self.chord_interpreter.interpret(
                chord,
                tonic=tonic,
                previous_root=previous_root,
                previous_intervals=previous_intervals,
            )
            next_index = event_index + 1
            while next_index < len(events) and events[next_index][0] == position:
                next_index += 1
            next_position = events[next_index][0] if next_index < len(events) else beat_offset
            duration = self._duration_class(max(1, next_position - position))

            tokens.append(Token(
                root=interpreted.root,
                pitch_class_set=interpreted.pitch_class_set,
                bass_interval=interpreted.bass_interval,
                duration=duration,
                beat_position=position - measure_starts[bar_position] + 1,
                bar_position=bar_position,
                section=section,
            ))
            previous_root = interpreted.absolute_root
            previous_intervals = interpreted.intervals
            event_index += 1

        return tokens

    def tokenize_songs(self, songs: list['Tune']) -> list[list[Token]]:
        return [self.tokenize_song(song) for song in songs]
