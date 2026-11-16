import re
from dataclasses import dataclass


PITCH_CLASSES = {
    'C': 0,
    'C#': 1,
    'Db': 1,
    'B#': 0,
    'Cb': 11,
    'D': 2,
    'D#': 3,
    'Eb': 3,
    'E': 4,
    'E#': 5,
    'Fb': 4,
    'F': 5,
    'F#': 6,
    'Gb': 6,
    'G': 7,
    'G#': 8,
    'Ab': 8,
    'A': 9,
    'A#': 10,
    'Bb': 10,
    'B': 11,
}

NO_CHORD_ROOT = 12


@dataclass(frozen=True)
class InterpretedChord:
    root: int
    pitch_class_set: tuple[int, ...]
    bass_interval: int
    absolute_root: int | None
    intervals: frozenset[int]


class ChordInterpreter:
    """Convert iReal chord symbols into key-relative pitch features."""

    _chord_pattern = re.compile(
        r'^(?P<root>[A-G](?:#|b)?|W)(?P<quality>.*?)(?:/(?P<bass>[A-G](?:#|b)?))?$',
        re.IGNORECASE,
    )
    _alteration_pattern = re.compile(r'([b#])(5|9|11|13)')

    @staticmethod
    def pitch_class(note: str) -> int:
        note = note[0].upper() + note[1:]
        try:
            return PITCH_CLASSES[note]
        except KeyError as error:
            raise ValueError(f'Unsupported note name: {note}') from error

    @classmethod
    def key_pitch_class(cls, key: str) -> int:
        match = re.match(r'^([A-G](?:#|b)?)-?$', key.strip(), re.IGNORECASE)
        if match is None:
            raise ValueError(f'Unsupported key: {key}')
        return cls.pitch_class(match.group(1))

    @classmethod
    def _chord_intervals(cls, quality: str) -> set[int]:
        quality_lower = quality.lower()
        is_minor_major = quality.startswith('-^')
        is_minor = quality.startswith('-')
        is_major = '^' in quality and not is_minor_major
        is_half_diminished = quality.startswith('h')
        is_diminished = quality.startswith('o')
        is_augmented = quality.startswith('+')
        is_suspended = 'sus' in quality_lower
        is_power = quality.startswith('5')

        if is_half_diminished:
            intervals = {0, 3, 6, 10}
        elif is_diminished:
            intervals = {0, 3, 6}
        elif is_augmented:
            intervals = {0, 4, 8}
        elif is_suspended:
            intervals = {0, 2 if 'sus2' in quality_lower else 5, 7}
        elif quality.startswith('2'):
            intervals = {0, 2, 7}
        elif is_power:
            intervals = {0, 7}
        elif is_minor:
            intervals = {0, 3, 7}
        else:
            intervals = {0, 4, 7}

        has_seventh = '7' in quality or is_major or is_minor_major
        if has_seventh:
            if is_minor_major or is_major:
                intervals.add(11)
            elif is_half_diminished or is_minor:
                intervals.add(10)
            elif is_diminished:
                intervals.add(9)
            else:
                intervals.add(10)

        if '6' in quality:
            intervals.add(9)
        if '9' in quality:
            intervals.add(2)
        if '11' in quality:
            intervals.add(5)
        if '13' in quality:
            intervals.add(9)
        if '11' in quality or '13' in quality:
            intervals.add(2)
        if 'add2' in quality_lower or 'add9' in quality_lower:
            intervals.add(2)
        if 'add4' in quality_lower or 'add11' in quality_lower:
            intervals.add(5)

        alterations = list(cls._alteration_pattern.finditer(quality))
        for alteration in alterations:
            accidental, extension = alteration.groups()
            natural_interval = {'5': 7, '9': 2, '11': 5, '13': 9}[extension]
            intervals.discard(natural_interval)
            intervals.add((natural_interval + (1 if accidental == '#' else -1)) % 12)

        if 'alt' in quality_lower:
            intervals.update({1, 3, 6, 8})
        return intervals

    @classmethod
    def interpret(
        cls,
        chord: str,
        tonic: int = 0,
        previous_root: int | None = None,
        previous_intervals: frozenset[int] | None = None,
    ) -> InterpretedChord:
        chord = chord.strip()
        if chord.upper() in {'N', 'NN', 'N.C.', 'NC', 'N.C'}:
            return InterpretedChord(
                NO_CHORD_ROOT, (0,) * 12, 0, None, frozenset()
            )

        match = cls._chord_pattern.fullmatch(chord)
        if match is None:
            raise ValueError(f'Unsupported chord symbol: {chord}')

        root_name = match.group('root')
        bass_name = match.group('bass')
        if root_name.upper() == 'W':
            if previous_root is None or previous_intervals is None:
                raise ValueError(f'Cannot resolve invisible-root chord without a previous chord: {chord}')
            root = previous_root
            intervals = set(previous_intervals)
        else:
            root = cls.pitch_class(root_name)
            intervals = cls._chord_intervals(match.group('quality'))

        bass = cls.pitch_class(bass_name) if bass_name else root
        pitch_set = tuple(int(interval in intervals) for interval in range(12))
        return InterpretedChord(
            root=(root - tonic) % 12,
            pitch_class_set=pitch_set,
            bass_interval=(bass - root) % 12,
            absolute_root=root,
            intervals=frozenset(intervals),
        )
