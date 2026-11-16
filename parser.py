import re
import urllib.parse
import itertools

__license__ = 'MIT'
__docformat__ = 'reStructuredText'


class Tune(object):
    """Represents the chords in a song, with functionality to import the iReal format.

    :ivar chord_string: A single string that has all chords of the tune, with bar lines, repeat markers, endings,
       codas etc.
    :ivar measures_as_strings: A list, for which every element corresponds to a single bar, containing the chords in
       string form. Repeats, codas etc. have been flattened.
    :ivar measures_as_beats: A list of measures, each containing one chord/event string per beat. Empty beats contain
       a single space.
    :ivar measure_sections: A list of section labels aligned with measures_as_strings; labels are None before the
       first section marker.
    :ivar title: The title
    :ivar composer: The composer
    :ivar style: The style (e.g. 'Swing', 'Bossa', 'Blues' etc.)
    :ivar key: The key (e.g. 'A', 'F#' etc)
    :ivar transpose: How many semitones to transpose
    :ivar comp_style: Accompaniment style (usually empty)
    :ivar bpm: Tempo in BPM (usually empty)
    :ivar repeats: How many repeats (usually empty)
    :ivar time_signature: Time signature as a tuple (e.g. (3,4), (4, 4), (5, 8) etc.)


    Notice, that some of these meta-data fields might
    be empty, depending on the input url.
    """

    _chords_prefix = """1r34LbKcu7"""

    chord_regex = re.compile(r'(?<!/)([A-GNn][^\sA-GN/]*(?:/[A-GN][#b]?)?)')

    @classmethod
    def _obfusc50(cls, block):
        """Unscrambles blocks of 50 by character substitution

        :param block: A string of 50 characters
        :return: An unscrambled string
        """
        result = list(block)
        # switch first 5 characters with last 5
        for i in range(5):
            result[i] = block[49 - i]
            result[49 - i] = block[i]
        # switch characters 10-24 like before:
        for i in range(10, 24):
            result[i] = block[49 - i]
            result[49 - i] = block[i]
        return ''.join(result)

    @classmethod
    def _unscramble_chord_string(cls, scrambled_string):
        """Unscrambles a single song in iReal's crazy format (Kudos to ironss and pianosnake for
        figuring this out!)
        :param scrambled_string: A scrambled chord string, corresponding to one tune
        :return: An unscrambled chord string.
        """
        unscrambled = ""
        while len(scrambled_string) > 50:
            temp = scrambled_string[:50]
            scrambled_string = scrambled_string[50:]
            if len(scrambled_string) < 2:
                unscrambled += temp
            else:
                unscrambled += cls._obfusc50(temp)
        unscrambled += scrambled_string
        return unscrambled

    @classmethod
    def _cleanup_chord_string(cls, chord_string):
        """Removes excessive whitespace, unnecessary stuff, empty measures etc. and return a nice,
        readable string
        :param chord_string: unscrambled chords in string form
        :return: cleaned up chord string
        """
        # unify symbol for new measure to |
        chord_string = re.sub(r'LZ|K', '|', chord_string)
        # unify symbol for no chord to n
        # chord_string = re.sub(r'[np]', 'n', chord_string)
        # unify symbol for one-bar repeat to x
        chord_string = re.sub(r'cl', 'x', chord_string)
        # remove stars with empty space in between
        chord_string = re.sub(r'\*\s*\*', '', chord_string)
        # remove vertical spacers
        chord_string = re.sub(r'Y+', '', chord_string)
        # remove empty space
        chord_string = re.sub(r'XyQ|,', ' ', chord_string)
        # remove empty measures
        chord_string = re.sub(r'\|\s*\|', '|', chord_string)
        # remove end markers
        chord_string = re.sub(r'Z', '', chord_string)
        # remove spaces behing bar lines
        chord_string = re.sub(r'\|\s+', '|', chord_string)
        # remove multiple white-spaces
        chord_string = re.sub(r'\s+', ' ', chord_string)
        # remove trailing white-space
        chord_string = chord_string.rstrip()
        return chord_string

    @classmethod
    def _remove_annotations(cls, chord_string):
        """Removes comments, section markers, alternative chords, time signatures etc.
        :param chord_string: A chord string
        :return: A cleaned up chord string
        """
        # unify symbol for new measure to |
        chord_string = re.sub(r'[\[\]]', '|', chord_string)
        # remove empty measures
        chord_string = re.sub(r'\|\s*\|', '|', chord_string)
        # remove comments
        chord_string = re.sub(r'<.*?>', '', chord_string)
        # remove alternative chords
        chord_string = re.sub(r'\([^)]*\)', '', chord_string)
        # remove unneeded single f (fermata)
        chord_string = re.sub(r'f', '', chord_string)
        # remove l unless it's part of an alt chord
        chord_string = re.sub(r'(?<!a)l(?!t)', '', chord_string)
        # remove s (for 'small), unless it's part of a sus chord
        chord_string = re.sub(r'(?<!su)s(?!us)', '', chord_string)
        # remove section markers
        chord_string = re.sub(r'\*\w', '', chord_string)
        # remove time signatures
        chord_string = re.sub(r'T\d+', '', chord_string)
        # remove the star that's sometimes after augmented chords
        chord_string = re.sub(r'\+\*', '+', chord_string)
        return chord_string

    @classmethod
    def _remove_markers(cls, chord_string):
        # remove part markers, segnos, codas etc
        return re.sub(r'U|S|Q|N\d', '', chord_string)

    @classmethod
    def _fill_long_repeats(cls, chord_string):
        """Replaces long repeats with multiple endings with the appropriate chords
        :param chord_string: A chord string
        :return: A chord string with filled repeats
        """
        repeat_match = re.search(r'{(.+?)}', chord_string)
        if repeat_match is None:
            return chord_string
        # is there a first ending in the repeat?
        number_matches = list(re.finditer(r'N(\d)', repeat_match.group(1)))
        if len(number_matches) > 1:
            repeated_measures = repeat_match.group(1)[:number_matches[0].start()]
            endings = [
                repeat_match.group(1)[match.end():(
                    number_matches[i + 1].start() if i + 1 < len(number_matches) else len(repeat_match.group(1))
                )]
                for i, match in enumerate(number_matches)
            ]
            expanded_repeat = ''.join(repeated_measures + ending for ending in endings)
            new_chord_string = (
                chord_string[:repeat_match.start()] + '|' + expanded_repeat + '|' +
                chord_string[repeat_match.end():]
            )
            return cls._fill_long_repeats(new_chord_string)
        number_match = number_matches[0] if number_matches else None
        if number_match is not None:
            first_repeat = repeat_match.group(1)
            # first, get rid of the first repeat number and the curly braces
            first_repeat = re.sub(r'N\d', '', first_repeat)
            # add bar line after curly brace if required:
            if re.match(r'\}\s*\|.*', chord_string[repeat_match.end()-1:]):
                bar_line = ''
            else:
                bar_line = '|'
            new_chord_string = chord_string[:repeat_match.start()] + '|' + first_repeat + bar_line + \
                               chord_string[repeat_match.end():]

            # remove the first repeat ending as well as segnos and codas from the saved repeat
            repeat = cls._remove_markers(re.search(r'([^N]+)N\d', repeat_match.group(1)).group(1))
            # find the next repeat ending markers and insert the repeated chords before them
            new_chord_string = re.sub(r'\|\s*N\d', lambda m: '|' + repeat, new_chord_string)
            return new_chord_string
        else:
            # it's only a simple repeat: easy!
            new_chord_string = chord_string[:repeat_match.start()] + '|' + repeat_match.group(1) + \
                               ' |' + cls._remove_markers(repeat_match.group(1)) + chord_string[
                                                                                   repeat_match.end():] + '|'
            # there could be another repeat somewhere, so:
            new_chord_string = cls._fill_long_repeats(new_chord_string)
            return new_chord_string

    @classmethod
    def _fill_codas(cls, chord_string):
        """Flatten D.C. al Coda and D.S. al Coda
        :param chord_string: A chord string
        :return: A chord string with filled D.C. or D.S.
        """

        qs = chord_string.count('Q')
        if qs > 2:
            raise RuntimeError('Could not parse codas: number of Qs expected to be 0, 1 or 2, not {}!'.format(qs))

        # coda is used to indicate an outro: just get rid of it!
        if qs == 1:
            chord_string = chord_string.replace('Q', '')

        # this implies a repeat from the head or a segno to the first 'Q' and then a jump to the second 'Q'
        if qs == 2:
            q1, q2 = [pos for pos, char in enumerate(chord_string) if char == 'Q']
            segno = chord_string.find('S')
            if segno == -1:
                segno = 0
            coda = chord_string[q2 + 1:]
            repeat = chord_string[segno + 1:q1]
            new_chord_string = chord_string[:q2] + repeat + ' |' + coda
            new_chord_string = re.sub(r'[QS]', '', new_chord_string)
            return new_chord_string
        return chord_string

    @classmethod
    def _fill_single_double_repeats(cls, measures, sections=None):
        """Replaces one- and two-measure repeat symbols with the appropriate chords
        :param measures: A list of measures (as strings)
        :return: A list of measures with filled repeats
        """
        # single repeats:
        for i in range(1, len(measures)):
            if measures[i] == 'x':
                measures[i] = cls._remove_markers(measures[i - 1])
        # double repeats:
        i = 2
        while i < len(measures):
            if measures[i] == 'r':
                first = cls._remove_markers(measures[i-2])
                second = cls._remove_markers(measures[i-1])
                measures = measures[:i] + [first, second] + measures[i+1:]
                if sections is not None:
                    sections[:] = sections[:i] + [sections[i], sections[i]] + sections[i+1:]
                i += 1
            i += 1
        return measures

    @classmethod
    def _fill_slashes(cls, measures):
        """Replace slash symbols (encoded as 'p') with the previous chord
        :param measures: List of measures (as strings)
        :return: A list of measures with filled slashes
        """
        for i in range(len(measures)):
            while measures[i].find('p') != -1:
                slash = measures[i].find('p')
                previous_chords = cls.chord_regex.findall(measures[i][:slash])
                if not previous_chords:
                    previous_chords = [
                        chord
                        for previous_measure in reversed(measures[:i])
                        for chord in cls.chord_regex.findall(previous_measure)
                    ]
                if not previous_chords:
                    measures[i] = measures[i][:slash] + ' ' + measures[i][slash + 1:]
                    continue
                prev_chord = previous_chords[-1]
                repeated_slashes = re.match(r'p+', measures[i][slash:]).group()
                replacement = ' '.join([prev_chord] * len(repeated_slashes))
                measures[i] = measures[i][:slash] + replacement + measures[i][slash + len(repeated_slashes):]
        return measures

    @classmethod
    def _add_space_between_chords(cls, measures):
        """
        Add a space between chords in the same measure
        :param measures: List of measures as strings
        :return: List of measures with added spaces
        """
        for i in range(len(measures)):
            chords = [chord.strip() for chord in cls.chord_regex.findall(measures[i])]
            measures[i] = ', '.join(chords)
        return measures

    @classmethod
    def _replace_no_chords(cls, measures):
        """
        Replace n and nn with N.C.
        :param measures: List of measures as strings
        :return: List of measures
        """
        for i in range(len(measures)):
            chords = measures[i].split(', ')
            chords = ['' if chord.strip().lower() in ('n', 'nn') else chord for chord in chords]
            measures[i] = ', '.join(chords) if any(chords) else ' '
        return measures

    @classmethod
    def _get_measures_with_sections(cls, chord_string):
        """Flatten a chord string into measure strings and aligned section labels."""
        chord_string = cls._cleanup_chord_string(chord_string)
        section_labels = {}

        def preserve_section(match):
            marker_id = len(section_labels)
            section_labels[marker_id] = match.group(1)
            return '\x01' + str(marker_id) + '\x02'

        chord_string = re.sub(r'\*([A-Za-z])', preserve_section, chord_string)
        chord_string = cls._remove_annotations(chord_string)
        chord_string = cls._fill_long_repeats(chord_string)
        chord_string = cls._fill_codas(chord_string)
        raw_measures = re.split(r'\||LZ|K|Z|{|}|\[|\]', chord_string)
        measures = []
        sections = []
        section = None
        for raw_measure in raw_measures:
            section_markers = re.findall(r'\x01(\d+)\x02', raw_measure)
            if section_markers:
                section = section_labels[int(section_markers[-1])]
            measure = re.sub(r'\x01\d+\x02', '', raw_measure)
            if measure:
                measures.append(measure)
                sections.append(section)
        measures = cls._fill_single_double_repeats(measures, sections)
        measures = cls._fill_slashes(measures)
        measures = cls._add_space_between_chords(measures)
        measures = cls._replace_no_chords(measures)

        return measures, sections

    @classmethod
    def _get_measures(cls, chord_string):
        """Return flattened measures as strings, discarding empty measures."""
        measures, _ = cls._get_measures_with_sections(chord_string)
        return measures

    @classmethod
    def _get_measures_as_beats(cls, chord_string, measures_as_strings, beats_per_measure):
        """Place iReal chord cells on beats and align them with the expanded measures."""
        empty_cell = '\x01'
        separator = '\x02'
        chord_string = chord_string.replace('XyQ', empty_cell * 3)
        chord_string = chord_string.replace(',', separator).replace(' ', empty_cell)
        chord_string = cls._cleanup_chord_string(chord_string)

        def repeat_marker(measure):
            return re.sub(r'[\x01\x02]', '', measure).strip()

        rows = []
        markers = []
        measure_beats = []
        current_beat_count = max(1, beats_per_measure)
        raw_measures = re.split(r'\||LZ|K|Z|{|}|\[|\]', chord_string)
        for raw_measure in raw_measures:
            time_signature = re.search(r'T(\d)\d', raw_measure)
            if time_signature:
                current_beat_count = int(time_signature.group(1))
            raw_measure = cls._remove_annotations(raw_measure)
            if not re.sub(r'[\x01\x02]', '', raw_measure).strip():
                continue
            cell_source = cls._remove_markers(raw_measure).replace(separator, '')
            canonical = cls._remove_markers(
                raw_measure.replace(separator, ' ').replace(empty_cell, ' ')
            )
            rows.append((cell_source, canonical))
            markers.append(repeat_marker(raw_measure))
            measure_beats.append(max(1, current_beat_count))

        for i in range(1, len(rows)):
            if markers[i] == 'x':
                rows[i] = rows[i - 1]
                markers[i] = ''
        i = 2
        while i < len(rows):
            if markers[i] == 'r':
                rows[i:i + 1] = [rows[i - 2], rows[i - 1]]
                markers[i:i + 1] = [markers[i - 2], markers[i - 1]]
                measure_beats[i:i + 1] = [measure_beats[i - 2], measure_beats[i - 1]]
                i += 1
            i += 1

        default_beat_count = max(1, beats_per_measure)
        candidate_beats = []
        canonical_measures = [row[1] for row in rows]
        cell_chord_regex = re.compile(
            r'(?<!/)(nn|[A-GNn][^\sA-GNn/p\x01]*(?:/[A-G][#b]?)?)'
        )
        for (cell_source, _), beat_count in zip(rows, measure_beats):
            cells = []
            segments = cell_source.split(empty_cell)
            for segment_index, segment in enumerate(segments):
                position = 0
                for match in cell_chord_regex.finditer(segment):
                    preceding_slashes = segment[position:match.start()].count('p')
                    cells.extend([' '] * preceding_slashes)
                    chord = match.group(1)
                    cells.append('N.C.' if chord.lower() in ('n', 'nn') else chord)
                    position = match.end()
                trailing_slashes = segment[position:].count('p')
                cells.extend([' '] * trailing_slashes)
                if segment_index < len(segments) - 1:
                    cells.append(' ')

            if not cells:
                candidate_beats.append([' '] * beat_count)
                continue
            beats = [' '] * beat_count
            for cell_index, chord in enumerate(cells):
                beat_index = min(beat_count - 1, cell_index * beat_count // len(cells))
                if chord.strip():
                    while beat_index < beat_count - 1 and beats[beat_index].strip():
                        beat_index += 1
                    if beats[beat_index].strip():
                        beats[beat_index] += ', ' + chord
                    else:
                        beats[beat_index] = chord
            candidate_beats.append(beats)

        canonical_measures = cls._fill_slashes(canonical_measures)
        canonical_measures = cls._add_space_between_chords(canonical_measures)
        canonical_measures = cls._replace_no_chords(canonical_measures)

        def chord_key(measure):
            measure = measure.replace('N.C.', '')
            return tuple(cls.chord_regex.findall(measure))

        candidate_keys = [chord_key(measure) for measure in canonical_measures]
        measures_as_beats = []
        source_index = 0
        for measure in measures_as_strings:
            key = chord_key(measure)
            matches = [i for i, candidate in enumerate(candidate_keys) if candidate == key]
            if not matches:
                measures_as_beats.append([' '] * default_beat_count)
                continue
            following = [i for i in matches if i >= source_index]
            source_index = following[0] if following else matches[0]
            measures_as_beats.append(candidate_beats[source_index])
            source_index += 1
        return measures_as_beats

    @classmethod
    def _get_time_signature(cls, chord_string):
        """Get the time signature form a chord string
        :param chord_string: A chord string containing a time signature
        :return: Time signature as a tuple, e.g. (3, 4)
        """
        match = re.search(r'T(\d)(\d)', chord_string)
        if match is not None:
            a = int(match.group(1))
            b = int(match.group(2))
            return a, b
        else:
            return 4, 4

    def __init__(self, tune_string):
        """Make a new Tune object from a scrambled string extracted from a url, corresponding to a
        single tune. Note: This function will *not* accept a full iReal url, e.g. a string starting with *ireal://* -
        use ``parse_ireal_url`` for this.
        :param tune_string: Scrambled string for a single tune
        """
        parts = re.split(r"=+", tune_string)
        self.title = parts[0]
        self.composer = parts[1]
        self.style = parts[2]
        self.key = parts[3]
        offset = 0
        self.transpose = None
        if not parts[4].startswith(self._chords_prefix):
            offset = 1
            try:
                self.transpose = int(parts[4])
            except ValueError:
                self.transpose = None
        chords_scrambled = parts[4 + offset][len(self._chords_prefix):]
        self.comp_style = parts[5 + offset] if len(parts) > 5 + offset else None
        self.bpm = parts[6 + offset] if len(parts) > 6 + offset else None
        self.repeats = parts[7 + offset] if len(parts) > 7 + offset else None

        self.raw_chord_string = self._unscramble_chord_string(chords_scrambled)
        self.chord_string = self._cleanup_chord_string(self.raw_chord_string)
        self.time_signature = self._get_time_signature(self.chord_string)
        self.measures_as_strings, self.measure_sections = self._get_measures_with_sections(self.chord_string)
        self.measures_as_beats = self._get_measures_as_beats(
            self.raw_chord_string, self.measures_as_strings, self.time_signature[0]
        )


    def __repr__(self):
        """A nice representation containing the meta-data and the chords
        :return: String representation
        """
        result = super().__repr__() + '\n'
        result += 'Title: ' + self.title + '\n'
        result += 'Composer: ' + self.composer + '\n'
        result += 'Style: ' + self.style + '\n'
        result += 'Key: ' + self.key + '\n'
        result += 'Transpose: ' + str(self.transpose) + '\n'
        result += 'Comp style: ' + str(self.comp_style) + '\n'
        result += 'BPM: ' + str(self.bpm) + '\n'
        result += 'Repeats: ' + str(self.repeats) + '\n'
        result += 'Time signature: {}/{}'.format(*self.time_signature) + '\n\n'
        result += 'Chord string:\n' + self.chord_string + '\n\n'
        result += 'Flattened measures:\n'
        args = [iter(self.measures_as_strings)] * 4
        for chords in ([e for e in t if e is not None] for t in itertools.zip_longest(*args)):
            for chord in chords:
                result += '| {:12}'.format(chord)
            result += '|\n'
        return result

    @staticmethod
    def parse_ireal_url(url, verbose=False):
        """Parses iReal urls into human- and machine-readable formats

        :param url: A url containing one or more tunes
        :return: A list of Tune objects

        Example:

        ``list_of_tunes = Tune.parse_ireal_url('irealb://Example%20Song=Composer...)```
        """
        url = urllib.parse.unquote(url)
        match = re.match(r'irealb://([^"]+)', url)
        if match is None:
            raise RuntimeError('Provided string is not a valid iReal url!')
        # split url into individual songs along ===
        songs = re.split("===", match.group(1))
        tunes = []
        for song in songs:
            if song != '':
                try:
                    tune = Tune(song)
                    tunes.append(tune)
                    if verbose:
                        print('Parsed {}'.format(tune.title))
                except Exception as err:
                    if verbose:
                        print('Could not import song', song)
                        print(str(err))
        return tunes

    def as_dict(self):
        """Returns a dictionary representation of the Tune class, with all meta-data and the chord string
        :return: A dictionary
        """
        return {
            'title': self.title,
            'composer': self.composer,
            'style': self.style,
            'key': self.key,
            'transpose': self.transpose,
            'comp_style': self.comp_style,
            'bpm': self.bpm,
            'repeats': self.repeats,
            'time_signature': self.time_signature,
            'chord_string': self.chord_string,
            'measures_as_strings': self.measures_as_strings,
            'measure_sections': self.measure_sections,
            'measures_as_beats': self.measures_as_beats
        }