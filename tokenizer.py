from parser import Song

class Token:
    def __init__(self, chord, duration, beat_position, measure_position, measure_number):
        self.chord = chord
        self.duration = duration
        self.beat_position = beat_position
        self.measure_position = measure_position
        self.measure_number = measure_number

class IRealTokenizer:
    # A token should have
    # A chord
    # A duration
    # beat position
    # measure position
    # measure number
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize_song(self, song: Song ) -> list:
        """
        Tokenizes an IRealPro string into a list of tokens.
        """
        # Split the IRealPro string into lines
        lines = ireal_string.strip().splitlines()
        tokens = []
        for line in lines:
            # Tokenize each line using the provided tokenizer
            line_tokens = self.tokenizer.tokenize(line)
            tokens.extend(line_tokens)
        return tokens