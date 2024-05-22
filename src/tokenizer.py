from typing import Sequence, List, Union, Tuple
from bisect import bisect_left
import json

import numpy as np


class Tokenizer:
    def __init__(self, quantize_rule_file: str, min_pitch=50, max_pitch=108):
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.num_classes_pitch = max_pitch - min_pitch + 1

        with open(quantize_rule_file) as f:
            quantize_rules = json.load(f)
        self._quantized_durations = quantize_rules['duration']
        self._quantized_intervals = quantize_rules['interval']

        self.velocity_mean = 64.69024144892079
        self.velocity_std = 19.02232476382681

    def _pitch_encode(self, pitch) -> int:
        if pitch < self.min_pitch:
            return 0
        elif pitch > self.max_pitch:
            return self.num_classes_pitch - 1
        else:
            return pitch - self.min_pitch

    def _pitch_decode(self, pitch: int) -> int:
        return pitch + self.min_pitch

    def _quantize_encode(self, value, quantized_values) -> int:
        pos = bisect_left(quantized_values, value)
        if pos == 0:
            return 0  # quantized_values[0]
        if pos == len(quantized_values):
            return len(quantized_values) - 1  # quantized_values[-1]
        before = quantized_values[pos - 1]
        after = quantized_values[pos]
        if after - value < value - before:
            return pos  # after
        else:
            return pos - 1  # before

    def _quantize_decode(self, idx: int, quantized_values) -> float:
        return quantized_values[idx]

    def _encode_velocity(self, velocity: int) -> float:
        return (velocity - self.velocity_mean) / self.velocity_std

    def _decode_velocity(self, velocity: float) -> int:
        velocity = velocity * self.velocity_std + self.velocity_mean
        return int(np.clip(velocity, 0, 255))

    def encode_note(self, note: dict) -> Tuple[int, int, int, float]:
        pitch_encoded = self._pitch_encode(note['pitch'])
        duration_encoded = self._quantize_encode(note['duration'], self._quantized_durations)
        interval_encoded = self._quantize_encode(note['interval'], self._quantized_intervals)
        velocity_encoded = self._encode_velocity(note['velocity'])
        return pitch_encoded, duration_encoded, interval_encoded, velocity_encoded

    def decode_note(self, token: Tuple[int, int, int, float]) -> dict:
        pitch_decoded = self._pitch_decode(token[0])
        duration_decoded = self._quantize_decode(token[1], self._quantized_durations)
        interval_decoded = self._quantize_decode(token[2], self._quantized_intervals)
        velocity_decoded = self._decode_velocity(token[3])
        note = {"pitch": pitch_decoded, "duration": duration_decoded, "interval": interval_decoded, "velocity": velocity_decoded}
        return note

    def tokenize(self, midi_data: Sequence[dict]) -> List[Tuple[int, int, int, float]]:
        return [self.encode_note(note) for note in midi_data]

    def token_from_id(self, ids: List[Tuple[int, int, int, float]]):
        return [self.decode_note(id) for id in ids]


if __name__ == '__main__':
    from midi_io import get_midi_files, load_file, save_file

    tokenizer = Tokenizer(quantize_rule_file='../figs/quantize_rules.json')
    for filepath in get_midi_files("../data"):
        midi_data = load_file(filepath)
        inputs_ids = tokenizer.tokenize(midi_data)
        midi_data_recovered = tokenizer.token_from_id(inputs_ids)
        for note, note_recovered in zip(midi_data, midi_data_recovered):
            print(note['pitch'], note_recovered['pitch'])
        save_file(midi_data_recovered, "./output.midi")
        save_file(midi_data, "./input.midi")
        # print(midi_data_recovered)
        break


