import json
import os
from typing import Dict, Literal

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from midi_data_encoder import MidiEncoder

from midi_io import get_midi_files, load_file


def segment_data(midi_data, max_length=256):
    """
    Segment the midi data into pieces with a maximum length.

    :param midi_data: List of note data where each note is a dictionary
    :param max_length: Maximum length of each segment
    :return: List of segmented midi data
    """
    segments = []
    for i in range(0, len(midi_data), max_length):
        segment = midi_data[i:i + max_length]
        segments.append(segment)
    return segments


class MidiDataset(Dataset):
    def __init__(self, data_path, cache_path="../data/cache", max_seq_len=256):
        self.data_path = data_path
        self.cache_path = cache_path
        self.max_seq_len = max_seq_len
        self.midi_encoder = MidiEncoder('../figs/quantize_rules.json', special_tokens=[-1, -2, -3])

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
            self._build_cache()

        self.cache_files = sorted([os.path.join(self.cache_path, f) for f in os.listdir(self.cache_path) if f.endswith('.npy')])

    def _build_cache(self):
        idx = 0
        for midi_file in tqdm(get_midi_files(self.data_path), "building cache"):
            midi_data = load_file(midi_file)
            for data in segment_data(midi_data, self.max_seq_len):
                pad_length = self.max_seq_len - len(data)
                pad_symbol = {'start': 0, 'end': 0, 'pitch': -3, 'velocity': 64, 'duration': 0, 'interval': 0}
                att_mask = [0] * len(data) + [1] * pad_length
                assert len(att_mask) == self.max_seq_len
                data += [pad_symbol] * pad_length
                encoded_data = self.midi_encoder.encode_notes(data)
                encoded_data['attention_mask'] = np.array(att_mask).astype(np.bool_)

                cache_file = os.path.join(self.cache_path, f"{idx}.npy")
                np.save(cache_file, encoded_data)
                idx += 1

    def __len__(self):
        return len(self.cache_files)

    def __getitem__(self, idx) -> Dict[Literal[
        'pitch_ids', 'duration_ids', 'interval_ids', 'start_times', 'velocities', 'attention_mask'], torch.Tensor]:
        encoded_data = np.load(self.cache_files[idx], allow_pickle=True).item()
        return encoded_data


if __name__ == '__main__':
    dataset = MidiDataset('../data')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in tqdm(dataloader):
        assert batch['pitch_ids'].shape[1] == 256
        assert batch['duration_ids'].shape[1] == 256
        assert batch['interval_ids'].shape[1] == 256
        assert batch['start_times'].shape[1] == 256
        assert batch['velocities'].shape[1] == 256
        assert batch['attention_mask'].shape[1] == 256
        # print(batch.shape, batch['duration_ids'].shape, batch['interval_ids'].shape)

