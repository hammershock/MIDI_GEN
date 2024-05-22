# dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset

from midi_io import get_midi_files, load_file, load_metadata
from midi_data_encoder import MidiEncoder
from segment_data import get_segments, load_segment


class MidiDataset(Dataset):
    def __init__(self, data_path, max_seq_len=256):
        self.midi_files = get_segments(data_path)
        self.max_seq_len = max_seq_len
        self.midi_encoder = MidiEncoder('../figs/quantize_rules.json', special_tokens=[-1, -2, -3])
        # self.meta_data_path = "../data/maestro-v3.0.0/maestro-v3.0.0.json"
        # self.meta_data = load_metadata(self.meta_data_path)

    def __len__(self):
        return len(self.midi_files)

    def __getitem__(self, idx):
        """
        Here We pad the data to max_seq_len

        :param idx:
        :return:
        """
        midi_data = load_segment(self.midi_files[idx])  # , max_length=self.max_seq_len
        pad_symbol = {'start': 0, 'end': 0, 'pitch': -3, 'velocity': 64, 'duration': 0, 'interval': 0}
        pad_length = self.max_seq_len - len(midi_data)
        att_mask = [0] * len(midi_data) + [1] * pad_length
        midi_data += [pad_symbol] * pad_length
        encoded_data = [self.midi_encoder.encode_note(note) for note in midi_data]
        pitch_ids, duration_ids, interval_ids, start_times, velocities = zip(*encoded_data)

        pitch_ids = torch.LongTensor(pitch_ids)
        duration_ids = torch.LongTensor(duration_ids)
        interval_ids = torch.LongTensor(interval_ids)
        start_times = torch.FloatTensor(start_times)
        velocities = torch.FloatTensor(velocities)

        # generate the attention mask properly...
        attention_mask = torch.LongTensor(np.array(att_mask)).bool()

        return {
            'pitch_ids': pitch_ids,  # note pitch symbols (special tokens included)
            'duration_ids': duration_ids,  # note durations
            'interval_ids': interval_ids,  # note intervals
            'start_times': start_times,  # FloatTensor, we shall use a proper embeddings for it
            'velocities': velocities,  # note intensity
            'attention_mask': attention_mask
        }


if __name__ == '__main__':
    dataset = MidiDataset('../data/segments')
    print(len(dataset))
