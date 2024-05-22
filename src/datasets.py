import torch
from torch.utils.data import Dataset

from midi_io import get_midi_files, load_file
from midi_data_encoder import MidiEncoder


class MidiDataset(Dataset):
    def __init__(self, data_path, max_seq_len=128):
        """

        :param data_path: root directory of midi files
        """
        self.midi_files = get_midi_files(data_path)
        self.max_seq_len = max_seq_len
        self.midi_encoder = MidiEncoder('../figs/quantize_rules.json')

    def __len__(self):
        return len(self.midi_files)

    def __getitem__(self, idx):
        midi_data = load_file(self.midi_files[idx])

        for note in midi_data:
            # note has keys: 'start', 'end', 'pitch', 'velocity', 'duration', 'interval'
            pitch_id, duration_id, start_time_value, velocity_value = self.midi_encoder.encode_note(note)
            # pitch_id: int, duration_id: int, start_time_value: float, velocity_value: float

        # TODO: wrap the notes with start symbol and end symbol, and pad the data into self.max_seq_len
        # TODO: find a good way to mark the special tokens, be sure special tokens has separate embeddings
        # be careful, pitch_id and duration_id should become torch.Long; start_time_value and velocity_value should
        # become torch.FloatTensor...
        # TODO: generate the attention mask properly...
