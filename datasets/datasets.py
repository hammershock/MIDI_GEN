import os
from pathlib import Path
from typing import Dict, Literal

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from tokenizer import get_midi_files, Tokenizer


CACHE_DIR = Path(__file__).resolve().parent / 'cache'


class MidiDataset(Dataset):
    def __init__(self, midi_root_path, max_len=256):
        self.midi_root_path = midi_root_path
        self.max_len = max_len
        self.tokenizer = Tokenizer()

        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
            self._build_cache()

        self.length = len(os.listdir(CACHE_DIR))

    def _build_cache(self):
        n = 0
        for midi_file in tqdm(get_midi_files(self.midi_root_path)):
            tokens = self.tokenizer.tokenize(midi_file)
            ids = self.tokenizer.tokens_to_ids(tokens)  # (N, 5)
            num_segments = (len(ids) + self.max_len - 1) // self.max_len  # 计算分段数目
            for idx in range(num_segments):
                start_idx = idx * self.max_len
                end_idx = start_idx + self.max_len
                segment = ids[start_idx:end_idx]
                if len(segment) < self.max_len:
                    pad_len = self.max_len - len(segment)
                    pad_array = np.array([[self.tokenizer.dictionary["<PAD>"], 0, 0, 1, idx]] * pad_len)
                    segment = np.vstack((segment, pad_array))
                token_ids = torch.from_numpy(segment[:, 0]).long()
                dur_ids = torch.from_numpy(segment[:, 1]).long()
                vel_values = torch.from_numpy(segment[:, 2]).float()
                attention_mask = torch.from_numpy(segment[:, 3]).bool()
                position_ids = torch.from_numpy(segment[:, 4]).long()
                data = {"duration_ids": dur_ids, "position_ids": position_ids, "attention_mask": attention_mask,
                        "ids": token_ids, "velocities": vel_values}
                np.save(CACHE_DIR / f'{n}.npy', data, allow_pickle=True)
                n += 1

    def __getitem__(self, index) -> Dict[Literal['duration_ids', 'position_ids', 'attention_mask', 'ids', 'velocities'], torch.Tensor]:
        cache_path = CACHE_DIR / f'{index}.npy'
        assert cache_path.exists()
        data = np.load(cache_path, allow_pickle=True).item()
        return data

    def __len__(self):
        return self.length


if __name__ == '__main__':
    dataset = MidiDataset(midi_root_path='../data')
    item = dataset[0]
    print(item)
