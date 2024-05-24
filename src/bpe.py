from collections import defaultdict
from typing import Dict, Sequence, Tuple

import numpy as np
from tqdm import tqdm
from pretty_midi import PrettyMIDI

from midi_io import get_midi_files

from joblib import Memory

memory = Memory(location='../.cache', verbose=0)


@memory.cache
def get_stats(data_dir):
    stats = defaultdict(int)
    files = get_midi_files(data_dir)
    for file in tqdm(files):
        pm = PrettyMIDI(file)
        for slice in pm.get_piano_roll(fs=100).T:
            indices = np.where(slice > 0)[0]
            stats[tuple(indices)] += 1
    return stats


def music_bpe(stats: Dict[tuple, int], vocab_size=200):
    def roll_pair(seq: Sequence[tuple]):
        for i in range(len(seq) - 1):
            yield seq[i], seq[i + 1]

    vocab = {}
    seqs = {[(i, ) for i in k]: v for k, v in stats.items()}

    while len(vocab) < vocab_size:
        candidates = defaultdict(int)
        for seq, cnt in seqs.items():
            for m, n in roll_pair(seq):
                if m + n not in vocab:
                    candidates[m + n] += cnt

        chosen = max(candidates, key=candidates.get)
        vocab[chosen] = candidates[chosen]
        # TODO: collate the chosen pair, and modify seqs


if __name__ == '__main__':
    data_dir = '../data/'
    stats = get_stats(data_dir)
    print(len(stats))