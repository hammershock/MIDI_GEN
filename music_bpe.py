"""
Music Byte Pair Encoding...
"""
import json
import os
from collections import defaultdict
from typing import List

from tqdm import tqdm

from dataset import load_file, get_midi_files


min_interval = 0.01



if __name__ == '__main__':
    for filepath in tqdm(get_midi_files("./data")):
        midi_data = load_file(filepath)
        # stats = process_data(midi_data)
        # filename = os.path.basename(filepath)
        # with open(f"./stats/{filename}.json", "w") as f:
        #     json.dump(stats, f)
        # break

