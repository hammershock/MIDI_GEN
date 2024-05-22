"""
Music Byte Pair Encoding...
"""
import json
import os
from collections import defaultdict
from typing import List

from tqdm import tqdm

from dataset import load_file, get_midi_files, memory


min_interval = 0.01


@memory.cache
def process_data(midi_data: List[dict]):
    midi_data.sort(key=lambda x: x['start'])
    stats = defaultdict(list)
    last_start = 0

    for note in midi_data:
        interval = note['start'] - last_start
        duration = note['duration']
        pitch = note['pitch']
        velocity = note['velocity']

        stats['interval'].append(interval)
        stats['pitch'].append(pitch)
        stats['velocity'].append(velocity)
        stats['duration'].append(duration)

        last_start = note['start']
    return stats


if __name__ == '__main__':
    for filepath in tqdm(get_midi_files("./data")):
        midi_data = load_file(filepath)
        stats = process_data(midi_data)
        filename = os.path.basename(filepath)
        with open(f"./stats/{filename}.json", "w") as f:
            json.dump(stats, f)
        break

