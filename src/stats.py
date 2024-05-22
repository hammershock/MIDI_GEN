"""
统计所有音乐的音符频率
"""
import json
from collections import defaultdict

from tqdm import tqdm

from midi_io import get_midi_files, load_file
from quantize import round_quantize


if __name__ == '__main__':
    vel_stats = defaultdict(int)
    pitch_stats = defaultdict(int)
    duration_stats = defaultdict(int)
    interval_stats = defaultdict(int)

    for filepath in tqdm(get_midi_files("../data")):
        midi_data = load_file(filepath)
        for note in midi_data:
            vel_stats[note['velocity']] += 1
            pitch_stats[note['pitch']] += 1

            duration = note['duration']
            interval = note['interval']

            quantized_duration = round_quantize(duration)
            quantized_interval = round_quantize(interval)

            duration_stats[quantized_duration] += 1
            interval_stats[quantized_interval] += 1

    with open('../figs/stats.json', 'w') as f:
        data = {'velocity': vel_stats, 'pitch': pitch_stats, 'duration': duration_stats, 'interval': interval_stats}
        json.dump(data, f)
