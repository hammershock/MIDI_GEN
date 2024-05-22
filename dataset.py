import os
import json
from typing import List, Dict, Literal, Union

from matplotlib import pyplot as plt
from pretty_midi import PrettyMIDI
from joblib import Memory
from tqdm import tqdm


memory = Memory('.cache', verbose=0)


def get_midi_files(data_directory, ext="midi"):
    filepaths = []
    for root, dirs, files in os.walk(data_directory):
        for filename in files:
            basename, file_ext = os.path.splitext(filename)
            if file_ext.endswith(ext):
                filepaths.append(os.path.join(root, filename))
    return filepaths


def process_data(midi_data: List[dict]):
    midi_data.sort(key=lambda x: x['start'])
    last_start = 0

    for note in midi_data:
        note['interval'] = note['start'] - last_start
        last_start = note['start']

    return midi_data


@memory.cache
def load_file(file_path: str) -> List[Dict[Literal['start', 'end', 'pitch', 'velocity', 'duration', 'interval'], Union[float, int, str]]]:
    midi = PrettyMIDI(file_path)

    midi_data = []
    for note in midi.instruments[0].notes:
        data = {
            'start': note.start,
            'end': note.end,  # end time ascends
            'pitch': note.pitch,  # 0-127
            'velocity': note.velocity,  # 0-127
            'duration': note.end - note.start
        }
        midi_data.append(data)

    return process_data(midi_data)


if __name__ == "__main__":
    data_dir = "./data"
    files = get_midi_files(data_dir)
    for file in tqdm(files):
        midi_data = load_file(file)
