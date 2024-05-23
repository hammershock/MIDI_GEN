import json
import os
from typing import List, Dict, Literal, Union

from joblib import Memory
from pretty_midi import PrettyMIDI, Instrument, Note
from tqdm import tqdm

from symbols import start_symbol

memory = Memory('../.cache', verbose=0)


def get_midi_files(data_directory, ext="midi"):
    filepaths = []
    for root, dirs, files in os.walk(data_directory):
        for filename in files:
            basename, file_ext = os.path.splitext(filename)
            if file_ext.endswith(ext):
                filepaths.append(os.path.join(root, filename))
    return filepaths


def _process_data(midi_data: List[dict]):
    midi_data.sort(key=lambda x: x['start'])
    last_start = 0

    for note in midi_data:
        note['interval'] = note['start'] - last_start
        last_start = note['start']

    return midi_data


@memory.cache
def load_file(file_path: str, max_length=None) -> List[
    Dict[Literal['start', 'end', 'pitch', 'velocity', 'duration', 'interval'], Union[float, int, str]]]:
    """
    load symbols from midi file and wrap the notes with start symbol and end symbol
    :param file_path:
    :param max_length:
    :return:
    """
    midi = PrettyMIDI(file_path)

    notes = midi.instruments[0].notes[:max_length - 2] if max_length is not None else midi.instruments[0].notes

    # We should find a good way to mark the special tokens, be sure special tokens has separate embeddings
    end_symbol = {'start': notes[-1].end, 'end': notes[-1].end, 'pitch': -2, 'velocity': 64, 'duration': 0}
    midi_data = [start_symbol] + [{
        'start': note.start,
        'end': note.end,  # end time ascends
        'pitch': note.pitch,  # 0-127
        'velocity': note.velocity,  # 0-127
        'duration': note.end - note.start
    } for note in notes] + [end_symbol]

    return _process_data(midi_data)


def save_file(midi_data: List[Dict[str, Union[float, int, str]]], file_path: str):
    midi = PrettyMIDI()
    instrument = Instrument(program=0)

    current_time = 0
    for note_data in midi_data:
        start_time = current_time + note_data['interval']
        end_time = start_time + note_data['duration']
        note = Note(
            velocity=int(note_data['velocity']),
            pitch=int(note_data['pitch']),
            start=start_time,
            end=end_time
        )
        instrument.notes.append(note)
        current_time = start_time

    midi.instruments.append(instrument)
    midi.write(file_path)


def load_metadata(file_path: str) -> Dict[str, Union[float, int, str]]:
    with open(file_path, 'r') as f:
        data = json.load(f)
        meta_data = {filename: data['duration'][key] for key, filename in data['midi_filename'].items()}
        return meta_data


if __name__ == "__main__":
    data_dir = "../data"
    files = get_midi_files(data_dir)
    for file in tqdm(files):
        midi_data = load_file(file)
        print(len(midi_data))

    # metadata_path = "../data/maestro-v3.0.0/maestro-v3.0.0.json"
    # load_metadata(metadata_path)
