import os
import json

from tqdm import tqdm

from midi_io import get_midi_files, load_file, save_file


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


if __name__ == "__main__":
    os.makedirs('../data/segments', exist_ok=True)

    for filepath in tqdm(get_midi_files("../data")):
        midi_data = load_file(filepath)
        segments = segment_data(midi_data, max_length=256)

        base_filename = os.path.basename(filepath)
        for idx, segment in enumerate(segments):
            segment_filename = f"../data/segments/{os.path.splitext(base_filename)[0]}_segment_{idx}.json"
            with open(segment_filename, 'w') as f:
                json.dump(segment, f)
