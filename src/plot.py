import pretty_midi
import matplotlib.pyplot as plt
import numpy as np

from midi_io import get_midi_files


def plot_piano_roll(pm, start_time=None, end_time=None):
    fig, ax = plt.subplots(figsize=(12, 8))

    # Extract piano roll
    piano_roll = pm.get_piano_roll(fs=100)
    # Set default start_time and end_time if not provided
    if start_time is None:
        start_time = 0
    if end_time is None:
        end_time = pm.get_end_time()

    # Create a time axis
    times = np.linspace(start_time, end_time, piano_roll.shape[1])

    # Create a colormap
    cmap = plt.get_cmap('viridis')

    # Plot piano roll
    for note in pm.instruments[0].notes:
        if start_time <= note.start < end_time:
            color = cmap(note.pitch / 128)
            ax.add_patch(plt.Rectangle((note.start, note.pitch), note.end - note.start, 1, color=color))

    ax.set_xlim([start_time, end_time])
    ax.set_ylim([0, 128])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pitch')
    ax.set_title('Piano Roll')
    plt.show()


def analyse_chords(filename):
    pm = pretty_midi.PrettyMIDI(filename)
    piano_roll = pm.get_piano_roll(fs=100)
    for time_slice in piano_roll.T:
        combinations = np.where(time_slice > 0)[0]
        strengths = time_slice[combinations]
        offsets = np.min(combinations) if len(combinations) > 0 else 0
        pattern = tuple(combinations - offsets)


if __name__ == '__main__':
    midi_files = get_midi_files("../data")
    analyse_chords(midi_files[0])
    # Load MIDI file
    # midi_data = pretty_midi.PrettyMIDI(midi_files[0])
    #
    # # Plot piano roll from 0 to 10 seconds
    # plot_piano_roll(midi_data, 100, 150)
