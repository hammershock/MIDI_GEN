import numpy as np
import torch
from tqdm import tqdm

from model import MidiTransformer
from midi_io import save_file
from midi_data_encoder import MidiEncoder
from symbols import start_symbol, end_symbol

if __name__ == '__main__':
    model_path = "../models/test.pt"
    seq_length = 500
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MidiTransformer(62, 32, 32, 96, 8, 6).to(device)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    encoder = MidiEncoder('../figs/quantize_rules.json', special_tokens=[-1, -2, -3])

    midi_data = [start_symbol]
    encoded_data = []
    last_start = 0.0

    for i in tqdm(range(seq_length)):
        input_ids = encoder.encode_notes(midi_data[-256:])
        input_ids = {k: v.to(device).unsqueeze(0) for k, v in input_ids.items()}
        embeddings = model.make_embeds(input_ids['pitch_ids'], input_ids['duration_ids'], input_ids['start_times'], input_ids['velocities']).to(device)
        outputs = model(embeddings, attention_mask=None)
        pitch_logit, duration_logit, interval_logit, velocity_logit = [item[0] for item in outputs]

        pitch: int = pitch_logit.argmax(dim=1).cpu().detach().numpy()[-1]
        duration: int = duration_logit.argmax(dim=1).cpu().detach().numpy()[-1]
        interval: int = interval_logit.argmax(dim=1).cpu().detach().numpy()[-1]
        velocity: float = velocity_logit.squeeze(1).cpu().detach().numpy()[-1]
        note = encoder.decode_note((pitch, duration, interval, velocity))
        if note['pitch'] == end_symbol:
            break
        note['start'] = last_start + note['interval']
        note['end'] = note['start'] + note['duration']
        last_start = note['start']
        print(last_start)
        midi_data.append(note)

    midi_data.pop(0)
    for note in midi_data:
        print(note['duration'], note['pitch'], note['velocity'], note['start'], note['end'])
    save_file(midi_data, '../media/test.midi')

