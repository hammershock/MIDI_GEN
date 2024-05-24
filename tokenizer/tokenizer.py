import json
import os

import numpy as np
from pretty_midi import PrettyMIDI
from tqdm import tqdm
from joblib import Memory

from .midi_token import SpecialToken, OffsetToken, NoteToken


memory = Memory("../.cache", verbose=0)
VOCAB_PATH = os.path.join(os.path.dirname(__file__), "vocab.json")


def get_midi_files(root_dir):
    return [os.path.join(root, file)for root, dirs, files in os.walk(root_dir)for file in files if file.endswith(".midi")]


@memory.cache
def tokenize(midi_file):
    pm = PrettyMIDI(midi_file)

    inst = pm.instruments[0]
    inst.notes.sort(key=lambda x: x.start)

    tokens = [SpecialToken("CLS"), NoteToken.from_note(inst.notes[0])]
    for prev, note in zip(inst.notes[:-1], inst.notes[1:]):
        tokens.append(NoteToken.from_note(note))
        tokens.append(OffsetToken(offset=note.start - prev.start))
    tokens = [token for token in tokens if not (isinstance(token, OffsetToken) and token.idx == 0)]
    tokens.append(SpecialToken("EOS"))

    return tokens


def build_dictionary():
    with open(VOCAB_PATH, "w") as f:
        vocab = [f"<Pitch {i}>" for i in range(NoteToken.MIN_PITCH, NoteToken.MAX_PITCH + 1)]
        vocab += ["<Offset>", "<CLS>", "<EOS>", "<PAD>", "<UNK>"]
        dictionary = {word: i for i, word in enumerate(vocab)}
        json.dump(dictionary, f)
    return dictionary


class Tokenizer:
    def __init__(self):
        self.dictionary = build_dictionary()

    def tokenize(self, midi_file):
        return tokenize(midi_file)

    def tokens_to_ids(self, tokens):
        ids = []
        unk, offset = self.dictionary["<UNK>"], self.dictionary["<Offset>"]
        # idx, dur_idx, vel, mask, pos_idx
        for idx, token in enumerate(tokens):
            pos = idx if (idx == 0 or tokens[idx - 1].token_type != token.token_type) else ids[-1][-1]
            if isinstance(token, NoteToken):
                ids.append((self.dictionary.get(f"<Pitch {token.pitch}>", unk), token.dur_idx, token.vel_norm, 0, pos))
            elif isinstance(token, OffsetToken):
                ids.append((self.dictionary[f"<Offset>"], token.idx, 0.0, 0, pos))
            elif isinstance(token, SpecialToken):
                ids.append((self.dictionary[f"<{token.spec_type.name}>"], 0, 0.0, 0, pos))
        return np.array(ids)


if __name__ == '__main__':
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(get_midi_files("../data")[0])
    ids = tokenizer.tokens_to_ids(tokens)

