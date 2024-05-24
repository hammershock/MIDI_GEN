import enum
import re
from abc import ABC
from typing import Union

import numpy as np
from pretty_midi import Note


class TokenType(enum.Enum):
    NOTE = enum.auto()
    SPECIAL = enum.auto()
    OFFSET = enum.auto()


class SpecialTokenType(enum.Enum):
    CLS = enum.auto()
    EOS = enum.auto()
    PAD = enum.auto()


class Token(ABC):
    token_type: TokenType


class NoteToken(Token):
    token_type = TokenType.NOTE
    DURATIONS = [0.03, 0.05, 0.08, 0.11, 0.13, 0.16, 0.21, 0.26, 0.3, 0.35, 0.4, 0.47, 0.54, 0.62, 0.70, 0.78,
                 0.87, 0.98, 1.1, 1.24, 1.39, 1.57, 1.76, 1.98, 2.2, 2.43, 2.65, 2.88, 3.14, 3.43, 3.78, 4.17]
    MIN_PITCH = 50
    MAX_PITCH = 108

    def __init__(self, *, pitch, duration, velocity):
        self.pitch = pitch
        self.duration = duration
        self.velocity = velocity

        differences = [abs(duration - value) for value in NoteToken.DURATIONS]
        self.dur_idx = differences.index(min(differences))

    @staticmethod
    def from_note(note: Note):
        return NoteToken(pitch=note.pitch, duration=note.duration, velocity=note.velocity)

    @property
    def vel_norm(self) -> float:
        return (self.velocity - 64.69) / 19.02

    def __repr__(self):
        return f'<Note pitch={self.pitch} duration={self.duration:.2f} velocity={self.velocity}>'


class SpecialToken(Token):
    token_type = TokenType.SPECIAL

    def __init__(self, spec_type: Union[SpecialTokenType, str]):
        if isinstance(spec_type, str):
            en_chars = re.findall(r'[a-zA-Z]', spec_type)
            result = ''.join([char.upper() for char in en_chars])
            self.spec_type = SpecialTokenType._member_map_[result]
        elif isinstance(spec_type, SpecialTokenType):
            self.spec_type = spec_type

    def __repr__(self):
        return f'<{self.spec_type.name}>'


class OffsetToken(Token):
    token_type = TokenType.OFFSET

    OFFSETS = [0.00, 0.01, 0.02, 0.05, 0.08, 0.09, 0.1, 0.11, 0.13, 0.16, 0.19, 0.22, 0.25, 0.28, 0.32, 0.37,
               0.42, 0.48, 0.53, 0.59, 0.66, 0.74, 0.85, 0.98, 1.12, 1.26, 1.41, 1.59, 1.8, 2.05, 2.33, 2.64]

    def __init__(self, *, offset=None, idx=None):
        if offset is not None:
            differences = [abs(offset - value) for value in OffsetToken.OFFSETS]
            self.idx = differences.index(min(differences))
            self.offset = offset
        elif idx is not None:
            self.idx = idx
            self.offset = OffsetToken.OFFSETS[idx]

    def __repr__(self):
        return f'<offset {self.idx}>'
