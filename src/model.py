# model.py
"""
Transformer decoder-only model: GPT

"""
import logging
import os
import sys
from typing import Sequence, Tuple, List

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from dataset import MidiDataset


class StartTimeEmbedding(nn.Module):
    """
    The embeddings are dynamically generated during training and inferencing
    i.e. Sine periodicity START Time embeddings
    Given the embed-dim, sequence length and Start times
    :param embed_dim:
    """
    def __init__(self, embed_dim):
        super(StartTimeEmbedding, self).__init__()
        self.embed_dim = embed_dim

    def forward(self, start_times):
        """
        start_times: (batch_size, seq_len) -> times at which notes start
        """
        batch_size, seq_len = start_times.size()
        position = start_times.unsqueeze(-1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * -(np.log(10000.0) / self.embed_dim)).to(start_times.device)
        pe = torch.zeros(batch_size, seq_len, self.embed_dim).to(start_times.device)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe


class MidiTransformer(nn.Module):
    def __init__(self, pitch_vocab_size, duration_vocab_size, interval_vocab_size, embed_dim, num_heads, num_layers):
        super(MidiTransformer, self).__init__()
        self.embed_dim = embed_dim

        # Embedding layers
        self.pitch_embedding = nn.Embedding(pitch_vocab_size, embed_dim)
        self.duration_embedding = nn.Embedding(duration_vocab_size, embed_dim)
        self.start_time_embedding = StartTimeEmbedding(embed_dim)
        self.velocity_embedding = nn.Linear(1, embed_dim)

        # Transformer decoder
        self.transformer = nn.Transformer(d_model=embed_dim, nhead=num_heads, num_decoder_layers=num_layers, batch_first=True)

        # Output heads
        self.pitch_head = nn.Linear(embed_dim, pitch_vocab_size)
        self.duration_head = nn.Linear(embed_dim, duration_vocab_size)
        self.interval_head = nn.Linear(embed_dim, interval_vocab_size)
        self.velocity_head = nn.Linear(embed_dim, 1)  # Regression output for velocities

    def forward(self, input_embeds, attention_mask=None):
        # Transformer decoding
        transformer_output = self.transformer(input_embeds, input_embeds, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=attention_mask)

        # Output predictions
        pitch_logits = self.pitch_head(transformer_output)
        duration_logits = self.duration_head(transformer_output)
        interval_logits = self.interval_head(transformer_output)
        velocity_logits = self.velocity_head(transformer_output)

        return pitch_logits, duration_logits, interval_logits, velocity_logits

    def make_embeds(self, pitch_input_ids, duration_input_ids, start_time_input_ids, velocity_input_ids):
        # Combine embeddings
        pitch_embeds = self.pitch_embedding(pitch_input_ids)
        duration_embeds = self.duration_embedding(duration_input_ids)
        start_time_embeds = self.start_time_embedding(start_time_input_ids)
        velocity_embeds = self.velocity_embedding(velocity_input_ids.unsqueeze(-1))
        return pitch_embeds + duration_embeds + start_time_embeds + velocity_embeds


if __name__ == "__main__":
    dataset = MidiDataset("../data/segments", max_seq_len=256)
    # pitch_vocab_size = dataset.midi_encoder.num_pitch
    pitch_vocab_size = 62
    duration_vocab_size = 32
    interval_vocab_size = 32

    model = MidiTransformer(62, 32, 32, 96, 8, 6)

    embed_dim = 96
    num_heads = 8
    num_layers = 6

    lr = 1e-4
    batch_size = 128
    epochs = 10
    save_path = "../models/model.pt"
    log_filename = "../logs/train.log"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model = MidiTransformer(pitch_vocab_size, duration_vocab_size, interval_vocab_size, embed_dim, num_heads,
                            num_layers)
    torch.save(model.state_dict(), "../models/test.pt")
