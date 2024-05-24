# model.py
"""
Transformer decoder-only model: GPT

"""
import logging
import math
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class MidiTransformer(nn.Module):
    def __init__(self, vocab_size, duration_vocab_size, num_heads, num_layers):
        super(MidiTransformer, self).__init__()
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, 48)
        self.duration_embedding = nn.Embedding(duration_vocab_size, 24)
        self.velocity_embedding = nn.Linear(1, 24)
        # concat the embeddings above and add positional embeddings
        self.positional_embedding = PositionalEncoding(96)

        embed_dim = 96

        # Transformer decoder
        self.transformer = nn.Transformer(d_model=embed_dim, nhead=num_heads, num_decoder_layers=num_layers, batch_first=True)

        # Output heads
        self.pitch_head = nn.Linear(embed_dim, vocab_size)
        self.duration_head = nn.Linear(embed_dim, duration_vocab_size)
        self.velocity_head = nn.Linear(embed_dim, 1)  # Regression output for velocities

    def make_embeds(self, pitch_input_ids, duration_input_ids, velocity_input_ids):
        # Get embeddings for each input type
        pitch_embeds = self.embedding(pitch_input_ids)
        duration_embeds = self.duration_embedding(duration_input_ids)
        velocity_embeds = self.velocity_embedding(velocity_input_ids.unsqueeze(-1).float())

        # Concatenate embeddings
        combined_embeds = torch.cat((pitch_embeds, duration_embeds, velocity_embeds), dim=-1)

        # Add positional embeddings
        combined_embeds = self.positional_embedding(combined_embeds)

        return combined_embeds

    def forward(self, input_embeds, attention_mask=None):
        # Transformer decoding
        transformer_output = self.transformer(input_embeds, input_embeds, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=attention_mask)

        # Output predictions
        pitch_logits = self.pitch_head(transformer_output)
        duration_logits = self.duration_head(transformer_output)
        velocity_logits = self.velocity_head(transformer_output)

        return pitch_logits, duration_logits, velocity_logits
