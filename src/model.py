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
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * -(np.log(10000.0) / self.embed_dim))
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

    def forward(self, input_embeds, attention_mask):
        # Transformer decoding
        transformer_output = self.transformer(input_embeds, input_embeds, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=attention_mask)

        # Output predictions
        pitch_logits = self.pitch_head(transformer_output)
        duration_logits = self.duration_head(transformer_output)
        interval_logits = self.interval_head(transformer_output)
        velocity_logits = self.velocity_head(transformer_output)

        return pitch_logits, duration_logits, interval_logits, velocity_logits


class MidiLoss(nn.Module):
    def __init__(self, return_all=False):
        super(MidiLoss, self).__init__()
        self.pitch_loss = nn.CrossEntropyLoss()
        self.duration_loss = nn.CrossEntropyLoss()
        self.interval_loss = nn.CrossEntropyLoss()
        self.velocity_loss = nn.MSELoss()
        self.return_all = return_all

    def forward(self, pitch_logits, duration_logits, interval_logits, velocity_logits, pitch_targets, duration_targets, interval_targets, velocity_targets):
        # Ensure logits and targets are properly reshaped
        pitch_loss = self.pitch_loss(pitch_logits.reshape(-1, pitch_logits.size(-1)), pitch_targets.reshape(-1))
        duration_loss = self.duration_loss(duration_logits.reshape(-1, duration_logits.size(-1)), duration_targets.reshape(-1))
        interval_loss = self.interval_loss(interval_logits.reshape(-1, interval_logits.size(-1)), interval_targets.reshape(-1))
        velocity_loss = self.velocity_loss(velocity_logits.reshape(-1), velocity_targets.reshape(-1))
        losses = pitch_loss, duration_loss, interval_loss, velocity_loss
        return losses if self.return_all else sum(losses)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = MidiDataset("../data/segments", max_seq_len=256)
    pitch_vocab_size = dataset.midi_encoder.num_pitch
    duration_vocab_size = 32
    interval_vocab_size = 32

    embed_dim = 96
    num_heads = 8
    num_layers = 6

    lr = 1e-5
    batch_size = 128
    epochs = 10
    save_path = "../models/model.pt"
    log_filename = "../logs/train.log"

    logging.basicConfig(filename=log_filename, level=logging.INFO)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    model = MidiTransformer(pitch_vocab_size, duration_vocab_size, interval_vocab_size, embed_dim, num_heads, num_layers)
    criterion = MidiLoss(return_all=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = np.zeros(4)
        p_bar = tqdm(dataloader, desc="Epoch {}/{}".format(epoch, epochs))
        for idx, input_data in enumerate(p_bar):
            optimizer.zero_grad()

            # Shift input_ids to the right for teacher forcing
            pitch_input_ids = input_data['pitch_ids'][:, :-1]
            duration_input_ids = input_data['duration_ids'][:, :-1]
            start_time_input_ids = input_data['start_times'][:, :-1]
            velocity_input_ids = input_data['velocities'][:, :-1]

            # Prepare target ids
            pitch_target_ids = input_data['pitch_ids'][:, 1:]
            duration_target_ids = input_data['duration_ids'][:, 1:]
            interval_target_ids = input_data['interval_ids'][:, 1:]
            velocity_target_ids = input_data['velocities'][:, 1:]

            # Combine embeddings
            pitch_embeds = model.pitch_embedding(pitch_input_ids)
            duration_embeds = model.duration_embedding(duration_input_ids)
            start_time_embeds = model.start_time_embedding(start_time_input_ids)
            velocity_embeds = model.velocity_embedding(velocity_input_ids.unsqueeze(-1))

            input_embeds = pitch_embeds + duration_embeds + start_time_embeds + velocity_embeds

            # Attention mask
            attention_mask = input_data['attention_mask'][:, :-1]

            # Forward pass
            pitch_logits, duration_logits, interval_logits, velocity_logits = model(input_embeds, attention_mask)

            # Calculate losses
            losses = criterion(pitch_logits, duration_logits, interval_logits, velocity_logits,
                             pitch_target_ids, duration_target_ids, interval_target_ids, velocity_target_ids)

            total_loss = sum(losses)
            total_loss.backward()
            optimizer.step()
            running_loss += np.array([loss.item() for loss in losses])
            # print(f"Total Loss: {loss.item()}")
            mean_loss = running_loss / (idx + 1)
            p_bar.set_postfix(loss=np.sum(mean_loss), pitch_loss=mean_loss[0], duration_loss=mean_loss[1], interval_loss=mean_loss[2], velocity_loss=mean_loss[3])
            logging.info(f'epoch {epoch + 1}, loss {running_loss / (idx + 1)}')
            if idx % 100 == len(dataloader) // 4:
                torch.save(model.state_dict(), save_path)

