"""
Transformer decoder-only model: GPT

"""
from typing import Sequence, Tuple, List

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# TODO: complete this!

def preprocess(input_ids: Sequence[Tuple[int, int, float, float]]) -> Tensor:
    """
    # :param input_ids: a sequence of encoded Tuple(pitch_id: int, duration_id: int, interval_id: int, normed_velocity: float)
    :param input_ids: a sequence of encoded Tuple(pitch_id: int, duration_id: int, start_time: float, normed_velocity: float)
    :return:
    """
    pass


class MusicDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MusicTransformer(nn.Module):
    def __init__(self, pitch_vocab_size, duration_vocab_size, interval_vocab_size, velocity_vocab_size, embed_dim,
                 num_heads, num_layers):
        super(MusicTransformer, self).__init__()

        # Embedding layers
        self.pitch_embedding = nn.Embedding(pitch_vocab_size, embed_dim)
        self.duration_embedding = nn.Embedding(duration_vocab_size, embed_dim)
        self.start_time_embedding = nn.Linear(1, embed_dim)  # assuming start_time is a scalar value
        self.velocity_embedding = nn.Linear(1, embed_dim)  # assuming normed_velocity is a scalar value

        # Transformer decoder
        self.transformer = nn.Transformer(d_model=embed_dim, nhead=num_heads, num_decoder_layers=num_layers)

        # Output heads
        self.pitch_head = nn.Linear(embed_dim, pitch_vocab_size)
        self.duration_head = nn.Linear(embed_dim, duration_vocab_size)
        self.interval_head = nn.Linear(embed_dim, interval_vocab_size)
        self.velocity_head = nn.Linear(embed_dim, velocity_vocab_size)

    def forward(self, input_ids):
        pitch_ids = input_ids[:, :, 0]
        duration_ids = input_ids[:, :, 1]
        start_times = input_ids[:, :, 2].unsqueeze(-1)
        normed_velocities = input_ids[:, :, 3].unsqueeze(-1)

        # Embedding lookup and combine
        pitch_embeds = self.pitch_embedding(pitch_ids)
        duration_embeds = self.duration_embedding(duration_ids)
        start_time_embeds = self.start_time_embedding(start_times)
        velocity_embeds = self.velocity_embedding(normed_velocities)

        combined_embeds = pitch_embeds + duration_embeds + start_time_embeds + velocity_embeds

        # Transformer decoding
        transformer_output = self.transformer(combined_embeds)

        # Output predictions
        pitch_logits = self.pitch_head(transformer_output)
        duration_logits = self.duration_head(transformer_output)
        interval_logits = self.interval_head(transformer_output)
        velocity_logits = self.velocity_head(transformer_output)

        return pitch_logits, duration_logits, interval_logits, velocity_logits


# Example usage
if __name__ == "__main__":
    # Dummy data
    input_data = [
        torch.tensor([[60, 1, 0.0, 0.5], [62, 1, 0.5, 0.6]], dtype=torch.float),
        torch.tensor([[64, 2, 1.0, 0.7], [65, 1, 1.5, 0.8]], dtype=torch.float)
    ]

    dataset = MusicDataset(input_data)
    dataloader = DataLoader(dataset, batch_size=2)

    # Hyperparameters
    pitch_vocab_size = 128
    duration_vocab_size = 10
    interval_vocab_size = 10
    velocity_vocab_size = 128
    embed_dim = 512
    num_heads = 8
    num_layers = 6

    model = MusicTransformer(pitch_vocab_size, duration_vocab_size, interval_vocab_size, velocity_vocab_size, embed_dim,
                             num_heads, num_layers)

    for batch in dataloader:
        pitch_logits, duration_logits, interval_logits, velocity_logits = model(batch)
        print("Pitch logits:", pitch_logits)
        print("Duration logits:", duration_logits)
        print("Interval logits:", interval_logits)
        print("Velocity logits:", velocity_logits)
