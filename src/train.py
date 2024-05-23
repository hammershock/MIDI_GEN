import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm

from dataset import MidiDataset
from model import MidiTransformer


class MidiLoss(nn.Module):
    def __init__(self, return_all=False):
        super(MidiLoss, self).__init__()
        self.pitch_loss = nn.CrossEntropyLoss()
        self.duration_loss = nn.CrossEntropyLoss()
        self.interval_loss = nn.CrossEntropyLoss()
        self.velocity_loss = nn.MSELoss()
        self.return_all = return_all

    def forward(self, pitch_logits, duration_logits, interval_logits, velocity_logits, pitch_targets, duration_targets,
                interval_targets, velocity_targets):
        # Ensure logits and targets are properly reshaped
        pitch_loss = self.pitch_loss(pitch_logits.reshape(-1, pitch_logits.size(-1)), pitch_targets.reshape(-1))
        duration_loss = self.duration_loss(duration_logits.reshape(-1, duration_logits.size(-1)),
                                           duration_targets.reshape(-1))
        interval_loss = self.interval_loss(interval_logits.reshape(-1, interval_logits.size(-1)),
                                           interval_targets.reshape(-1))
        velocity_loss = self.velocity_loss(velocity_logits.reshape(-1), velocity_targets.reshape(-1))
        losses = pitch_loss, duration_loss, interval_loss, velocity_loss
        return losses if self.return_all else sum(losses)


if __name__ == "__main__":
    dataset = MidiDataset("../data/segments", max_seq_len=256)
    pitch_vocab_size = dataset.midi_encoder.num_pitch
    duration_vocab_size = 32
    interval_vocab_size = 32

    embed_dim = 96
    num_heads = 8
    num_layers = 6

    lr = 1e-3
    batch_size = 24
    epochs = 10
    save_path = "../models/model.pt"
    log_filename = "../logs/train.log"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logging.basicConfig(filename=log_filename, level=logging.INFO)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    model = MidiTransformer(pitch_vocab_size, duration_vocab_size, interval_vocab_size, embed_dim, num_heads,
                            num_layers).to(device)
    try:
        model.load_state_dict(torch.load(save_path))
    except:
        pass
    criterion = MidiLoss(return_all=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = np.zeros(4)
        p_bar = tqdm(dataloader, desc="Epoch {}/{}".format(epoch, epochs))
        for idx, input_data in enumerate(p_bar):
            optimizer.zero_grad()
            input_data = {k: v.to(device) for k, v in input_data.items()}
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

            input_embeds = model.make_embeds(pitch_input_ids, duration_input_ids, start_time_input_ids, velocity_input_ids)

            # Attention mask
            attention_mask = input_data['attention_mask'][:, :-1].to(device)

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
            p_bar.set_postfix(loss=np.sum(mean_loss), pitch_loss=mean_loss[0], duration_loss=mean_loss[1],
                              interval_loss=mean_loss[2], velocity_loss=mean_loss[3])
            logging.info(f'epoch {epoch + 1}, loss {running_loss / (idx + 1)}')
            torch.save(model.state_dict(), save_path)

