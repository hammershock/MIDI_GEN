import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm

from datasets import MidiDataset
from model import MidiTransformer


class MidiLoss(nn.Module):
    def __init__(self, return_all=False):
        super(MidiLoss, self).__init__()
        self.pitch_loss = nn.CrossEntropyLoss()
        self.duration_loss = nn.CrossEntropyLoss()
        self.velocity_loss = nn.MSELoss()
        self.return_all = return_all

    def forward(self, pitch_logits, duration_logits, velocity_logits, pitch_targets, duration_targets, velocity_targets):
        # Ensure logits and targets are properly reshaped
        pitch_loss = self.pitch_loss(pitch_logits.reshape(-1, pitch_logits.size(-1)), pitch_targets.reshape(-1))
        duration_loss = self.duration_loss(duration_logits.reshape(-1, duration_logits.size(-1)),
                                           duration_targets.reshape(-1))
        velocity_loss = self.velocity_loss(velocity_logits.reshape(-1), velocity_targets.reshape(-1))
        # losses = pitch_loss, duration_loss, velocity_loss
        return pitch_loss

        return losses if self.return_all else sum(losses)



if __name__ == "__main__":
    dataset = MidiDataset("../data", max_len=256)
    vocab_size = len(dataset.tokenizer.dictionary)
    duration_vocab_size = 32

    lr = 1e-3
    batch_size = 2
    epochs = 10
    save_path = "../models/model.pt"
    log_filename = "../logs/train.log"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logging.basicConfig(filename=log_filename, level=logging.INFO)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16)
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)

    model = MidiTransformer(vocab_size, duration_vocab_size, num_heads=8, num_layers=6).to(device)
    try: model.load_state_dict(torch.load(save_path))
    except: pass

    criterion = MidiLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        p_bar = tqdm(dataloader, desc="Epoch {}/{}".format(epoch, epochs))
        for idx, input_data in enumerate(p_bar):
            # 'duration_ids', 'position_ids', 'attention_mask', 'ids', 'velocities'
            optimizer.zero_grad()
            input_data = {k: v.to(device) for k, v in input_data.items()}  # move to device
            # Shift input_ids to the right for teacher forcing
            pitch_input_ids = input_data['ids'][:, :-1]
            duration_input_ids = input_data['duration_ids'][:, :-1]
            velocity_input_ids = input_data['velocities'][:, :-1]
            attention_mask = input_data['attention_mask'][:, :-1]  # Attention mask
            position_ids = input_data['position_ids'][:, :-1]  # position_ids

            # Prepare target ids
            pitch_target_ids = input_data['ids'][:, 1:]
            duration_target_ids = input_data['duration_ids'][:, 1:]
            velocity_target_ids = input_data['velocities'][:, 1:]

            input_embeds = model.make_embeds(pitch_input_ids, duration_input_ids, velocity_input_ids)

            # Forward pass
            pitch_logits, duration_logits, velocity_logits = model(input_embeds, attention_mask)

            # Calculate losses
            total_loss = criterion(pitch_logits, duration_logits, velocity_logits, pitch_target_ids, duration_target_ids, velocity_target_ids)
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
            # print(f"Total Loss: {loss.item()}")
            mean_loss = running_loss / (idx + 1)
            p_bar.set_postfix(loss=np.sum(mean_loss))
        logging.info(f'epoch {epoch + 1}, loss {running_loss / len(dataloader)}')
        torch.save(model.state_dict(), save_path)

