import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os
from tqdm import tqdm
import numpy as np

from text import text_to_sequence

from functools import partial


def pad_1D(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = np.pad(x, (0, length - x.shape[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_1D_tensor(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = F.pad(x, (0, length - x.shape[0]))
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = torch.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):

    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, max_len - np.shape(x)[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad_2D_tensor(inputs, maxlen=None):

    def pad(x, max_len):
        if x.size(0) > max_len:
            raise ValueError("not max_len")

        s = x.size(1)
        x_padded = F.pad(x, (0, 0, 0, max_len-x.size(0)))
        return x_padded[:, :s]

    if maxlen:
        output = torch.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(x.size(0) for x in inputs)
        output = torch.stack([pad(x, max_len) for x in inputs])

    return output


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)

        return txt


def get_data_to_buffer(config):
    buffer = list()
    text = process_text(config["text_path"])

    for i in tqdm(range(len(text)), desc='Loading data to buffer'):

        mel_filename = os.path.join(config["mel_path"], "ljspeech-mel-%05d.npy" % (i + 1))
        mel = np.load(mel_filename)
        duration = np.load(os.path.join(config["alignment_path"], str(i) + ".npy"))
        pitch = np.load(os.path.join(config["pitch_path"], str(i) + ".npy"))
        pitch_mean = np.array([pitch[0]])
        pitch_std = np.array([pitch[1]])
        pitch = pitch[2:]
        energy = np.load(os.path.join(config["energy_path"], str(i)+".npy"))
        
        character = text[i][0: len(text[i]) - 1]
        character = np.array(text_to_sequence(character, config["text_cleaners"]))

        character = torch.from_numpy(character)
        duration = torch.from_numpy(duration)
        pitch = torch.from_numpy(pitch)
        pitch_mean = torch.from_numpy(pitch_mean)
        pitch_std = torch.from_numpy(pitch_std)
        energy = torch.from_numpy(energy)
        mel = torch.from_numpy(mel)

        buffer.append({"text": character, 
                       "duration": duration,
                       "pitch": pitch,
                       "pitch_mean": pitch_mean,
                       "pitch_std": pitch_std,
                       "energy": energy,
                       "mel": mel})

    return buffer


class BufferDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer
        self.length_dataset = len(self.buffer)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.buffer[idx]


def reprocess_tensor(batch, cut_list):
    texts = [batch[ind]["text"] for ind in cut_list]
    mel_targets = [batch[ind]["mel"] for ind in cut_list]
    durations = [batch[ind]["duration"] for ind in cut_list]
    pitches = [batch[ind]["pitch"] for ind in cut_list]
    pitch_means = [batch[ind]["pitch_mean"] for ind in cut_list]
    pitch_stds = [batch[ind]["pitch_std"] for ind in cut_list]
    energies = [batch[ind]["energy"] for ind in cut_list]

    length_text = np.array([])
    for text in texts:
        length_text = np.append(length_text, text.size(0))

    src_pos = list()
    max_len = int(max(length_text))
    for length_src_row in length_text:
        src_pos.append(np.pad([i+1 for i in range(int(length_src_row))],
                              (0, max_len-int(length_src_row)), 'constant'))
    src_pos = torch.from_numpy(np.array(src_pos))

    length_mel = np.array(list())
    for mel in mel_targets:
        length_mel = np.append(length_mel, mel.size(0))

    mel_pos = list()
    max_mel_len = int(max(length_mel))
    for length_mel_row in length_mel:
        mel_pos.append(np.pad([i+1 for i in range(int(length_mel_row))],
                              (0, max_mel_len-int(length_mel_row)), 'constant'))
    mel_pos = torch.from_numpy(np.array(mel_pos))

    texts = pad_1D_tensor(texts)
    mel_targets = pad_2D_tensor(mel_targets)
    durations = pad_1D_tensor(durations)
    pitches = pad_1D_tensor(pitches)
    pitch_means = torch.tensor(pitch_means)
    pitch_stds = torch.tensor(pitch_stds)
    energies = pad_1D_tensor(energies)

    out = {"text": texts.long(),
           "mel": mel_targets.float(),
           "duration": durations.float(),
           "pitch": pitches.float(),
           "pitch_mean": pitch_means.float(),
           "pitch_std": pitch_stds.float(),
           "energy": energies.float(),
           "mel_pos": mel_pos.long(),
           "src_pos": src_pos.long(),
           "mel_max_len": max_mel_len}

    return out


def collate_fn_tensor(batch, config=None):
    len_arr = np.array([d["text"].size(0) for d in batch])
    index_arr = np.argsort(-len_arr)
    batchsize = len(batch)
    real_batchsize = batchsize // config["batch_expand_size"]

    cut_list = list()
    for i in range(config["batch_expand_size"]):
        cut_list.append(index_arr[i * real_batchsize: (i + 1) * real_batchsize])

    output = list()
    for i in range(config["batch_expand_size"]):
        output.append(reprocess_tensor(batch, cut_list[i]))

    return output



def get_dataloader(config):
    buffer = get_data_to_buffer(config)

    dataset = BufferDataset(buffer)

    training_loader = DataLoader(
        dataset,
        batch_size=config["batch_expand_size"] * config["batch_size"],
        shuffle=True,
        collate_fn=partial(collate_fn_tensor, config=config),
        drop_last=True,
        num_workers=2
    )

    return training_loader

