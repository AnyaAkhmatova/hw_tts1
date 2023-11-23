import torch
from torch import nn

import numpy as np

from .transformers import Encoder, Decoder
from .predictors import LengthRegulator, PitchPredictor, EnergyPredictor


def get_mask_from_lengths(lengths, max_len=None):
    if max_len == None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, 1, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()

    return mask


class FastSpeech2(nn.Module):

    def __init__(self, 
                 max_seq_len,
                 vocab_size,
                 encoder_n_layer,
                 encoder_dim,
                 encoder_head,
                 encoder_conv1d_filter_size,
                 decoder_n_layer,
                 decoder_dim,
                 decoder_head,
                 decoder_conv1d_filter_size,
                 pad_id,
                 fft_conv1d_kernel,
                 fft_conv1d_padding,
                 dropout,
                 duration_predictor_filter_size,
                 duration_predictor_kernel_size,
                 pitch_predictor_filter_size,
                 pitch_predictor_kernel_size,
                 pitch_n_emb,
                 min_pitch, 
                 max_pitch,
                 energy_predictor_filter_size,
                 energy_predictor_kernel_size,
                 energy_n_emb,
                 min_energy,
                 max_energy,
                 num_mels):
        super(FastSpeech2, self).__init__()

        self.encoder = Encoder(max_seq_len,
                               vocab_size,
                               encoder_n_layer,
                               encoder_dim,
                               encoder_head,
                               pad_id,
                               encoder_conv1d_filter_size,
                               fft_conv1d_kernel,
                               fft_conv1d_padding,
                               dropout)

        self.length_regulator = LengthRegulator(encoder_dim,
                                                duration_predictor_filter_size,
                                                duration_predictor_kernel_size,
                                                dropout)
        
        self.pitch_predictor = PitchPredictor(encoder_dim,
                                              pitch_predictor_filter_size,
                                              pitch_predictor_kernel_size,
                                              dropout)
        self.pitch_embed = nn.Embedding(pitch_n_emb, 
                                        encoder_dim)
        self.pitch_boundaries = torch.linspace(min_pitch, 
                                               max_pitch,
                                               pitch_n_emb + 1).unsqueeze(0).unsqueeze(0)
        
        self.energy_predictor = EnergyPredictor(encoder_dim,
                                                energy_predictor_filter_size,
                                                energy_predictor_kernel_size,
                                                dropout)
        self.energy_embed = nn.Embedding(energy_n_emb, 
                                         encoder_dim)
        self.energy_boundaries = torch.linspace(min_energy, 
                                                max_energy,
                                                energy_n_emb + 1).unsqueeze(0).unsqueeze(0)
        self.decoder = Decoder(max_seq_len,
                               decoder_n_layer,
                               decoder_dim,
                               decoder_head,
                               pad_id,
                               decoder_conv1d_filter_size,
                               fft_conv1d_kernel,
                               fft_conv1d_padding,
                               dropout)

        self.mel_linear = nn.Linear(decoder_dim, num_mels)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, 
                src_seq, 
                src_pos, 
                mel_pos=None, 
                mel_max_length=None, 
                target_duration=None, 
                target_pitch=None,
                target_pitch_mean=None,
                target_pitch_std=None,
                target_energy=None,
                alpha=1.0,
                beta=1.0,
                gamma=1.0):
        self.pitch_boundaries = self.pitch_boundaries.to(src_seq.device)
        self.energy_boundaries = self.energy_boundaries.to(src_seq.device)
        
        enc_output, non_pad_mask = self.encoder(src_seq, src_pos)

        lr_output, duration_predictor_output, durations = self.length_regulator(enc_output, alpha, target_duration, mel_max_length)
        got_expansion = duration_predictor_output.sum(-1).cpu()
        new_enc_pos = list()
        max_new_enc_pos = lr_output.shape[1]
        for i in range(got_expansion.shape[0]):
            new_enc_pos.append(np.pad([i+1 for i in range(int(got_expansion[i].item()))],
                                      (0, max_new_enc_pos-int(got_expansion[i].item())), 'constant'))
        new_enc_pos = torch.from_numpy(np.array(new_enc_pos)).to(lr_output.device)

        pitch, mean, std = self.pitch_predictor(lr_output)
        if target_pitch is not None:          
            new_pitch = target_pitch * target_pitch_std.reshape(-1, 1) + \
                         target_pitch_mean.reshape(-1, 1)
        else:
            new_pitch = pitch * std.reshape(-1, 1) + \
                         mean.reshape(-1, 1)
        if beta != 1.0:
            new_pitch = new_pitch + torch.log(torch.ones_like(new_pitch)*beta)
        new_pitch = new_pitch.unsqueeze(-1).repeat(1, 1, self.pitch_boundaries.shape[-1])
        pitch_ids = (~(new_pitch < self.pitch_boundaries)).sum(-1) - 1
        pitch_ids = torch.clamp(pitch_ids, min=0, max=self.pitch_boundaries.shape[-1] - 2).long()
        pitch_output = self.pitch_embed(pitch_ids)
        pitch_output = self.mask_tensor(pitch_output, new_enc_pos, mel_max_length)

        energy = self.energy_predictor(lr_output)
        if target_energy is not None:
            new_energy = target_energy
        else:
            new_energy = energy
        if gamma != 1.0:
            new_energy = new_energy * gamma
        new_energy = new_energy.unsqueeze(-1).repeat(1, 1, self.energy_boundaries.shape[-1])
        energy_ids = (~(new_energy < self.energy_boundaries)).sum(-1) - 1
        energy_ids = torch.clamp(energy_ids, min=0, max=self.energy_boundaries.shape[-1] - 2).long()
        energy_output = self.energy_embed(energy_ids)
        energy_output = self.mask_tensor(energy_output, new_enc_pos, mel_max_length)

        output = lr_output + pitch_output + energy_output

        dec_output = self.mel_linear(self.decoder(output, new_enc_pos))
        if target_duration is not None: # training
            dec_output = self.mask_tensor(dec_output, mel_pos, mel_max_length)
        else:
            dec_output = self.mask_tensor(dec_output, new_enc_pos, mel_max_length)
        return dec_output, durations, pitch, mean, std, energy
    
    