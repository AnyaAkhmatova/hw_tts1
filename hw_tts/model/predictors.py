import torch
from torch import nn
import torch.nn.functional as F


def create_alignment(base_mat, duration_predictor_output):
    N, L = duration_predictor_output.shape
    for i in range(N):
        count = 0
        for j in range(L):
            for k in range(duration_predictor_output[i][j]):
                base_mat[i][count+k][j] = 1
            count = count + duration_predictor_output[i][j]
    return base_mat


class Transpose(nn.Module):
    def __init__(self, dim_1, dim_2):
        super().__init__()
        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def forward(self, x):
        return x.transpose(self.dim_1, self.dim_2)


class DurationPredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self, 
                 encoder_dim,
                 duration_predictor_filter_size,
                 duration_predictor_kernel_size,
                 dropout):
        super(DurationPredictor, self).__init__()

        self.input_size = encoder_dim
        self.filter_size = duration_predictor_filter_size
        self.kernel = duration_predictor_kernel_size
        self.conv_output_size = duration_predictor_filter_size
        self.dropout = dropout

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.ReLU(),
            nn.LayerNorm(self.filter_size),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.ReLU(),
            nn.LayerNorm(self.filter_size),
            nn.Dropout(self.dropout)
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)

        out = self.linear_layer(encoder_output)
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self, 
                 encoder_dim,
                 duration_predictor_filter_size,
                 duration_predictor_kernel_size,
                 dropout):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = DurationPredictor(encoder_dim,
                                                    duration_predictor_filter_size,
                                                    duration_predictor_kernel_size,
                                                    dropout)

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(
            torch.sum(duration_predictor_output, -1), -1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length-output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        durations = torch.exp(self.duration_predictor(x)) * alpha
        if target is not None:   
            if target.shape[1] != durations.shape[1]:
                new_target = F.pad(target, (0, durations.shape[-1] - target.shape[-1])) * alpha
            else:
                new_target = target * alpha
            duration_predictor_output = torch.round(new_target).int()
        else:
            duration_predictor_output = torch.round(durations).int()
        output = self.LR(x, duration_predictor_output, mel_max_length)
        return output, duration_predictor_output, durations
    

class PitchPredictor(nn.Module):

    def __init__(self, 
                 encoder_dim,
                 pitch_predictor_filter_size,
                 pitch_predictor_kernel_size,
                 dropout):
        super(PitchPredictor, self).__init__()

        self.input_size = encoder_dim
        self.filter_size = pitch_predictor_filter_size
        self.kernel = pitch_predictor_kernel_size
        self.conv_output_size = pitch_predictor_filter_size
        self.dropout = dropout

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.ReLU(),
            nn.LayerNorm(self.filter_size),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.ReLU(),
            nn.LayerNorm(self.filter_size),
            nn.Dropout(self.dropout)
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)
        self.mean_pred = nn.Linear(self.conv_output_size, 1)
        self.std_pred = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)

        out = self.linear_layer(encoder_output)
        mean = self.mean_pred(encoder_output.mean(1))
        std = self.std_pred(encoder_output.mean(1))
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out, mean, std


class EnergyPredictor(nn.Module):

    def __init__(self, 
                 encoder_dim,
                 energy_predictor_filter_size,
                 energy_predictor_kernel_size,
                 dropout):
        super(EnergyPredictor, self).__init__()

        self.input_size = encoder_dim
        self.filter_size = energy_predictor_filter_size
        self.kernel = energy_predictor_kernel_size
        self.conv_output_size = energy_predictor_filter_size
        self.dropout = dropout

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.ReLU(),
            nn.LayerNorm(self.filter_size),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.ReLU(),
            nn.LayerNorm(self.filter_size),
            nn.Dropout(self.dropout)
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)

        out = self.linear_layer(encoder_output)
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out
    
