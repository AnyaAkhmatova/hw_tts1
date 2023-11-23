import torch.nn as nn


class FastSpeech2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, 
                mel_predicted, 
                duration_predicted, 
                pitch_predicted, 
                pitch_mean_predicted, 
                pitch_std_predicted, 
                energy_predicted,
                mel, 
                duration,
                pitch, 
                pitch_mean, 
                pitch_std, 
                energy, 
                **kwargs):
        
        mel_loss = self.l1_loss(mel_predicted, mel)
        duration_loss = self.mse_loss(duration_predicted, duration.float())
        pitch_loss = self.mse_loss(pitch_predicted, pitch)
        pitch_mean_loss = self.mse_loss(pitch_mean_predicted, pitch_mean)
        pitch_std_loss = self.mse_loss(pitch_std_predicted, pitch_std)
        energy_loss = self.mse_loss(energy_predicted, energy)

        total_loss = mel_loss + duration_loss + pitch_loss + pitch_mean_loss + pitch_std_loss + energy_loss
        
        return total_loss, mel_loss, duration_loss, pitch_loss, pitch_mean_loss, pitch_std_loss, energy_loss
    
    