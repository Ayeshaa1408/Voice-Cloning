import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram

class VoiceEncoder(nn.Module):
    def __init__(self):
        super(VoiceEncoder, self).__init__()
        self.melspectrogram = MelSpectrogram()
        self.encoder = nn.Sequential(
            nn.Conv1d(80, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(512, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )
    
    def forward(self, waveform):
        mel_spec = self.melspectrogram(waveform)
        encoded = self.encoder(mel_spec)
        return encoded.mean(dim=2)
