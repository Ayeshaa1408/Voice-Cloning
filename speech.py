import torch
import torch.nn as nn

class SpeechSynthesizer(nn.Module):
    def __init__(self):
        super(SpeechSynthesizer, self).__init__()
        self.synthesizer = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(512, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, 80, kernel_size=5, stride=1, padding=2),
        )
    
    def forward(self, encoded_voice):
        return self.synthesizer(encoded_voice)
