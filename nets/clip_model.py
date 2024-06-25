import clip
import torch.nn as nn


class ClipModel(nn.Module):
    def __init__(self, bit=128):
        super(ClipModel, self).__init__()
        self.model = clip.load('RN50')
        self.linear = nn.Linear(1024, bit)

    def forward(self, x):
        out = self.model(x)
        out = self.linear(out).tanh()

        return out