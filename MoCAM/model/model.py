import torch
import torch.nn as nn
import numpy as np
from model.TCN import TemporalConvNet
import torch.nn.functional as F

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, inputs):
        y1 = self.tcn(inputs)
        o = self.linear(y1[:,:,-1])
        return self.softmax(o)

