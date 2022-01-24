
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class Model(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation=1, padding='same', dropout=0.2):
        super(Model, self).__init__()
        print('in',n_inputs)
        print('out',n_outputs)
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        print(11111)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        print(222222)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                    self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None : 
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i 
            in_channels = num_inputs if i==0 else num_channels[i-1]
            out_channels = num_channels[i]
            print('inin',in_channels)
            print('out', out_channels)
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)
    

class PURE1D(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(PURE1D, self).__init__()
        self.net = Model(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
    def forward(self, inputs):
        y1 = self.tcn(inputs)
        print('y1 shape', y1.shape)
        o = self.linear(y1[:,:,-1])
        print('o shape', o.shape)
        return F.log_softmax(0, dim=1)