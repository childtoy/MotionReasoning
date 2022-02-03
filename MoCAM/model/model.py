import torch
import torch.nn as nn
import numpy as np
from model.TCN import TemporalConvNet
from model.PURE1D import Pure1dNet
from model.PURE3D import Pure3dNet
import torch.nn.functional as F
from model.MLP import MlpNet
from model.MOT import MotionTransformer

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, inputs):
        y1 = self.tcn(inputs)
        output = self.linear(y1[:,:,-1])
        return F.log_softmax(output)



class PURE1D(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, dropout):
        super(PURE1D, self).__init__()
        self.net = Pure1dNet(input_size, output_size, kernel_size=kernel_size, dropout=dropout)
    def forward(self, inputs):
        output = self.net(inputs)

        return F.log_softmax(output)



class PURE3D(nn.Module):
    def __init__(self, input_size, n_channels, output_size, kernel_size, dropout):
        super(PURE3D, self).__init__()
        self.net = Pure3dNet(input_size, n_channels, output_size, kernel_size=kernel_size, dropout=dropout)
    def forward(self, inputs):
        output = self.net(inputs)

        return F.log_softmax(output)


class MLP(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super(MLP, self).__init__()
        self.net = MlpNet(input_size, output_size, dropout=dropout)
        # self.softmax = nn.Softmax(dim=1)
    def forward(self, inputs):
        output = self.net(inputs)
        return F.log_softmax(output)
    
        # return self.softmax(output)        

class MoT(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super(MoT, self).__init__()
        self.net = MotionTransformer(seq_len=60, in_chans=35, num_classes=7, embed_dim=35, depth=12,
                 num_heads=7, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm)                 
        # self.softmax = nn.Softmax(dim=1)
    def forward(self, inputs):
        output = self.net(inputs)
        return F.log_softmax(output)
    