
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import torch

class Pure3dNet(nn.Module):
    def __init__(self, n_inputs, n_channels, n_outputs, kernel_size, dilation=1, padding='same', dropout=0.2):
        super(Pure3dNet, self).__init__()
        self.conv1_1 = weight_norm(nn.Conv2d(n_inputs, n_channels, 3, stride=1, padding=padding, dilation=dilation))
        self.conv1_2 = weight_norm(nn.Conv2d(n_channels, n_channels*2, 3, stride=1, padding=padding, dilation=dilation))
        self.conv1_3 = weight_norm(nn.Conv2d(n_channels*2, n_channels*2, 3, stride=2, padding='valid', dilation=dilation))
        self.relu = nn.ReLU()

        self.conv2_1 = weight_norm(nn.Conv2d(n_channels*2, n_channels*2, kernel_size, stride=1, padding=padding, dilation=dilation))
        self.conv2_2 = weight_norm(nn.Conv2d(n_channels*2, n_channels*2, kernel_size, stride=1, padding=padding, dilation=dilation))
        self.conv2_3 = weight_norm(nn.Conv2d(n_channels*2, n_channels*2, kernel_size, stride=2, padding='valid', dilation=dilation))

        self.conv3_1 = weight_norm(nn.Conv2d(n_channels*2, n_channels*4, kernel_size, stride=1, padding=padding, dilation=dilation))
        self.conv3_2 = weight_norm(nn.Conv2d(n_channels*4, n_channels*4, kernel_size, stride=1, padding=padding, dilation=dilation))
        self.conv3_3 = weight_norm(nn.Conv2d(n_channels*4, n_channels*4, kernel_size, stride=2, padding='valid', dilation=dilation))

        # self.conv4_1 = weight_norm(nn.Conv1d(n_inputs*2, n_inputs*2, kernel_size, stride=1, padding=padding, dilation=dilation))
        # self.conv4_2 = weight_norm(nn.Conv1d(n_inputs*2, n_inputs*1, kernel_size, stride=1, padding=padding, dilation=dilation))
        # self.conv4_3 = weight_norm(nn.Conv1d(n_inputs*1, n_inputs*1, kernel_size, stride=2, padding='valid', dilation=dilation))

        # self.conv5_1 = weight_norm(nn.Conv1d(n_inputs*1, n_inputs*1, kernel_size, stride=1, padding=padding, dilation=dilation))
        # self.conv5_2 = weight_norm(nn.Conv1d(int(n_inputs*1),int(n_inputs*0.5), kernel_size, stride=1, padding=padding, dilation=dilation))
        # self.conv5_3 = weight_norm(nn.Conv1d(int(n_inputs*0.5), int(n_inputs*0.5), kernel_size, stride=1, padding=padding, dilation=dilation))

        self.dropout = nn.Dropout(dropout)
        # self.avgPool = nn.AvgPool1d(7)
        self.linear = nn.Linear(n_channels*4, n_outputs)
        self.net1 = nn.Sequential(self.conv1_1, self.relu, self.conv1_2, self.relu, self.conv1_3, self.relu)
        self.net2 = nn.Sequential(self.conv2_1, self.relu, self.conv2_2, self.relu, self.conv2_3, self.relu)
        self.net3 = nn.Sequential(self.conv3_1, self.relu, self.conv3_2, self.relu, self.conv3_3, self.relu)
        # self.net4 = nn.Sequential(self.conv4_1, self.relu, self.conv4_2, self.relu, self.conv4_3, self.relu)                                  
        # self.net5 = nn.Sequential(self.conv5_1, self.relu, self.conv5_2, self.relu, self.conv5_3, self.relu)
                                    
        # self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
    #     self.init_weights()

    def init_weights(self):
        self.conv1_1.weight.data.normal_(0, 0.01)
        self.conv1_2.weight.data.normal_(0, 0.01)
        self.conv1_3.weight.data.normal_(0, 0.01)
        self.conv2_1.weight.data.normal_(0, 0.01)
        self.conv2_2.weight.data.normal_(0, 0.01)
        self.conv2_3.weight.data.normal_(0, 0.01)
        self.conv3_1.weight.data.normal_(0, 0.01)
        self.conv3_2.weight.data.normal_(0, 0.01)
        self.conv3_3.weight.data.normal_(0, 0.01)
        # self.conv4_1.weight.data.normal_(0, 0.01)
        # self.conv4_2.weight.data.normal_(0, 0.01)
        # self.conv4_3.weight.data.normal_(0, 0.01)
        # self.conv5_1.weight.data.normal_(0, 0.01)
        # self.conv5_2.weight.data.normal_(0, 0.01)
        # self.conv5_3.weight.data.normal_(0, 0.01)
        
   
    def forward(self, x):
        out = self.net1(x)
        out = self.net2(out)
        out = self.net3(out)
        # out = self.net4(out)
        # out = self.net5(out)
        out = self.dropout(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        output = self.linear(out)
        return output
