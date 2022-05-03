
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import torch
class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True, **kwargs) -> None:
        super(Conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs) # kernel size = ...
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        if not self.activation:
            return self.batchnorm(self.conv(x))
        return self.relu(self.batchnorm(self.conv(x))
class Res_block(nn.Module):
    def __init__(self, in_channels, red_channels, out_channels, is_plain=False):
        super(Res_block,self).__init__()
        self.relu = nn.ReLU()
        self.is_plain = is_plain
        
        if in_channels==64:
            self.convseq = nn.Sequential(
                                    Conv_block(in_channels, red_channels, kernel_size=1, padding=0),
                                    Conv_block(red_channels, red_channels, kernel_size=3, padding=1),
                                    Conv_block(red_channels, out_channels, activation=False, kernel_size=1, padding=0)
            )
            self.iden = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        elif in_channels == out_channels:
            self.convseq = nn.Sequential(
                                    Conv_block(in_channels, red_channels, kernel_size=1, padding=0),
                                    Conv_block(red_channels, red_channels, kernel_size=3, padding=1),
                                    Conv_block(red_channels, out_channels, activation=False, kernel_size=1, padding=0)
            )
            self.iden = nn.Identity()
        else:
            self.convseq = nn.Sequential(
                                    Conv_block(in_channels, red_channels, kernel_size=1, padding=0, stride=2),
                                    Conv_block(red_channels, red_channels, kernel_size=3, padding=1),
                                    Conv_block(red_channels, out_channels, activation=False, kernel_size=1, padding=0)
                
            )
            self.iden = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        
    def forward(self, x):
        y = self.convseq(x)
        if self.is_plain:
            x = y
        else:
            x = y + self.iden(x)
        x = self.relu(x)  # relu(skip connection)
        return x

class ResNet(nn.Module):
    def __init__(self, in_channels=3 , num_classes=1000, is_plain=False):
        self.num_classes = num_classes
        super(ResNet, self).__init__()
        self.conv1 = Conv_block(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv2_x = nn.Sequential(
                                        Res_block(64, 64, 256, is_plain),
                                        Res_block(256, 64, 256, is_plain),
                                        Res_block(256, 64, 256, is_plain)
        )
        
        self.conv3_x = nn.Sequential(
                                        Res_block(256, 128, 512, is_plain),
                                        Res_block(512, 128, 512, is_plain),
                                        Res_block(512, 128, 512, is_plain),
                                        Res_block(512, 128, 512, is_plain)
        )

        self.conv4_x = nn.Sequential(
                                        Res_block(512, 256, 1024, is_plain),
                                        Res_block(1024, 256, 1024, is_plain),
                                        Res_block(1024, 256, 1024, is_plain),
                                        Res_block(1024, 256, 1024, is_plain),
                                        Res_block(1024, 256, 1024, is_plain),
                                        Res_block(1024, 256, 1024, is_plain)
        )
        
        self.conv5_x = nn.Sequential(
                                        Res_block(1024, 512, 2048, is_plain),
                                        Res_block(2048, 512, 2048, is_plain),
                                        Res_block(2048, 512, 2048, is_plain),
        )

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(2048,num_classes)

    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = F.adaptive_avg_pool2d(out, (1, 1))
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x        