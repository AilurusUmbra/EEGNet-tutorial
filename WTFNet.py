import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torch.cuda import device
from torch.utils.data import DataLoader


class SE(nn.Module):
    def __init__(self, n_features=32, reduction=4):
        super(SE, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 16)')

        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):

        y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        y = x * y
        return y
"""
    def __init__(self, nout=64, reduce_rate=16):
        super(SE, self).__init__()
        self.gp = nn.AvgPool2d(1)
        self.rb1 = WTFNet()
        self.se = nn.Sequential(nn.Linear(nout, nout // reduce_rate),
                                nn.ReLU(inplace=True),
                                nn.Linear(nout // reduce_rate, nout),
                                nn.Sigmoid())
    def forward(self, input):
        x = input
        x = self.rb1(x)

        #print(x.shape)
        #print("gp?", self.gp(x).shape)
        b, c, _, _ = x.size()
        y = self.gp(x)
        y = self.se(y).view(b, c, 1, 1)
        y = x * y.expand_as(x)
        out = y + input
        return out
"""

class WTFNet(nn.Module):
    def __init__(self, activation=nn.ELU, sq=False):
        super(WTFNet, self).__init__()

        self.sq = sq

        self.firstConv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.2)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.2)
        )

        self.classify = nn.Sequential(
            nn.Linear(in_features=736, out_features=2, bias=True)
        )

        self.seblock = SE(n_features=32)

    def forward(self, x):
        
        x = self.firstConv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
       
        if self.sq:
            # SE Block
            x = self.seblock(x)
            
            #x = torch.add(x, y)

        x = x.view(-1, self.classify[0].in_features)
        x = self.classify(x)

        return x 

