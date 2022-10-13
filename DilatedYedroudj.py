import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
# 30 SRM filtes
from .srm_filter_kernel import all_normalized_hpf_list
# Global covariance pooling
from .MNCOV import *  # MPNCOV
from torch.optim import SGD

def initWeights(module):
    if type(module) == nn.Conv2d:
        if module.weight.requires_grad:
            nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')

    if type(module) == nn.Linear:
        nn.init.normal_(module.weight.data, mean=0, std=0.01)
        nn.init.constant_(module.bias.data, val=0)

class TLU(nn.Module):
    def __init__(self, threshold):
        super(TLU, self).__init__()

        self.threshold = threshold

    def forward(self, input):
        output = torch.clamp(input, min=-self.threshold, max=self.threshold)

        return output


# https://gist.github.com/erogol/a324cc054a3cdc30e278461da9f1a05e
class SPPLayer(nn.Module):
    def __init__(self, num_levels):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels

    def forward(self, x):
        bs, c, h, w = x.size()
        pooling_layers = []
        for i in range(self.num_levels):
            kernel_size = h // (2 ** i)

            tensor = F.avg_pool2d(x, kernel_size=kernel_size,
                                  stride=kernel_size).view(bs, -1)
            pooling_layers.append(tensor)
        x = torch.cat(pooling_layers, dim=-1)
        return x


# absult value operation
class ABS(nn.Module):
    def __init__(self):
        super(ABS, self).__init__()

    def forward(self, input):
        output = torch.abs(input)
        return output


# add operation
class ADD(nn.Module):
    def __init__(self):
        super(ADD, self).__init__()

    def forward(self, input1, input2):
        output = torch.add(input1, input2)
        return output


# Pre-processing Module
class HPF(nn.Module):
    def __init__(self):
        super(HPF, self).__init__()

        # Load 30 SRM Filters
        all_hpf_list_5x5 = []

        for hpf_item in all_normalized_hpf_list:
            if hpf_item.shape[0] == 3:
                hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')

            all_hpf_list_5x5.append(hpf_item)
        all_hpf_list_5x5 = np.array(all_hpf_list_5x5)

        hpf_weight = nn.Parameter(torch.Tensor(all_hpf_list_5x5).view(30, 1, 5, 5), requires_grad=False)

        self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = hpf_weight

        # Truncation, threshold = 3
        self.tlu = TLU(3.0)

    def forward(self, input):

        output = self.hpf(input)
        output = self.tlu(output)

        return output

class A_Trous(torch.nn.Module):
    def __init__(self):
        super(A_Trous, self).__init__()
        self.conv_1 = nn.Conv2d(30, 10, 5, padding='same')
        self.conv_2 = nn.Conv2d(30, 10, 5, padding='same', dilation=2)
        self.conv_3 = nn.Conv2d(30, 10, 5, padding='same', dilation=4)

    def forward(self, input):
        x = self.conv_1(input)
        y = self.conv_2(input)
        z = self.conv_3(input)
        return torch.cat((x, y, z), dim=1)


class YedroudjNet(torch.nn.Module):
    def __init__(self):
        super(YedroudjNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name="Dilated-YedroudjNet"
        self.lr = 5e-3
        self.weight_decay = 5e-4
        self.group1 = HPF()  # pre-processing Layer 1
        # self.conv0.weight = torch.nn.Parameter(srm)

        self.conv1 = A_Trous()
        self.abs = ABS()
        self.bn1 = nn.BatchNorm2d(30)
        # Trunc T= 3
        self.tlu3 = TLU(3.0)
        self.conv2 = A_Trous()

        self.bn2 = nn.BatchNorm2d(30)
        # Trunc T = 1
        self.tlu1 = TLU(1.0)
        self.pool = torch.nn.AvgPool2d(kernel_size=5, stride=2,
                                       padding=2)  # the same pool layer well be used to L3 and L4
        self.conv3 = torch.nn.Conv2d(in_channels=30, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # Layer 4
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  # Layer 5
        self.bn5 = nn.BatchNorm2d(128)

        self.conv6 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # Layer 6
        self.bn6 = nn.BatchNorm2d(64)

        self.conv7 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  # Layer 7
        self.bn7 = nn.BatchNorm2d(128)

        # self.spp_layer = SPPLayer(spp_level) # spp_level = 1 Global averge pooling

        self.fc1 = torch.nn.Linear(128, 256)
        self.fc2 = torch.nn.Linear(256, 1024)
        self.fc3 = torch.nn.Linear(1024, 2)

        initWeights(self.conv1)
        initWeights(self.conv2)
        initWeights(self.conv3)
        initWeights(self.conv4)
        initWeights(self.conv5)
        initWeights(self.conv6)
        initWeights(self.conv7)
        initWeights(self.fc1)
        initWeights(self.fc2)
        initWeights(self.fc3)

    def forward(self, x):
        x = self.group1(x)
        x = self.conv1(x)
        x = self.abs(x)
        # x =  F.relu(x)
        x = self.bn1(x)
        x = self.tlu3(x)

        x = self.bn2(self.conv2(x))
        x = self.tlu1(x)
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        # print("conv3_2",x.shape)
        # x = torch.add(res ,x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        x = F.relu(self.bn5(self.conv5(x)))
        # x = self.spp_layer(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        # print("x",x.shape)
        # print("x",x.view(-1, 256).shape)
        x = x.view(-1, 128)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return (x)

    def reset_parameters(self):
        pass

    def get_optimizer(self):
        return SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)

    def get_loss(self, inputs, labels):
        loss_f = nn.CrossEntropyLoss().to(self.device)
        outputs = self(inputs.float())
        return outputs, loss_f(outputs, labels.float())
