import numpy as np
import struct as st
import torch
import torch.nn as nn
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def norm_angle(angle):
    norm_angle = sigmoid(10 * (abs(angle) / 45.0 - 1))
    return norm_angle

class Branch(nn.Module):
    def __init__(self, feat_dim):
        super(Branch, self).__init__()
        self.fc1 = nn.Linear(feat_dim, feat_dim)
        self.fc2 = nn.Linear(feat_dim, feat_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, yaw):
        x = self.fc1(input)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)

        yaw = yaw.view(yaw.size(0),1)
        yaw = yaw.expand_as(x)

        feature = yaw * x + input
        return feature
