import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class feature_mapping(nn.Module):
    def __init__(self, input_size, feature_dim, feature_bd, device):
        super().__init__()
        self.input_size = input_size
        self.feature_dim = feature_dim
        self.device = device
        self.centers = torch.randn(self.input_size, self.feature_dim).to(self.device) / feature_bd
        self.bias = torch.rand(self.feature_dim).to(self.device) * 2 * np.pi
        self.centers[:,0] *= 0
        self.bias[0] *= 0
    def forward(self, x):
        return torch.cos(torch.matmul(x,self.centers) + self.bias) / self.feature_dim ** 0.5

class linear_Q(nn.Module):
    def __init__(self, input_size, feature_dim, output_size, feature_mapping):
        super().__init__()
        self.feature_mapping = feature_mapping
        self.linear = nn.Linear(feature_dim, output_size, bias=False)
    def forward(self, x): # output Q(s,:)
        x = self.feature_mapping(x)
        x = self.linear(x)
        return x
    def log_prob(self,x):
        x = self.feature_mapping(x)
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x
    def init(self):
        nn.init.zeros_(self.linear.weight)

class linear_policy(nn.Module):
    def __init__(self, input_size, feature_dim, output_size, temperature, feature_mapping):
        super().__init__()
        self.feature_mapping = feature_mapping
        self.linear = nn.Linear(feature_dim, output_size, bias=False)
        # self.linear.weight.requires_grad = False
        self.temperature = temperature
    def forward(self, x, steps=1):
        x = self.feature_mapping(x)
        x = self.linear(x)
        x = F.softmax(x / (self.temperature * steps ** 0.5), dim=1)
        return x
    def policy_act(self, x, steps=1):
        x = self.feature_mapping(x)
        x = self.linear(x)
        x = F.softmax(x / (self.temperature * steps ** 0.5), dim=0)
        return x
    def log_prob(self, x):
        x = self.feature_mapping(x)
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x
    def init(self):
        nn.init.zeros_(self.linear.weight)