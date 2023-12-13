import copy
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import random

class Mahalanobis(nn.Module):
    def __init__(self, in_features, num, r=64):
        super(Mahalanobis, self).__init__()
        self.in_features = in_features
        self.out_features = num
        self.classnum = num
        self.margin = 0.1
        self.weights = []
        self.biases = []
        self.r = r

        self.bias = Parameter(torch.Tensor(num, in_features))
        self.weight = Parameter(torch.Tensor(num, r, in_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(0))
        nn.init.constant_(self.bias, 0.0)

    def set_parameter(self, weights, biases):
        self.weight = Parameter(torch.cat(weights, dim=0))
        self.bias = Parameter(torch.cat(biases, dim=0))
        self.classnum = self.weight.shape[0]

    @staticmethod
    def mahala_concate(Mahala1, Mahala2):
        output = Mahalanobis(Mahala1.in_features, Mahala1.out_features, r=Mahala1.r).cuda()
        weights = [Mahala1.weight, Mahala2.weight]
        biases = [Mahala1.bias, Mahala2.bias]
        output.set_parameter(weights, biases)
        return output

    def add_para(self, num):
        self.weight.data.expand()

    def forward(self, x):
        """
        shape of x is BxN, B is the batch size
        """
        B, N = x.shape
        # x = x.unsqueeze(1).expand(B, self.classnum, N)
        s_all = []
        # for i in range(self.classnum):
        #     f = torch.matmul(self.weight[i], (x - self.bias[i]).t())
        #     s = torch.square(torch.norm(f.t(), dim=1, p=2).squeeze().contiguous().view(B,1))
        #     s_all.append(s)
        # out = torch.cat(s_all,dim=1)
        h = x.unsqueeze(1).expand(B, self.classnum, N) - self.bias
        expanded_weight = self.weight.unsqueeze(0).expand(B, self.classnum, self.r, N)
        h = h.view(B*self.classnum, N).unsqueeze(2)
        expanded_weight = expanded_weight.reshape(B*self.classnum, self.r, N)

        s_r = torch.matmul(expanded_weight, h).squeeze()
        s = torch.square(torch.norm(s_r, dim=1, p=2))
        out = s.view(B, self.classnum)

        return out
