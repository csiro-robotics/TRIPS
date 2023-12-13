import torch
from torch import nn
import torch.nn.functional as F


def cross_entropy_w_temp_scaling(outputs, targets, exp=1.0, size_average=True, eps=1e-5):
    """
    Calculates cross-entropy with temperature scaling
    outputs: prediction output from target model
    target: prediction output from source model
    
    LwF method uses this function to calculate the distillation loss.
    """

    out = torch.nn.functional.softmax(outputs, dim=1)
    tar = torch.nn.functional.softmax(targets, dim=1)

    if exp != 1:
        out = out.pow(exp)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        tar = tar.pow(exp)
        tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
    out = out + eps / out.size(1)
    out = out / out.sum(1).view(-1, 1).expand_as(out)
    ce = -(tar * out.log()).sum(1)
    if size_average:
        ce = ce.mean()
    return ce


def binary_cross_entropy_w_sigmoid(outputs, targets):
    """
    Calculates cross-entropy with sigmoid 
    outputs: prediction output from target model
    target: prediction output from source model

    iCaRL method uses this function to calculate the distillation loss.
    """

    out = torch.sigmoid(outputs)
    tar = torch.sigmoid(targets)

    ce = sum(torch.nn.functional.binary_cross_entropy(out[:, y], tar[:, y]) for y in range(tar.size()[1]))
    return ce