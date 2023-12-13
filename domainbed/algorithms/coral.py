import copy
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import os
import json

#  import higher

from domainbed import networks
from domainbed.lib.misc import random_pairs_of_minibatches
from domainbed.optimizers import get_optimizer
from domainbed.algorithms.algorithms import Algorithm
from domainbed.algorithms.erm import ERM

def to_minibatch(x, y):
    minibatches = list(zip(x, y))
    return minibatches


class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, current_session, num_of_exemplar, gaussian):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains, hparams, current_session, num_of_exemplar)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(
            x1_norm
        )
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=(0.001, 0.01, 0.1, 1, 10, 100, 1000)):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= nmb * (nmb - 1) / 2

        self.optimizer.zero_grad()
        (objective + (self.hparams["mmd_gamma"] * penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {"loss": objective.item(), "penalty": penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, current_session, num_of_exemplar):
        super(MMD, self).__init__(input_shape, num_classes, num_domains, hparams, current_session, num_of_exemplar, gaussian=True)


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, current_session, num_of_exemplar):
        super(CORAL, self).__init__(input_shape, num_classes, num_domains, hparams, current_session, num_of_exemplar, gaussian=False)
        self.name = "CORAL"

    def load_previous_model_param(self, dir, test_envs, type):
        if type == 'last_step':
            old_model_path = os.path.join(dir, "TE{0}_last_step.pth".format(test_envs[0]))
        elif type == 'iid':
            old_model_path = os.path.join(dir, "TE{0}_best_iid.pth".format(test_envs[0]))
        elif type == 'oracle':
            old_model_path = os.path.join(dir, "TE{0}_best_oracle.pth".format(test_envs[0]))
        else:
            raise ValueError("Something wrong with the model type.")

        print('------- old_model_path: {0}'.format(old_model_path))
        old_model_dict = torch.load(old_model_path)
        old_model_param_dict = old_model_dict['model_dict']
        """
        for k, v in old_model_param_dict.items():
            print('old_model_param_dict | key: {0}, value: {1}'.format(k, v.size()))
        """
        network_dict = self.network.state_dict()
        for k, v in network_dict.items():
            # print('network_dict | key: {0}, value: {1}'.format(k, v.size()))
            if k == '1.weight': # classifier - weight
                num_old_cls = old_model_param_dict['network.{0}'.format(k)].size()[0]
                num_total_cls = network_dict[k].size()[0]
                # print('num_old_cls: {0}, num_total_cls: {1}'.format(num_old_cls, num_total_cls))
                network_dict[k][:num_old_cls,:] = old_model_param_dict['network.{0}'.format(k)]
            elif k == '1.bias': # classifier - bias
                num_old_cls = old_model_param_dict['network.{0}'.format(k)].size()[0]
                num_total_cls = network_dict[k].size()[0]
                # print('num_old_cls: {0}, num_total_cls: {1}'.format(num_old_cls, num_total_cls))
                network_dict[k][:num_old_cls] = old_model_param_dict['network.{0}'.format(k)]
            else:
                network_dict[k] = old_model_param_dict['network.{0}'.format(k)]
        self.network.load_state_dict(network_dict)

    def encode(self, x):  # for target model obtain features
        return self.featurizer(x)


