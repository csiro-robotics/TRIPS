import copy
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import os
import json
from torch.nn import Parameter
import math
import random

#  import higher

from domainbed import networks
from domainbed.lib.misc import random_pairs_of_minibatches
from domainbed.optimizers import get_optimizer

from domainbed.models.resnet_mixstyle import (
    resnet18_mixstyle_L234_p0d5_a0d1,
    resnet34_mixstyle_L234_p0d5_a0d1,
    resnet50_mixstyle_L234_p0d5_a0d1,
)
from domainbed.models.resnet_mixstyle2 import (
    resnet18_mixstyle2_L234_p0d5_a0d1,
    resnet34_mixstyle2_L234_p0d5_a0d1,
    resnet50_mixstyle2_L234_p0d5_a0d1,
)

from domainbed.algorithms.algorithms import Algorithm


class Mixstyle(Algorithm):
    """MixStyle w/o domain label (random shuffle)"""

    def __init__(self, input_shape, num_classes, num_domains, hparams, current_session):
        assert input_shape[1:3] == (224, 224), "Mixstyle support R18 and R50 only"
        super().__init__(input_shape, num_classes, num_domains, hparams, current_session)
        if hparams["resnet18"]:
            # network = resnet18_mixstyle_L234_p0d5_a0d1()
            network = resnet34_mixstyle2_L234_p0d5_a0d1()
        else:
            network = resnet50_mixstyle_L234_p0d5_a0d1()
        self.featurizer = networks.ResNet(input_shape, self.hparams, network)

        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = self.new_optimizer(self.network.parameters())

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)

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


class Mixstyle2(Algorithm):
    """MixStyle w/ domain label"""

    def __init__(self, input_shape, num_classes, num_domains, hparams, current_session):
        assert input_shape[1:3] == (224, 224)
        super().__init__(input_shape, num_classes, num_domains, hparams, current_session)
        self.name = 'Mixstyle2'
        if hparams["resnet18"]:
            network = resnet18_mixstyle2_L234_p0d5_a0d1()
        elif hparams["resnet34"]:
            network = resnet34_mixstyle2_L234_p0d5_a0d1()
        else:
            network = resnet50_mixstyle2_L234_p0d5_a0d1()
        self.featurizer = networks.ResNet(input_shape, self.hparams, network)

        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = self.new_optimizer(self.network.parameters())

    def pair_batches(self, xs, ys):
        xs = [x.chunk(2) for x in xs]
        ys = [y.chunk(2) for y in ys]
        N = len(xs)
        pairs = []
        for i in range(N):
            j = i + 1 if i < (N - 1) else 0
            xi, yi = xs[i][0], ys[i][0]
            xj, yj = xs[j][1], ys[j][1]

            pairs.append(((xi, yi), (xj, yj)))

        return pairs

    def update(self, x, y, **kwargs):
        pairs = self.pair_batches(x, y)
        loss = 0.0

        for (xi, yi), (xj, yj) in pairs:
            #  Mixstyle2:
            #  For the input x, the first half comes from one domain,
            #  while the second half comes from the other domain.
            x2 = torch.cat([xi, xj])
            y2 = torch.cat([yi, yj])
            loss += F.cross_entropy(self.predict(x2), y2)

        loss /= len(pairs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)

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

