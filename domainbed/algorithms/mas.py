import copy
from copy import deepcopy
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
from domainbed.lib.fast_data_loader import FastDataLoader

#  import higher

from domainbed import networks
from domainbed.lib.misc import random_pairs_of_minibatches
from domainbed.optimizers import get_optimizer

from domainbed.models.resnet_mixstyle import (
    resnet18_mixstyle_L234_p0d5_a0d1,
    resnet50_mixstyle_L234_p0d5_a0d1,
)
from domainbed.models.resnet_mixstyle2 import (
    resnet18_mixstyle2_L234_p0d5_a0d1,
    resnet50_mixstyle2_L234_p0d5_a0d1,
)

from domainbed.algorithms.algorithms import Algorithm
from domainbed.losses.common_loss_func import *


def compute_mas_importance(model, eval_meta, test_envs):
    precision_matrix_dict = {}
    final_precision_matrix = {}
    num_of_used_dataset = 0
    
    print('--- compute_mas_importance ---')
    for name, loader_kwargs, weights in eval_meta:
        env_name, inout = name.split("_")
        env_num = int(env_name[3:])
        print('env_name: {0}, inout: {1}, env_num: {2}'.format(env_name, inout, env_num))

        if inout == "out": 
            print('skip...')
            continue
        if env_num in test_envs and inout == "in": 
            print('skip...')
            continue

        if isinstance(loader_kwargs, dict):
            loader = FastDataLoader(**loader_kwargs)
        elif isinstance(loader_kwargs, FastDataLoader):
            loader = loader_kwargs
        else:
            raise ValueError(loader_kwargs)
        
        precision_matrix = mas_importance_matrix(model, loader)
        precision_matrix_dict["{0}_{1}".format(inout, env_num)] = precision_matrix
        num_of_used_dataset = num_of_used_dataset + 1 

    precision_matrice_index = 0
    for key_1, value_1 in precision_matrix_dict.items():
        for key_2, value_2 in precision_matrix_dict[key_1].items():
            if precision_matrice_index == 0:
                final_precision_matrix[key_2] = precision_matrix_dict[key_1][key_2]
            else:
                final_precision_matrix[key_2] = final_precision_matrix[key_2] + precision_matrix_dict[key_1][key_2]
        precision_matrice_index = precision_matrice_index + 1

    for key, value in final_precision_matrix.items():
        final_precision_matrix[key] = final_precision_matrix[key] / num_of_used_dataset
    
    return final_precision_matrix
        

def mas_importance_matrix(model, data_loader):
    # max_importance = 0
    # min_importance = 100
    precision_matrices = {}

    model.train()
    total_num_imgs = 0
    for index, batch in enumerate(data_loader, 0):
        images = batch["x"].to("cuda")
        targets = batch["y"].to("cuda")
        img_ids = batch["img_id"]

        model.zero_grad()  # set the model gradient to zero
        # MAS allows any unlabeled data to do the estimation, we choose the current data as in main experiments
        # Page 6: labels not required, "...use the gradients of the squared L2-norm of the learned function output."
        loss = torch.norm(model(images), p=2, dim=1).mean()
        loss.backward()    
        
        num_of_imgs = len(targets)
        total_num_imgs = total_num_imgs + num_of_imgs

        if index == 0:
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    if parameter.grad is not None:
                        precision_matrices[name] = ((parameter.grad.data ** 2)) * num_of_imgs
        else:
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    if parameter.grad is not None:
                        precision_matrices[name] += ((parameter.grad.data ** 2)) * num_of_imgs
    print('total_num_imgs: {0}'.format(total_num_imgs))
    for key, value in precision_matrices.items():
        precision_matrices[key] = precision_matrices[key] / total_num_imgs

    return precision_matrices


class MAS(Algorithm):
    """
    MAS
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, current_session, old_parameters, precision_matrix, hparam_lambda=10000):
        super(MAS, self).__init__(input_shape, num_classes, num_domains, hparams, current_session)
        # extra parameter: hparam_lambda
        self.name = 'MAS'
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        self.current_session = current_session
        if hparam_lambda < 1:
            raise RuntimeError('Something is wrong with the hyperparameter lambda.')

        if current_session > 0:  # incremental session
            self.hparam_lambda = hparam_lambda
            self.old_parameters = old_parameters
            for name, value in self.old_parameters.items():
                old_parameters[name] = old_parameters[name].to("cuda")
            self.precision_matrix = precision_matrix

    def mas_penalty(self, model, old_parameters, precision_matrix):
        mas_loss = 0

        for name, parameter in model.named_parameters():
            if name in precision_matrix:
                old_parameter_name = "network.{0}".format(name)
                if name == "1.weight":  # weight parameters for FC layer
                    parameter = parameter[precision_matrix[name].size()[0], :]
                if name == "1.bias":
                    parameter = parameter[precision_matrix[name].size()[0]]
                buff_loss = precision_matrix[name] * ((parameter - old_parameters[old_parameter_name]) ** 2)
                mas_loss += buff_loss.sum()
        return mas_loss

    def update(self, x, y, **kwargs): 
        if self.current_session == 0: # base session
            loss_dict = self.update_base(x, y, **kwargs)
        else: # incremental session
            loss_dict = self.update_incremental(x, y, **kwargs)
        return loss_dict

    def update_base(self, x, y, **kwargs): 
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def update_incremental(self, x, y, **kwargs):
        loss_dict = {}
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        cross_entropy_loss = F.cross_entropy(self.predict(all_x), all_y)
        mas_loss = self.mas_penalty(self.network, self.old_parameters, self.precision_matrix)
        total_loss = cross_entropy_loss + self.hparam_lambda * mas_loss
        # print('mas_loss: {0} * {1}, cross_entropy_loss: {2}, total_loss: {3}'.format(self.hparam_lambda, mas_loss, cross_entropy_loss, total_loss))
            
        loss_dict["cross_entropy_loss"] = cross_entropy_loss.item()
        loss_dict["mas_loss"] = mas_loss.item()
        loss_dict["loss"] = total_loss.item()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return loss_dict

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
