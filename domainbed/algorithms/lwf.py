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

from domainbed import networks
from domainbed.lib.misc import random_pairs_of_minibatches
from domainbed.optimizers import get_optimizer

from domainbed.algorithms.algorithms import Algorithm
from domainbed.losses.common_loss_func import *


class ERM_DIST(Algorithm):
    """
    LwF: Empirical Risk Minimization (ERM) + Knowledge Distillation (DIST)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, current_session, num_of_exemplar, num_old_cls, temperature=1):
        super(ERM_DIST, self).__init__(input_shape, num_classes, num_domains, hparams, current_session)
        self.name = 'ERM_DIST'
        self.num_of_exemplar = num_of_exemplar
        self.num_old_cls = num_old_cls
        self.current_session = current_session
        self.lambda_c = hparams["lambda_c"]
        self.lambda_d = hparams["lambda_d"]
        print('ERM_DIST | lambda_c: {0}, lambda_d: {1}'.format(self.lambda_c, self.lambda_d))

        if current_session == 0:  # base session 
            self.featurizer = networks.Featurizer(input_shape, self.hparams)
            self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
            self.network = nn.Sequential(self.featurizer, self.classifier)

            self.optimizer = get_optimizer(
            hparams["optimizer"],
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
            )
        else:  # incremental sessions
            self.target_featurizer = networks.Featurizer(input_shape, self.hparams)
            self.target_classifier = nn.Linear(self.target_featurizer.n_outputs, num_classes)
            self.target_network = nn.Sequential(self.target_featurizer, self.target_classifier)
            self.source_featurizer = networks.Featurizer(input_shape, self.hparams)
            self.source_classifier = nn.Linear(self.source_featurizer.n_outputs, num_old_cls)
            self.source_network = nn.Sequential(self.source_featurizer, self.source_classifier)
            
            for name, param in self.source_network.named_parameters():
                param.requires_grad = False
            self.source_network.eval()

            self.optimizer = get_optimizer(
                hparams["optimizer"],
                self.target_network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

            self.temperature = temperature # temperature for distillation 


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
        
        if self.current_session > 1:
            network_name = 'target_network'
        else:
            network_name = 'network'
        target_network_dict = self.target_network.state_dict()
        for k, v in target_network_dict.items():
            if k == '1.weight': # classifier - weight
                num_old_cls = old_model_param_dict['{0}.{1}'.format(network_name, k)].size()[0]
                num_total_cls = target_network_dict[k].size()[0]
                target_network_dict[k][:num_old_cls,:] = old_model_param_dict['{0}.{1}'.format(network_name, k)]
            elif k == '1.bias': # classifier - bias
                num_old_cls = old_model_param_dict['{0}.{1}'.format(network_name, k)].size()[0]
                num_total_cls = target_network_dict[k].size()[0]
                target_network_dict[k][:num_old_cls] = old_model_param_dict['{0}.{1}'.format(network_name, k)]
            else:
                target_network_dict[k] = old_model_param_dict['{0}.{1}'.format(network_name, k)]
        self.target_network.load_state_dict(target_network_dict)
        
        source_network_dict = self.source_network.state_dict()
        for k, v in source_network_dict.items():
            if k == '1.weight': # classifier - weight
                num_old_cls = old_model_param_dict['{0}.{1}'.format(network_name, k)].size()[0]
                num_total_cls = source_network_dict[k].size()[0]
                source_network_dict[k][:num_old_cls,:] = old_model_param_dict['{0}.{1}'.format(network_name, k)]
            elif k == '1.bias': # classifier - bias
                num_old_cls = old_model_param_dict['{0}.{1}'.format(network_name, k)].size()[0]
                num_total_cls = source_network_dict[k].size()[0]
                source_network_dict[k][:num_old_cls] = old_model_param_dict['{0}.{1}'.format(network_name, k)]
            else:
                source_network_dict[k] = old_model_param_dict['{0}.{1}'.format(network_name, k)]
        self.source_network.load_state_dict(source_network_dict)

    def update(self, x, y, **kwargs): 
        if self.current_session == 0: # base session
            loss_dict = self.update_base(x, y, **kwargs)
        else: # incremental session
            loss_dict = self.update_incremental(x, y, **kwargs)
        return loss_dict

    def update_base(self, x, y, **kwargs): 
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        all_feature = self.featurizer(all_x)
        all_output = self.classifier(all_feature)
        loss = F.cross_entropy(all_output, all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}
         
    def update_incremental(self, x, y, **kwargs):  # back-propagation to update backbone
        loss_dict = {}
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        target_feature = self.target_featurizer(all_x)
        source_feature = self.source_featurizer(all_x)
        target_output = self.target_classifier(target_feature)
        source_output = self.source_classifier(source_feature)
        target_output_old_cls = target_output[:, :source_output.size()[1]]
        target_output_new_cls = target_output[:, source_output.size()[1]:]

        cross_entropy_loss = F.cross_entropy(target_output, all_y)
        distillation_loss = cross_entropy_w_temp_scaling(target_output_old_cls, source_output, exp=1.0/self.temperature)
        total_loss = self.lambda_c * cross_entropy_loss + self.lambda_d * distillation_loss
        # print('lambda_c: {0}, lambda_d: {1}'.format(self.lambda_c, self.lambda_d))
        # print('cross_entropy_loss: {0} | distillation_loss: {1} | total_loss: {2}'.format(cross_entropy_loss, distillation_loss, total_loss))

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        loss_dict['cross_entropy_loss'] = cross_entropy_loss.item()
        loss_dict['distillation_loss'] = distillation_loss.item()
        loss_dict['loss'] = total_loss.item()

        return loss_dict

    def source_predict(self, x):  # for source model evaluation
        return self.source_network(x)

    def predict(self, x):  # for target evaluation
        if self.current_session == 0:
            return self.network(x)
        else:
            return self.target_network(x)

    def train_mode(self):
        if self.current_session == 0:
            self.network.train()
        else:
            self.source_network.eval()
            self.target_network.train()
    
    def eval_mode(self):
        if self.current_session == 0:
            self.network.eval()
        else:
            self.source_network.eval()
            self.target_network.eval()

    def encode(self, x):  # for target model obtain features
        if self.current_session == 0:
            return self.featurizer(x)
        else:
            return self.target_featurizer(x)




