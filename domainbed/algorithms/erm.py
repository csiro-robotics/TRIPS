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
from domainbed.losses.focal_loss import FocalLoss

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, current_session, num_of_exemplar):
        super(ERM, self).__init__(input_shape, num_classes, num_domains, hparams, current_session)
        self.name = 'ERM'
        self.current_session = current_session
        self.num_of_exemplar = num_of_exemplar
        self.Mixup_old_exemplar = hparams["exemplar_mixup"]
        self.Mixup_old_exemplar_times = hparams["exemplar_mixup_times"]
        self.Focal_loss = hparams["focal_loss"]
        self.hparam_Focal_gamma = hparams["focal_loss_gamma"]
        print('ERM | self.Focal_loss: {0}, self.hparam_Focal_gamma: {1}'.format(self.Focal_loss, self.hparam_Focal_gamma))
        print('ERM | self.Mixup_old_exemplar: {0}, self.Mixup_old_exemplar_times: {1}'.format(self.Mixup_old_exemplar, self.Mixup_old_exemplar_times))
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)

        self.optimizer = get_optimizer(hparams["optimizer"], self.network.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"])
        if self.Focal_loss:
            self.focal_loss = FocalLoss(gamma=self.hparam_Focal_gamma)

    def update(self, x, y, **kwargs):
        if self.num_of_exemplar > 0 and self.Mixup_old_exemplar and self.current_session > 0:
            x[-1], y[-1] = self.intra_class_intra_domain_mixup(x[-1], y[-1], kwargs["img_id"][-1], kwargs["envs"])

        all_x = torch.cat(x)
        all_y = torch.cat(y)
        if self.Focal_loss:
            loss = self.focal_loss(self.predict(all_x), all_y)
        else:
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

    def intra_class_intra_domain_mixup(self, x, y, img_id, domain_list, alpha=20.0, mix_times=4):  # mixup based
        batch_size = x.size()[0]

        domain_label = []
        for i in range(len(img_id)):
                current_img_id = img_id[i]
                domain_id = current_img_id.split('/')[5]
                domain_id_label = -1
                for k in range(len(domain_list)):
                    if domain_id == domain_list[k]:
                        domain_id_label = k
                        break
                domain_label.append(domain_id_label)
        all_domain = torch.IntTensor(domain_label)
        all_domain = all_domain.to("cuda")

        mix_data = []
        mix_target = []

        # print('fusion_aug_one_image | before fusion | length of the data: {0}, size of image: {1}'.format(len(y), x.size()))
        for _ in range(mix_times):
            index = torch.randperm(batch_size).cuda()
            for i in range(batch_size):
                if y[i] == y[index[i]] and domain_label[i] == domain_label[index[i]]:
                    new_label = y[i]
                    lam = np.random.beta(alpha, alpha)
                    if lam < 0.4 or lam > 0.6:
                        lam = 0.5
                    mix_data.append(lam * x[i] + (1 - lam) * x[index, :][i])
                    mix_target.append(new_label)

        new_target = torch.Tensor(mix_target)
        y = torch.cat((y, new_target.cuda().long()), 0)
        for item in mix_data:
            x = torch.cat((x, item.unsqueeze(0)), 0)
        # print('fusion_aug_one_image | after fusion | length of the data: {0}, size of image: {1}'.format(len(y), x.size()))

        return x, y
