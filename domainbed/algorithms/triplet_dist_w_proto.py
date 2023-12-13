import copy
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import os
import json

from domainbed import networks
from domainbed.lib.misc import random_pairs_of_minibatches
from domainbed.optimizers import get_optimizer

from domainbed.algorithms.algorithms import Algorithm
from domainbed.losses.triplet_loss import DomainTripletLoss
from domainbed.losses.common_loss_func import *
from domainbed.lib.proto_shift import PrototypeDrifting


class TRIPLET_DIST_W_PROTO(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams, current_session, num_of_exemplar, num_old_cls, temperature, old_prototype_dict, triplet_dist_type, check=False):
        super(TRIPLET_DIST_W_PROTO, self).__init__(input_shape, num_classes, num_domains, hparams, current_session)
        self.name = 'TRIPLET_DIST_W_PROTO'
        self.current_session = current_session
        self.w_triplet = hparams["w_triplet"]
        self.margin = hparams["margin"]
        self.triplet_dist_type = triplet_dist_type
        self.lambda_c = hparams["lambda_c"]
        self.lambda_d = hparams["lambda_d"]
        self.lambda_t = hparams["lambda_t"]
        self.w_proto = hparams["w_proto"]
        self.proto_semantic_shift = hparams["PROTO_semantic_shifting"]
        self.proto_shift_dict = {}
        self.proto_shift_dict["sigma"] = hparams["sigma"]
        self.proto_shift_dict["mean_MovingAvg_eta"] = hparams["PROTO_mean_MovingAvg_eta"]
        self.proto_shift_dict["mean_Balance_beta"] = hparams["PROTO_mean_Balance_beta"]
        self.proto_shift_dict["using_delta"] = hparams["using_delta"]
        self.Data_Augmentation = hparams["Data_Augmentation"]
        self.Proto_augmentation = hparams["PROTO_augmentation"]
        self.PROTO_augmentation_w_COV = hparams["PROTO_augmentation_w_COV"]
        self.cov_shift_dict = {}
        self.cov_shift_dict["cov_Shrinkage_alpha"] = hparams["PROTO_cov_Shrinkage_alpha"]
        self.cov_shift_dict["cov_MovingAvg_eta"] = hparams["PROTO_cov_MovingAvg_eta"]
        self.cov_shift_dict["cov_Balance_beta"] = hparams["PROTO_cov_Balance_beta"]
        
        self.PROTO_cov_sketching = hparams["PROTO_cov_sketching"]
        self.PROTO_cov_sketchingRatio = hparams["PROTO_cov_sketchingRatio"]
        
        if self.Data_Augmentation:
            self.num_of_old_cls = num_old_cls * 4
            self.num_of_cls = num_classes * 4
        else:
            self.num_of_old_cls = num_old_cls
            self.num_of_cls = num_classes
        self.radius = None

        print('TRIPLET_DIST_W_PROTO | w_triplet: {0}, triplet_dist_type: {1}, margin: {2}'.format(self.w_triplet, self.triplet_dist_type, self.margin))
        print('TRIPLET_DIST_W_PROTO | w_proto: {0}, proto_semantic_shift: {1}'.format(self.w_proto, self.proto_semantic_shift))
        print('TRIPLET_DIST_W_PROTO | Data_Augmentation: {0}'.format(self.Data_Augmentation))
        print('TRIPLET_DIST_W_PROTO | Proto_augmentation: {0}'.format(self.Proto_augmentation))
        print('TRIPLET_DIST_W_PROTO | PROTO_augmentation_w_COV: {0}'.format(self.PROTO_augmentation_w_COV))
        print('TRIPLET_DIST_W_PROTO | proto_shift_dict - using_delta: {0}'.format(self.proto_shift_dict["using_delta"]))
        print('TRIPLET_DIST_W_PROTO | proto_shift_dict - sigma: {0}, mean_MovingAvg_eta: {1}, mean_Balance_beta: {2}'.format(self.proto_shift_dict["sigma"], self.proto_shift_dict["mean_MovingAvg_eta"], self.proto_shift_dict["mean_Balance_beta"]))
        print('TRIPLET_DIST_W_PROTO | cov_shift_dict - cov_Shrinkage_alpha: {0}, cov_MovingAvg_eta: {1}, cov_Balance_beta: {2}'.format(self.cov_shift_dict["cov_Shrinkage_alpha"], self.cov_shift_dict["cov_MovingAvg_eta"], self.cov_shift_dict["cov_Balance_beta"]))
        print('TRIPLET_DIST_W_PROTO | PROTO_cov_sketching: {0}, PROTO_cov_sketchingRatio: {1}'.format(self.PROTO_cov_sketching, self.PROTO_cov_sketchingRatio))

        if current_session == 0:  # base session
            self.featurizer = networks.Featurizer(input_shape, self.hparams)
            self.classifier = nn.Linear(self.featurizer.n_outputs, self.num_of_cls)
            self.network = nn.Sequential(self.featurizer, self.classifier)

            self.optimizer = get_optimizer(
                hparams["optimizer"],
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )
            self.triplet_loss = DomainTripletLoss(hparams, margin=self.margin, hard_factor=0, feature_output=self.featurizer.n_outputs, dist_type=self.triplet_dist_type)
        else:  # incremental session
            self.target_featurizer = networks.Featurizer(input_shape, self.hparams)
            self.target_classifier = nn.Linear(self.target_featurizer.n_outputs, self.num_of_cls)
            self.target_network = nn.Sequential(self.target_featurizer, self.target_classifier)

            self.source_featurizer = networks.Featurizer(input_shape, self.hparams)
            self.source_classifier = nn.Linear(self.source_featurizer.n_outputs, self.num_of_old_cls)
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
            self.triplet_loss = DomainTripletLoss(hparams, margin=self.margin, hard_factor=0, feature_output=self.target_featurizer.n_outputs, dist_type=self.triplet_dist_type)
        
            self.temperature = temperature  # temperature for distillation

            self.old_prototype = old_prototype_dict["cls_wise_avg_feature"]
            self.prototype_cls_list = old_prototype_dict["cls_wise_cls_label"]   
            self.radius = old_prototype_dict["radius_value"] 
            self.cls_wise_cov = old_prototype_dict["cls_wise_cov"]
            if self.PROTO_cov_sketching:
                self.sketch_mat = old_prototype_dict["sketch_mat"]
                self.n_sketch_mat_ratio = old_prototype_dict["sketch_mat_ratio"]
            else:
                self.sketch_mat = None
                self.n_sketch_mat_ratio = None

            if self.proto_semantic_shift:
                self.prototype_shifting = PrototypeDrifting(self.proto_shift_dict, len(self.old_prototype), self.PROTO_augmentation_w_COV, self.cov_shift_dict, PROTO_sketching=self.PROTO_cov_sketching, sketch_mat=self.sketch_mat)

    def load_previous_model_param(self, dir, test_envs, type):
        if type == 'last_step':
            old_model_path = os.path.join(dir, "TE{0}_last_step.pth".format(test_envs[0]))
        elif type == 'iid':
            old_model_path = os.path.join(dir, "TE{0}_best_iid.pth".format(test_envs[0]))
        elif type == 'oracle':
            old_model_path = os.path.join(dir, "TE{0}_best_oracle.pth".format(test_envs[0]))
        else:
            raise ValueError("Something wrong with the model type.")

        print('load_previous_model_param | old_model_path: {0}'.format(old_model_path))
        old_model_dict = torch.load(old_model_path)
        old_model_param_dict = old_model_dict['model_dict']
        
        if self.current_session  > 1:
            network_name = 'target_network'
        else:
            network_name = 'network'
        target_network_dict = self.target_network.state_dict()

        for k, v in target_network_dict.items():
            if k == '1.weight':  # classifier - weight
                num_old_cls = old_model_param_dict['{0}.{1}'.format(network_name, k)].size()[0]
                num_total_cls = target_network_dict[k].size()[0]
                target_network_dict[k][:num_old_cls,:] = old_model_param_dict['{0}.{1}'.format(network_name, k)]
            elif k == '1.bias':  # classifier - bias
                num_old_cls = old_model_param_dict['{0}.{1}'.format(network_name, k)].size()[0]
                num_total_cls = target_network_dict[k].size()[0]
                target_network_dict[k][:num_old_cls] = old_model_param_dict['{0}.{1}'.format(network_name, k)]
            else:
                target_network_dict[k] = old_model_param_dict['{0}.{1}'.format(network_name, k)]
        self.target_network.load_state_dict(target_network_dict)
        
        source_network_dict = self.source_network.state_dict()
        for k, v in source_network_dict.items():
            if k == '1.weight':  # classifier - weight
                num_old_cls = old_model_param_dict['{0}.{1}'.format(network_name, k)].size()[0]
                num_total_cls = source_network_dict[k].size()[0]
                source_network_dict[k][:num_old_cls,:] = old_model_param_dict['{0}.{1}'.format(network_name, k)]
            elif k == '1.bias':  # classifier - bias
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
        all_img_id = kwargs["img_id"]
        domain_list = kwargs["envs"]
        all_domain = self.generate_domain_label(all_img_id, domain_list)

        if self.Data_Augmentation:
            all_x = torch.stack([torch.rot90(all_x, k, (2, 3)) for k in range(4)], 1)
            all_x = all_x.view(-1, 3, 224, 224)
            all_y = torch.stack([all_y * 4 + k for k in range(4)], 1).view(-1)
            all_domain = torch.stack([all_domain for k in range(4)], 1).view(-1)

        all_feature = self.featurizer(all_x)
        all_output = self.classifier(all_feature)
        if self.w_triplet:
            cross_entropy_loss = F.cross_entropy(all_output, all_y)
            triplet_loss_val = self.triplet_loss(all_feature, all_y, all_domain)[0]
            loss = self.lambda_c * cross_entropy_loss + self.lambda_t * triplet_loss_val
            # print('TRIPLET_DIST_W_PROTO | w_triplet | cross_entropy_loss: {0}, triplet_loss: {1}, total_loss: {2}'.format(cross_entropy_loss, triplet_loss_val, loss))
        else:
            cross_entropy_loss = F.cross_entropy(all_output, all_y)
            loss = self.lambda_c * cross_entropy_loss
            # print('TRIPLET_DIST_W_PROTO | no_triplet | cross_entropy_loss: {0}, total_loss: {1}'.format(cross_entropy_loss, loss))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}
            
    def update_incremental(self, x, y, **kwargs):
        loss_dict = {}
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        all_img_id = kwargs["img_id"]
        domain_list = kwargs["envs"]
        all_domain = self.generate_domain_label(all_img_id, domain_list)

        # self-supervised learning based data augmentation
        if self.Data_Augmentation:
            all_x = torch.stack([torch.rot90(all_x, k, (2, 3)) for k in range(4)], 1)
            all_x = all_x.view(-1, 3, 224, 224)
            all_y = torch.stack([all_y * 4 + k for k in range(4)], 1).view(-1)
            all_domain = torch.stack([all_domain for k in range(4)], 1).view(-1)

        target_feature = self.target_featurizer(all_x)
        source_feature = self.source_featurizer(all_x)
        target_output = self.target_classifier(target_feature)
        source_output = self.source_classifier(source_feature)
        target_output_old_cls = target_output[:, :source_output.size()[1]]
        target_output_new_cls = target_output[:, source_output.size()[1]:]
        
        # 1: prototype semantic shifting 
        if self.proto_semantic_shift:
            # print('------ proto_semantic_shift')
            x_new_feature = target_feature.detach()
            x_old_feature = source_feature.detach()
            self.updated_mean, self.updated_cov = self.prototype_shifting.prototype_update(x_old_feature, all_y, x_new_feature, all_y, self.old_prototype, self.cls_wise_cov)
            # print('~~~ updated_mean size: {0}'.format(self.updated_mean.size()))
            # print('~~~ updated_cov size: {0}'.format(self.updated_cov.size()))
        else:
            self.updated_mean = self.old_prototype

        # 2:  prototype augmentation
        if self.Proto_augmentation:
            # print('------ Proto_augmentation')
            if not self.PROTO_augmentation_w_COV:
                # generate pseudo features with mean
                augmented_prototype, augmented_prototype_label = self.prototype_augmentation(self.updated_mean, self.prototype_cls_list, all_y)
            else:
                # generate pseudo features with mean and covariance
                augmented_prototype, augmented_prototype_label = self.prototype_augmentation_w_cov(self.updated_mean, self.updated_cov, self.prototype_cls_list, all_y)
        else:
            augmented_prototype = self.updated_mean.float().to("cuda")
            augmented_prototype_label = self.prototype_cls_list.long().to("cuda")

        # print('augmented_prototype | device: {0}, type" {1}'.format(augmented_prototype.get_device(), augmented_prototype.type()))
        # print('augmented_prototype_label | device: {0}, type" {1}'.format(augmented_prototype_label.get_device(), augmented_prototype_label.type()))
        
        # calculate diferent loss terms
        cross_entropy_loss = self.calculate_new_cls_learning_loss(target_output, all_y, augmented_prototype, augmented_prototype_label)
        distillation_loss_val = cross_entropy_w_temp_scaling(target_output_old_cls, source_output, exp=1.0/self.temperature)
        if self.w_triplet:
            triplet_loss_val = self.calculate_triplet_loss(target_feature, all_y, all_domain, augmented_prototype)
            total_loss = self.lambda_c * cross_entropy_loss + self.lambda_d * distillation_loss_val + self.lambda_t * triplet_loss_val 
            # print('TRIPLET_DIST_W_PROTO | w_triplet | cross_entropy_loss: {0}, triplet_loss: {1}, distillation_loss: {2}, total_loss: {3}'.format(cross_entropy_loss, triplet_loss_val, distillation_loss_val, total_loss))
        else:
            total_loss = self.lambda_c * cross_entropy_loss + self.lambda_d * distillation_loss_val
            # print('TRIPLET_DIST_W_PROTO | no_triplet | cross_entropy_loss: {0}, distillation_loss: {1}, total_loss: {2}'.format(cross_entropy_loss, distillation_loss_val, total_loss))

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        if self.w_triplet:
            loss_dict['triplet_loss'] = triplet_loss_val.item()
        loss_dict['cross_entropy_loss'] = cross_entropy_loss.item()
        loss_dict['distillation_loss'] = distillation_loss_val.item()
        loss_dict['loss'] = total_loss.item() 

        return loss_dict

    def predict(self, x):  # for target evaluation
        if self.current_session == 0:
            return self.predict_base(x)
        else:
            return self.predict_incremental(x)

    def predict_base(self, x):  # used in evaluation (evaluator.py)
        output = self.network(x)
        if self.Data_Augmentation:
            output = output[:, ::4]  # only compute predictions on original class nodes
        return output

    def predict_incremental(self, x):  # used in evaluation (evaluator.py)
        output = self.target_network(x)
        if self.Data_Augmentation:
            output = output[:, ::4]  # only compute predictions on original class nodes        
        return output

    def source_predict(self, x):  # for source model evaluation
        output = self.source_network(x)
        if self.Data_Augmentation:
            output = output[:, ::4]  # only compute predictions on original class nodes
        return output

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


    def generate_domain_label(self, all_img_id, domain_list):
        domain_label = []

        for i in range(len(all_img_id)):
            for j in range(len(all_img_id[i])):
                img_id = all_img_id[i][j]
                # print('img_id: {0}'.format(img_id))
                domain_id = img_id.split('/')[-3]
                # print('domain_id: {0}'.format(domain_id))
                domain_id_label = -1
                for k in range(len(domain_list)):
                    # print('k: {0}, domain_list[k]: {1}'.format(k, domain_list[k]))
                    if domain_id == domain_list[k]:
                        domain_id_label = k
                        break
                # print('domain_id_label: {0}'.format(domain_id_label))
                domain_label.append(domain_id_label)
        # print('domain_label: {0}'.format(domain_label))
        all_domain = torch.IntTensor(domain_label)
        all_domain = all_domain.to("cuda")
        # print('all_domain: {0}'.format(all_domain))  
        return all_domain


    def calculate_new_cls_learning_loss(self, target_output, all_y, updated_prototype, prototype_cls_list):
        batch_data_cross_entropy_loss = F.cross_entropy(target_output, all_y)
        
        if self.w_proto:
            prototype_predict = self.target_classifier(updated_prototype)
            prototype_cross_entropy_loss = F.cross_entropy(prototype_predict, prototype_cls_list)
            cross_entropy_loss = prototype_cross_entropy_loss + batch_data_cross_entropy_loss
            # print('TRIPLET_DIST_W_PROTO | w_proto | batch_data_cross_entropy_loss: {0}, prototype_cross_entropy_loss: {1}'.format(batch_data_cross_entropy_loss, prototype_cross_entropy_loss))
        else:
            cross_entropy_loss = batch_data_cross_entropy_loss
            # print('TRIPLET_DIST_W_PROTO | no_proto | batch_data_cross_entropy_loss: {0}'.format(batch_data_cross_entropy_loss))
        # print('TRIPLET_DIST_W_PROTO | cross_entropy_loss: {0}'.format(cross_entropy_loss))
        return cross_entropy_loss

    def calculate_triplet_loss(self, target_feature, all_y, all_domain, updated_prototype):
        if self.w_proto:
            triplet_loss_val = self.triplet_loss(target_feature, all_y, all_domain, old_prototype=updated_prototype)[0]
            # print('TRIPLET_DIST_W_PROTO | w_proto | triplet_loss_val: {0}'.format(triplet_loss_val))
        else:
            triplet_loss_val = self.triplet_loss(target_feature, all_y, all_domain)[0]
            # print('TRIPLET_DIST_W_PROTO | no_proto | triplet_loss_val: {0}'.format(triplet_loss_val))

        return triplet_loss_val


    def prototype_augmentation(self, original_prototype, original_prototype_label, all_y):
        aug_prototype = []
        aug_prototype_label = []

        original_prototype = np.float32(original_prototype.cpu())
        index = list(range(len(original_prototype_label)))

        for _ in range(all_y.size()[0]):
            np.random.shuffle(index)
            temp = original_prototype[index[0]] + np.random.normal(0, 1, 512) * self.radius
            aug_prototype.append(temp)
            if self.Data_Augmentation:
                aug_prototype_label.append(4 * original_prototype_label[index[0]])
            else:
                aug_prototype_label.append(original_prototype_label[index[0]])
        
        aug_prototype = torch.from_numpy(np.float32(np.asarray(aug_prototype))).float().to("cuda")
        aug_prototype_label = torch.from_numpy(np.asarray(aug_prototype_label)).type(torch.LongTensor).to("cuda")
        return aug_prototype, aug_prototype_label

           
    def prototype_augmentation_w_cov(self, original_prototype, original_cov, original_prototype_label, all_y):
        # print('Pseudo sampling with mean and covariance.')

        aug_prototype = []
        aug_prototype_label = []

        index = list(range(len(original_prototype_label)))

        indexes_vec=torch.zeros(all_y.size()[0], dtype=torch.long)
        for ii in range(all_y.size()[0]):
            np.random.shuffle(index)
            indexes_vec[ii]=index[0]
            
        #repeat cov and means by classes
        expanded_prototypes=original_prototype[indexes_vec, :]
        expanded_covs=original_cov[indexes_vec, :, :]
        # print(expanded_prototypes.size())
        # print(expanded_covs.size())

        xNormDist=torch.distributions.MultivariateNormal(loc=expanded_prototypes, covariance_matrix=expanded_covs)
        mnSamples=xNormDist.sample( [ 1 ] ).squeeze(0)
        # print(mnSamples.size())
        # print(mnSamples[0:4,0:5])

        if self.PROTO_cov_sketching:
            mnSamples=mnSamples.mm(self.sketch_mat.t())*self.n_sketch_mat_ratio

        aug_prototype=mnSamples.type(torch.cuda.FloatTensor)

        if self.Data_Augmentation:
            aug_prototype_label=original_prototype_label[indexes_vec] * 4 
        else:
            aug_prototype_label=original_prototype_label[indexes_vec]
    
        aug_prototype_label = aug_prototype_label.type(torch.LongTensor)
        aug_prototype_label=aug_prototype_label.to("cuda")
        #aug_prototype_label=aug_prototype_label.tolist()

        return aug_prototype, aug_prototype_label


    