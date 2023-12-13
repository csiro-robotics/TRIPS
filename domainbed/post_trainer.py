import collections
import json
import time
import copy
import os
from pathlib import Path

import numpy as np
import torch

from domainbed.lib.prototype import generate_prototype
from domainbed.lib.parameter import load_parameters
from domainbed.algorithms.ewc import compute_ewc_importance
from domainbed.algorithms.mas import compute_mas_importance

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def json_handler(v):
    if isinstance(v, (Path, range)):
        return str(v)
    raise TypeError(f"`{type(v)}` is not JSON Serializable")


def post_train(test_envs, args, eval_meta, algorithm, current_session, iid_prototype_dict=None, oracle_prototype_dict=None, old_precision_matrix=None):
    print('--- post train --- prototype, exemplar, importance matrix')

    if 'DIST' in algorithm.name:
        algorithm.eval_mode()
    else:
        algorithm.eval()
    
    #######################################################
    # store prototype
    #######################################################
    ckpt_dir = args.out_dir / "checkpoints"
    prototype_dir = args.out_dir / "prototypes"
    prototype_dir.mkdir(exist_ok=True)

    if 'DIST' in algorithm.name and current_session > 0:
        model = copy.deepcopy(algorithm.target_featurizer)
    else:
        model = copy.deepcopy(algorithm.featurizer)

    if args.model_type == 'iid':
        model_path = os.path.join(ckpt_dir, 'TE{0}_best_iid.pth'.format(test_envs[0]))
        load_parameters(model, model_path, algorithm.name, current_session)
        prototype_dict = generate_prototype(current_session, eval_meta, test_envs, model, args.debug, args.num_old_cls, args.num_new_cls, args.num_of_exemplar, args.PROTO_class_wise_domain_wise, args.PROTO_cov_sketching, args.PROTO_cov_sketchingRatio)
        if current_session == 0:
            proto_check = iid_prototype_dict
        else:
            if args.PROTO_class_wise_domain_wise:
                proto_check = iid_prototype_dict["cls_wise_domain_wise_avg_feature"]
            else:
                proto_check = iid_prototype_dict["cls_wise_avg_feature"]
        if not proto_check is None:
            prototype_dict["radius_value"] = iid_prototype_dict["radius_value"]
            if args.PROTO_class_wise_domain_wise:
                num_total_prototype = prototype_dict["cls_wise_domain_wise_avg_feature"].size(0)
                num_total_cls = args.num_old_cls + args.num_new_cls
                N = int(num_total_prototype / num_total_cls)
                # print('prototype_dict["cls_wise_domain_wise_avg_feature"] size: {0}'.format(prototype_dict["cls_wise_domain_wise_avg_feature"].size()))
                # print('prototype_dict["cls_wise_domain_wise_cls_label"] length: {0}'.format(len(prototype_dict["cls_wise_domain_wise_cls_label"])))
                # print("num_total_prototype: {0}, num_total_cls: {1}, N: {2}".format(num_total_prototype, num_total_cls, N))
                for i in range(N):
                    start_index = i * num_total_cls
                    end_index = i * num_total_cls + args.num_old_cls
                    start_index_old = i * args.num_old_cls
                    end_index_old = (i + 1) * args.num_old_cls
                    # print('i: {0}, start_index: {1}, end_index: {2}, end_index_old: {3}'.format(i, start_index, end_index, end_index_old))
                    prototype_dict["cls_wise_domain_wise_avg_feature"][start_index: end_index,:] = iid_prototype_dict["cls_wise_domain_wise_avg_feature"][start_index_old: end_index_old,:]
                    prototype_dict["cls_wise_domain_wise_cls_label"][start_index: end_index] = iid_prototype_dict["cls_wise_domain_wise_cls_label"][start_index_old: end_index_old]
            else:
                num_total_prototype = prototype_dict["cls_wise_avg_feature"].size(0)
                num_total_cls = args.num_old_cls + args.num_new_cls
                N = int(num_total_prototype / num_total_cls)
                for i in range(N):
                    start_index = i * num_total_cls
                    end_index = i * num_total_cls + args.num_old_cls
                    start_index_old = i * args.num_old_cls
                    end_index_old = (i + 1) * args.num_old_cls
                    prototype_dict["cls_wise_avg_feature"][start_index: end_index,:] = iid_prototype_dict["cls_wise_avg_feature"][start_index_old: end_index_old,:]
                    prototype_dict["cls_wise_cls_label"][start_index: end_index] = iid_prototype_dict["cls_wise_cls_label"][start_index_old: end_index_old]
                    if (prototype_dict["cls_wise_cov"] is not None) and (iid_prototype_dict["cls_wise_cov"] is not None):
                        prototype_dict["cls_wise_cov"][start_index: end_index] = iid_prototype_dict["cls_wise_cov"][start_index_old: end_index_old]
        save_path = os.path.join(prototype_dir, 'TE{0}_prototype_list_for_iid_model.pth'.format(test_envs[0]))
        torch.save(prototype_dict, save_path)
    elif args.model_type == 'oracle':
        model_path = os.path.join(ckpt_dir, 'TE{0}_best_oracle.pth'.format(test_envs[0]))
        load_parameters(model, model_path, algorithm.name, current_session)
        prototype_dict = generate_prototype(current_session, eval_meta, test_envs, model, args.debug, args.num_old_cls, args.num_new_cls, args.num_of_exemplar, args.PROTO_class_wise_domain_wise)
        if current_session == 0:
            proto_check = iid_prototype_dict
        else:
            if args.PROTO_class_wise_domain_wise:
                proto_check = oracle_prototype_dict["cls_wise_domain_wise_avg_feature"]
            else:
                proto_check = oracle_prototype_dict["cls_wise_avg_feature"]
        if not proto_check is None:
            prototype_dict["radius_value"] = oracle_prototype_dict["radius_value"]
            if args.PROTO_class_wise_domain_wise:
                num_total_prototype = prototype_dict["cls_wise_domain_wise_avg_feature"].size(0)
                num_total_cls = args.num_old_cls + args.num_new_cls
                N = int(num_total_prototype / num_total_cls)
                for i in range(N):
                    start_index = i * num_total_cls
                    end_index = i * num_total_cls + args.num_old_cls
                    end_index_old = i * args.num_old_cls
                    prototype_dict["cls_wise_domain_wise_avg_feature"][start_index: end_index,:] = oracle_prototype_dict["cls_wise_domain_wise_avg_feature"][start_index: end_index_old,:]
                    prototype_dict["cls_wise_domain_wise_cls_label"][start_index: end_index,:] = oracle_prototype_dict["cls_wise_domain_wise_cls_label"][start_index: end_index_old,:]
            else:
                num_total_prototype = prototype_dict["cls_wise_avg_feature"].size(0)
                num_total_cls = args.num_old_cls + args.num_new_cls
                N = int(num_total_prototype / num_total_cls)
                for i in range(N):
                    start_index = i * num_total_cls
                    end_index = i * num_total_cls + args.num_old_cls
                    end_index_old = i * args.num_old_cls
                    prototype_dict["cls_wise_avg_feature"][start_index: end_index,:] = oracle_prototype_dict["cls_wise_avg_feature"][start_index: end_index_old,:]
                    prototype_dict["cls_wise_cls_label"][start_index: end_index] = oracle_prototype_dict["cls_wise_cls_label"][start_index: end_index_old]
        save_path = os.path.join(prototype_dir, 'TE{0}_prototype_list_for_oracle_model.pth'.format(test_envs[0]))
        torch.save(prototype_dict, save_path)
    else:
        raise RuntimeError('Post train | Something wrong with the model type.')

    #######################################################
    # store importance matrix
    #######################################################
    if 'DIST' in algorithm.name and current_session > 0:
        model = copy.deepcopy(algorithm.target_network)
    else:
        model = copy.deepcopy(algorithm.network)

    if args.store_ewc_importance:
        importance_matrix_dir = args.out_dir / "importance_matrix"
        importance_matrix_dir.mkdir(exist_ok=True)
        ewc_importance_matrix = {}
        current_ewc_importance_matrix = compute_ewc_importance(model, eval_meta, test_envs)
        if old_precision_matrix:
            for name, value in current_ewc_importance_matrix.items():
                if name == '1.weight':
                    old_precision_value = torch.zeros(current_ewc_importance_matrix[name].size()).to("cuda")
                    old_precision_value[:old_precision_matrix[name].size()[0],:] = old_precision_matrix[name]
                    ewc_importance_matrix[name] = current_ewc_importance_matrix[name] + old_precision_value
                elif name == '1.bias':
                    old_precision_value = torch.zeros(current_ewc_importance_matrix[name].size()).to("cuda")
                    old_precision_value[:old_precision_matrix[name].size()[0]] = old_precision_matrix[name]
                    ewc_importance_matrix[name] = current_ewc_importance_matrix[name] + old_precision_value
                else:
                    ewc_importance_matrix[name] = current_ewc_importance_matrix[name] + old_precision_matrix[name]
        else:
            ewc_importance_matrix = current_ewc_importance_matrix
        save_path = os.path.join(importance_matrix_dir, 'TE{0}_{1}_ewc_importance_matrix.pth'.format(test_envs[0], args.model_type))
        torch.save(ewc_importance_matrix, save_path)
    elif args.store_mas_importance:
        importance_matrix_dir = args.out_dir / "importance_matrix"
        importance_matrix_dir.mkdir(exist_ok=True)
        mas_importance_matrix = {}
        current_mas_importance_matrix = compute_mas_importance(model, eval_meta, test_envs)
        if old_precision_matrix:
            for name, value in current_mas_importance_matrix.items():
                if name == '1.weight':
                    old_precision_value = torch.zeros(current_mas_importance_matrix[name].size()).to("cuda")
                    old_precision_value[:old_precision_matrix[name].size()[0],:] = old_precision_matrix[name]
                    mas_importance_matrix[name] = current_mas_importance_matrix[name] + old_precision_value
                elif name == '1.bias':
                    old_precision_value = torch.zeros(current_mas_importance_matrix[name].size()).to("cuda")
                    old_precision_value[:old_precision_matrix[name].size()[0]] = old_precision_matrix[name]
                    mas_importance_matrix[name] = current_mas_importance_matrix[name] + old_precision_value
                else:
                    mas_importance_matrix[name] = current_mas_importance_matrix[name] + old_precision_matrix[name]
        else:
            mas_importance_matrix = current_mas_importance_matrix
        save_path = os.path.join(importance_matrix_dir, 'TE{0}_{1}_mas_importance_matrix.pth'.format(test_envs[0], args.model_type))
        torch.save(mas_importance_matrix, save_path)

    


