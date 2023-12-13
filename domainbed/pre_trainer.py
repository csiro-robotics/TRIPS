import collections
import json
import time
import copy
import os
from pathlib import Path

import numpy as np
import torch

from domainbed.lib.parameter import load_parameters_dict
from domainbed.lib.prototype import load_prototype


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def pre_train(args, test_envs, output_path=None):
    print('--- pre train --- parameter, prototype, importance matrix')
    
    if output_path == None:
        output_path = Path(".") / Path("train_output") / args.dataset / "trial_seed_{0}".format(args.trial_seed)
    old_info_path = os.path.join(output_path, args.load_old_info)
    #######################################################
    # load old model parameter
    #######################################################
    old_parameters = load_parameters_dict(old_info_path, test_envs, args.model_type)

    #######################################################
    # load old class-wise prototypes
    #######################################################
    old_prototype_dict = load_prototype(old_info_path, test_envs, args.model_type, args.PROTO_class_wise_domain_wise)

    #######################################################
    # load importance matrix
    #######################################################
    if args.store_ewc_importance and args.current_session > 0:
        precision_matrix = load_importance_matrix(old_info_path, test_envs, args.model_type , 'ewc')
    elif args.store_mas_importance and args.current_session > 0:
        precision_matrix = load_importance_matrix(old_info_path, test_envs, args.model_type , 'mas')
    else:
        precision_matrix = None
    
    return old_parameters, old_prototype_dict, precision_matrix
    

def load_importance_matrix(file_dir, test_envs, model_type, method_type):
    importance_matrix = {}

    file_name = 'importance_matrix/TE{0}_{1}_{2}_importance_matrix.pth'.format(test_envs[0], model_type, method_type)
    file_path = os.path.join(file_dir, file_name)
    # print('file_path: {0}'.format(file_path))
    importance_matrix = torch.load(file_path)
    return importance_matrix