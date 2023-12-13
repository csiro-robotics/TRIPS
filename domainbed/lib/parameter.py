from copy import deepcopy

import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import variable
import torch.utils.data

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def store_parameters(model):
    param_dict = {}

    for name, parameter in model.named_parameters():
        param_dict[name] = parameter.clone().detach()
    """
    for name, parameter in param_dict.items():
        print('name: {0} | parameter: {1}'.format(name, parameter))
    """

    return param_dict


def load_parameters(model, parameter_path, algorithm_name, current_session):
    loaded_model = torch.load(parameter_path)
    loaded_model_dict = loaded_model['model_dict']
    model_dict = model.state_dict()
    
    for key, value in model_dict.items():
        if 'DIST' in algorithm_name and current_session > 0:
            key_name = 'target_featurizer.{0}'.format(key)
        else:
            key_name = 'featurizer.{0}'.format(key)
        # print('key: {0} | key_name: {1}'.format(key, key_name))
        model_dict[key] = loaded_model_dict[key_name]
    model.load_state_dict(model_dict)


def load_parameters_dict(file_dir, test_envs, model_type):
    file_name = 'checkpoints/TE{0}_best_{1}.pth'.format(test_envs[0], model_type)
    
    # print('file_dir: {0}'.format(file_dir))
    # print('file_name: {0}'.format(file_name))
    file_path = os.path.join(file_dir, file_name)
    # print('file_path: {0}'.format(file_path))

    loaded_model = torch.load(file_path)
    loaded_model_dict = loaded_model['model_dict']
    return loaded_model_dict

