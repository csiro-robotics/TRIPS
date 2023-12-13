import collections
import torch
import torch.nn as nn
import os
import json
import numpy as np
import torch.nn.functional as F
import random
from domainbed.lib.fast_data_loader import PrototypeDataLoader

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def create_s_dense(hi,si,c):
    d = hi.size(0)
    out = torch.zeros((d,c),device=hi.device)  # in*out
    out[torch.arange(0,d),hi.type(torch.LongTensor)]=si
    return out

def choose_h_sk_mat(nSketchDimC, nFeatDimC,device):
    nRep=int(np.ceil(nFeatDimC/nSketchDimC))
    rand_array=torch.zeros(nFeatDimC,device=device)
    for i in range(nRep):
        rand_array_i=torch.randperm(int(nSketchDimC))
        rand_array[i*nSketchDimC:(i+1)*nSketchDimC]=rand_array_i

    return rand_array

def choose_s_sk_mat(nSketchDimC, nFeatDimC,device):
    nRep=int(np.ceil(nFeatDimC/nSketchDimC))
    rand_array=[-1, 1]*nRep
    random.shuffle(rand_array)
    return torch.tensor(rand_array,dtype=torch.float32,device=device)

def create_sketch_mat(dim, sketch_dim,device):
    h1 = choose_h_sk_mat(sketch_dim, dim,device)
    s1=choose_s_sk_mat(2, dim,device)
    sdense1 = create_s_dense(h1,s1,sketch_dim)
      
    return sdense1

def generate_average_feature_w_all_training_data(current_session, eval_meta, test_envs, model, debug=False, num_old_cls=0, num_new_cls=0, PROTO_class_wise_domain_wise=False, PROTO_cov_sketching=False, PROTO_cov_sketchingRatio=1):
    total_embedding = {}
    total_label = {}
    cls_wise_domain_wise_avg_feature = []
    cls_wise_domain_wise_cls_label = []
    cls_wise_domain_wise_domain_label = []
    cls_wise_domain_wise_num_img = []
    cls_wise_domain_wise_cls_type_mask = [] # 0 for old class and 1 for new class
    total_cascaded_embedding = []
    total_cascaded_label = []
    cls_wise_avg_feature = []
    cls_wise_cls_label = []
    cls_wise_cls_type_mask = [] # 0 for old class and 1 for new class
    radius = []
    prototype_dict = {}
    cls_wise_cov=[]

    for name, loader_kwargs, weights in eval_meta:
        env_name, inout = name.split("_")
        env_num = int(env_name[3:])
        # print('generate_average_feature_w_all_training_data | env_name: {0}, inout: {1}, env_num: {2}'.format(env_name, inout, env_num))
        if inout == "out": 
            # print('skip...')
            continue
        if env_num in test_envs and inout == "in": 
            # print('skip...')
            continue
        
        if isinstance(loader_kwargs, dict):
            loader = PrototypeDataLoader(**loader_kwargs)
        elif isinstance(loader_kwargs, PrototypeDataLoader):
            loader = loader_kwargs
        else:
            raise ValueError(loader_kwargs)
                
        model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                embedding = model(x)
                # print('embedding size(): {0}'.format(embedding.size()))
                embedding_list.append(embedding.cpu())
                label_list.append(y.cpu())
            embedding_list = torch.cat(embedding_list, dim=0)
            label_list = torch.cat(label_list, dim=0)
        total_embedding["{0}_{1}".format(inout, env_num)] = embedding_list
        total_label ["{0}_{1}".format(inout, env_num)] = label_list

    # --- generate class-wise prototypes ---
    print('--- generate class-wise prototypes ---')
    for name, value in total_embedding.items():
        total_cascaded_embedding.append(total_embedding[name])
        total_cascaded_label.append(total_label[name])
    total_cascaded_embedding = torch.cat(total_cascaded_embedding, dim=0)
    total_cascaded_label = torch.cat(total_cascaded_label, dim=0)

    if PROTO_cov_sketching:
        print('prototype | use sketching to reduce covariance size, sketching_ratio: {0}'.format(PROTO_cov_sketchingRatio))
        # feature size
        n_size_in=total_cascaded_embedding.size()[1]
        n_size_out=int(n_size_in/PROTO_cov_sketchingRatio)
        n_ratio=n_size_out/n_size_in
        # create sketch matrix
        sketch_mat=create_sketch_mat(n_size_in, n_size_out, "cuda:0").type(torch.cuda.DoubleTensor)
        prototype_dict["sketch_mat"] = sketch_mat
        prototype_dict["sketch_mat_ratio"] = n_ratio
    else:
        prototype_dict["sketch_mat"] = None
        prototype_dict["sketch_mat_ratio"] = None
    
    for index in range((num_old_cls + num_new_cls)):
        class_index = (total_cascaded_label == index).nonzero()
        embedding_this = total_cascaded_embedding[class_index.squeeze(-1)].type(torch.cuda.DoubleTensor)
        
        if PROTO_cov_sketching:
            embedding_this=embedding_this.mm(sketch_mat)
        
        embedding_mean = embedding_this.mean(0, keepdims=True)
        embedding_this_zerocenter=embedding_this-embedding_mean
        embedding_cov=embedding_this_zerocenter.t().mm(embedding_this_zerocenter)
        cls_wise_cov.append(embedding_cov.unsqueeze(0))
        if current_session == 0:
            radius.append(embedding_cov.trace().cpu().numpy() / embedding_mean.size()[1])

        cls_wise_avg_feature.append(embedding_mean)
        cls_wise_cls_label.append(torch.ones(1)*index)
        if index < num_old_cls:
            cls_wise_cls_type_mask.append(0)
        else:
            cls_wise_cls_type_mask.append(1)
    cls_wise_avg_feature = torch.cat(cls_wise_avg_feature)
    cls_wise_cov=torch.cat(cls_wise_cov)
    cls_wise_cls_label=torch.cat(cls_wise_cls_label)

    if current_session == 0:
            radius_value = np.sqrt(np.mean(radius))

    prototype_dict["cls_wise_avg_feature"] = cls_wise_avg_feature
    prototype_dict["cls_wise_cov"] = cls_wise_cov
    prototype_dict["cls_wise_cls_label"] = cls_wise_cls_label
    prototype_dict["cls_wise_cls_type_mask"] = cls_wise_cls_type_mask
    if current_session == 0:
        prototype_dict["radius_value"] = radius_value
    else:
        prototype_dict["radius_value"] = 0

    return prototype_dict


def generate_prototype(current_session, eval_meta, test_envs, model, debug, num_old_cls, num_new_cls, num_of_exemplar=0, PROTO_class_wise_domain_wise=False, PROTO_cov_sketching=False, PROTO_cov_sketchingRatio=1):
    prototype_dict = generate_average_feature_w_all_training_data(current_session, eval_meta, test_envs, model, debug, num_old_cls, num_new_cls, PROTO_class_wise_domain_wise, PROTO_cov_sketching, PROTO_cov_sketchingRatio)
    
    return prototype_dict


def load_prototype(file_dir, test_envs, model_type, PROTO_class_wise_domain_wise):
    overall_avg_feature = []
    overall_avg_cls = []
    cls_type_mask = []

    file_name = 'prototypes/TE{0}_prototype_list_for_{1}_model.pth'.format(test_envs[0], model_type)
    file_path = os.path.join(file_dir, file_name)

    prototype_dict = torch.load(file_path)

    cls_wise_avg_feature = prototype_dict['cls_wise_avg_feature']
    cls_wise_cls_label = prototype_dict['cls_wise_cls_label']
    cls_wise_cls_type_mask = prototype_dict['cls_wise_cls_type_mask']
    radius_value = prototype_dict["radius_value"]
    cls_wise_cov = prototype_dict["cls_wise_cov"]
    sketch_mat = prototype_dict["sketch_mat"]
    n_ratio = prototype_dict["sketch_mat_ratio"]
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('cls_wise_avg_feature size: {0}'.format(cls_wise_avg_feature.size()))
    print('cls_wise_cls_label: {0}'.format(cls_wise_cls_label))
    print('cls_wise_cls_type_mask: {0}'.format(cls_wise_cls_type_mask))
    print('radius_value: {0}'.format(radius_value))
    print('cls_wise_cov size: {0}'.format(cls_wise_cov.size()))
    if not sketch_mat == None:
        print('sketch_mat size: {0}'.format(sketch_mat.size()))
    else:
        print('sketch_mat: {0}'.format(sketch_mat))
    print('n_ratio: {0}'.format(n_ratio))
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    return prototype_dict


