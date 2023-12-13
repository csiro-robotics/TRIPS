import collections
import torch
import torch.nn as nn
import os
import json
import numpy as np
import torch.nn.functional as F
from sklearn.metrics.pairwise import pairwise_distances
from domainbed.lib.fast_data_loader import PrototypeDataLoader

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    

def store_exemplar(eval_meta, test_envs, algorithm, debug, num_old_cls, num_new_cls, num_of_exemplar, eval_metric='cosine'):
    exemplar_dict = {}

    final_exemplar_feature, final_exemplar_cls, final_exemplar_img_id = evaluate_and_choose_exemplar(eval_meta, test_envs, algorithm, debug, num_old_cls, num_new_cls, num_of_exemplar, eval_metric)

    # exemplar_dict['final_exemplar_feature'] = final_exemplar_feature
    exemplar_dict['final_exemplar_cls'] = final_exemplar_cls
    exemplar_dict['final_exemplar_img_id'] = final_exemplar_img_id
    """
    for i in range(len(final_exemplar_img_id)):
        for j in range(len(final_exemplar_img_id[i])):
            print('i: {0}, j: {1}, final_exemplar_img_id[i][j]: {2}'.format(i, j, final_exemplar_img_id[i][j]))
    """    
    return exemplar_dict


def evaluate_and_choose_exemplar(eval_meta, test_envs, model, debug=False, num_old_cls=0, num_new_cls=0, num_of_exemplar=5, eval_metric='cosine'):
    total_embedding = {}
    total_label = {}
    total_img_id = {}
    total_cascaded_embedding = []
    total_cascaded_label = []
    total_cascaded_img_id = []
    final_img_id = []
    overall_avg_feature = []
    overall_avg_cls = []
    cls_type_mask = [] # 0 for old class and 1 for new class

    for name, loader_kwargs, weights in eval_meta:
        env_name, inout = name.split("_")
        env_num = int(env_name[3:])
        # print('env_name: {0}, inout: {1}, env_num: {2}'.format(env_name, inout, env_num))
        if inout == "out": 
            # print('skip...')
            continue
        if env_num in test_envs and inout == "in": 
            # print('skip...')
            continue

        if isinstance(loader_kwargs, dict):
            loader = PrototypeDataLoader(**loader_kwargs)
        elif isinstance(loader_kwargs, FastDataLoader):
            loader = loader_kwargs
        else:
            raise ValueError(loader_kwargs)
        
        embedding_list = []
        label_list = []
        img_id_list = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                imd_id = batch["img_id"]
                embedding = model(x)
                embedding_list.append(embedding.cpu())
                label_list.append(y.cpu())
                for i in range(len(imd_id)):
                    img_id_list.append(imd_id[i])
            embedding_list = torch.cat(embedding_list, dim=0)
            label_list = torch.cat(label_list, dim=0)
        total_embedding["{0}_{1}".format(inout, env_num)] = embedding_list
        total_label ["{0}_{1}".format(inout, env_num)] = label_list
        total_img_id["{0}_{1}".format(inout, env_num)] = img_id_list

    for name, value in total_embedding.items():
        # print('name: {0}, value size: {1}'.format(name, value.size()))  # cls_wise_prototype
        total_cascaded_embedding.append(total_embedding[name])
        total_cascaded_label.append(total_label[name])
        total_cascaded_img_id.append(total_img_id[name])
    total_cascaded_embedding = torch.cat(total_cascaded_embedding, dim=0)
    total_cascaded_label = torch.cat(total_cascaded_label, dim=0)
    # print('total_cascaded_embedding size(): {0}'.format(total_cascaded_embedding.size()))
    # print('total_cascaded_label: {0}'.format(total_cascaded_label))
    # print('total_cascaded_img_id: {0}'.format(total_cascaded_img_id))
    for i in range(len(total_cascaded_img_id)):
        for j in range(len(total_cascaded_img_id[i])):
            final_img_id.append(total_cascaded_img_id[i][j])
    # print('length of final_img_id: {0}'.format(len(final_img_id)))
    
    # --- generate the average feature with all data ---
    # print('num_old_cls: {0}, num_new_cls: {1}'.format(num_old_cls, num_new_cls))
    for index in range((num_old_cls + num_new_cls)):
            class_index = (total_cascaded_label == index).nonzero()
            embedding_this = total_cascaded_embedding[class_index.squeeze(-1)]
            embedding_this = embedding_this.mean(0, keepdims=True).cuda()
            overall_avg_feature.append(embedding_this)
            overall_avg_cls.append(index)
            if index < num_old_cls:
                cls_type_mask.append(0)
            else:
                cls_type_mask.append(1)

    # --- find the closest features and corresponding images ---
    final_exemplar_feature = []
    final_exemplar_cls = []
    final_exemplar_img_id = []
    
    for index in range(num_old_cls, (num_old_cls + num_new_cls)):
        top_close_feature = []
        top_distance = []
        top_close_img_id = []
        
        class_index = (total_cascaded_label == index).nonzero()
        embedding_this = total_cascaded_embedding[class_index.squeeze(-1)]
        img_id_this = []
        for i in range(len(class_index)):
            img_id_this.append(final_img_id[class_index[i]])

        for i in range(len(embedding_this)):
            if i < num_of_exemplar:
                embedding_this_buff = embedding_this[i].view(-1, embedding_this[i].size()[0])
                top_close_feature.append(embedding_this_buff)
                distance = pairwise_distances(np.asarray(embedding_this_buff.cpu()), np.asarray(overall_avg_feature[index].cpu()), metric=eval_metric)
                top_distance.append(distance)
                top_close_img_id.append(img_id_this[i])
            else:
                embedding_this_buff = embedding_this[i].view(-1, embedding_this[i].size()[0])
                distance = pairwise_distances(np.asarray(embedding_this_buff.cpu()), np.asarray(overall_avg_feature[index].cpu()), metric=eval_metric)
                for j in range(len(top_distance)):
                    if distance < top_distance[j]:
                        top_close_feature[j] = embedding_this_buff
                        top_distance[j] = distance
                        top_close_img_id[j] = img_id_this[i]
                        break
        
        final_exemplar_feature.append(top_close_feature)
        final_exemplar_cls.append(index)
        final_exemplar_img_id.append(top_close_img_id)

    return final_exemplar_feature, final_exemplar_cls, final_exemplar_img_id


def load_exemplar(file_dir, test_envs, model_type):
    img_id = []

    file_name = 'exemplars/TE{0}_{1}_model_nearst_5_exemplar.txt'.format(test_envs[0], model_type)
    file_path = os.path.join(file_dir, file_name)
    # print('file_path: {0}'.format(file_path))

    with open(file_path) as f:
        contents = f.read()
    
    return img_id




            
        





