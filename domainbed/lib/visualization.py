import collections
import torch
import torch.nn as nn
import os
import json
import numpy as np
import torch.nn.functional as F
from domainbed.lib.fast_data_loader import PrototypeDataLoader
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def plot_embedding(data, label, title, path, num_of_cls, start_cls=0):
    x_min = np.min(data, 0)
    x_max = np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    # print('number of data: {0}'.format(data.shape[0]))
    # print('number of class: {0}'.format(num_of_cls))
    # print('label: {0}'.format(label))
    plt.figure()
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1((label[i] - start_cls) / float(num_of_cls)),
                 fontdict={'weight': 'bold', 'size': 9})
    # plt.xticks([])
    # plt.yticks([])
    # plt.title(title)
    plt.axis('off')


    plt.savefig('{0}/{1}.jpg'.format(path, title))


def draw_tsne_figure(feature, label, path, title, num_of_cls, start_cls):
    feature_list = []
    label_list = []

    if len(feature) != len(label):
        raise ValueError('Something is wrong')

    for i in range(len(feature)):
        feature[i] = feature[i].clone().detach()
        buff = feature[i].view(1, 512)
        feature_list.append(buff.cpu().detach().numpy())
        label[i] = label[i].clone().detach()
        label_list.append(label[i].cpu().detach().numpy())
        # label_list.append(label[i])
    feature_list = np.concatenate(feature_list, axis=0)

    # tsne = TSNE(n_components=2, init='random', random_state=5, verbose=1)
    tsne = TSNE(n_components=2,init='pca',random_state=501)
    result = tsne.fit_transform(feature_list)
    # print('path: {0}'.format(path))
    plot_embedding(result, label_list, title, path, num_of_cls, start_cls)


def plot_embedding_w_avg_embedding(data, label, title, path, num_of_cls, num_of_avg_feature, start_cls=0):

    x_min = np.min(data, 0)
    x_max = np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    # print('number of data: {0}'.format(data.shape[0]))
    # print('number of class: {0}'.format(num_of_cls))
    # print('label: {0}'.format(label))
    plt.figure()
    for i in range(0, data.shape[0] - num_of_avg_feature):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1((label[i] - start_cls) / float(num_of_cls)),
                 fontdict={'weight': 'normal', 'size': 9})

    for i in range(data.shape[0] - num_of_avg_feature, data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color='black',
                 fontdict={'weight': 'bold', 'size': 12})
    # plt.xticks([])
    # plt.yticks([])
    # plt.title(title)
    plt.axis('off')
    plt.savefig('{0}/{1}.jpg'.format(path, title))


def draw_tsne_figure_w_avg_embedding(feature, label, avg_feature, avg_label, path, title, num_of_cls, start_cls):
    feature_list = []
    label_list = []

    if len(feature) != len(label):
        raise ValueError('Something is wrong')

    for i in range(len(feature)):
        feature[i] = feature[i].clone().detach().view(1, -1)
        buff = feature[i].view(1, 512)
        feature_list.append(buff.cpu().detach().numpy())
        label[i] = label[i].clone().detach()
        label_list.append(label[i].cpu().detach().numpy())

    # ---------------------------------------------------------

    if len(avg_feature) != len(avg_label):
        raise ValueError('Something is wrong')
    num_of_avg_feature = len(avg_feature)

    for i in range(len(avg_feature)):
        avg_feature[i] = avg_feature[i].view(1, -1)
        buff = avg_feature[i].view(1, 512)
        feature_list.append(buff.cpu().detach().numpy())
        label_list.append(avg_label[i])
    feature_list = np.concatenate(feature_list)

    # tsne = TSNE(n_components=2, init='random', random_state=5, verbose=1)
    tsne = TSNE(n_components=2,init='pca',random_state=501)
    feature_result = tsne.fit_transform(feature_list)

    plot_embedding_w_avg_embedding(feature_result, label_list, title, path, num_of_cls, num_of_avg_feature, start_cls)


def visualize(eval_meta, test_envs, model, path, dataset_type, debug=False, num_old_cls=0, num_new_cls=0, old_prototypes=None):
    total_embedding = {}
    total_label = {}
    total_cascaded_embedding = []
    total_cascaded_label = []
    overall_avg_feature = []
    overall_avg_cls = []

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
    
    for name, value in total_embedding.items():
        # print('name: {0}, value size: {1}'.format(name, value.size()))  # cls_wise_prototype
        total_cascaded_embedding.append(total_embedding[name])
        total_cascaded_label.append(total_label[name])
    total_cascaded_embedding = torch.cat(total_cascaded_embedding, dim=0)
    total_cascaded_label = torch.cat(total_cascaded_label, dim=0)
    print('total_cascaded_embedding size(): {0}'.format(total_cascaded_embedding.size()))
    # print('total_cascaded_label: {0}'.format(total_cascaded_label))
    
    # generate the average feature with all data
    # print('num_old_cls: {0}, num_new_cls: {1}'.format(num_old_cls, num_new_cls))
    for index in range((num_old_cls + num_new_cls)):
            class_index = (total_cascaded_label == index).nonzero()
            embedding_this = total_cascaded_embedding[class_index.squeeze(-1)]
            embedding_this = embedding_this.mean(0, keepdims=True).cuda()
            overall_avg_feature.append(embedding_this)
            overall_avg_cls.append(index)
    
    if 'INC' in dataset_type:
        title = 'tsne_new_cls_test_{0}'.format(test_envs[0])
    elif 'TSNE' in dataset_type:
        title = 'tsne_old_and_new_cls_test_{0}'.format(test_envs[0])
    else:
        title = 'tsne_test_{0}'.format(test_envs[0])
    # print('title: {0}'.format(title))
    num_of_cls = num_old_cls + num_new_cls

    if old_prototypes == None:
        # draw_tsne_figure(total_cascaded_embedding, total_cascaded_label, path, title, num_of_cls, start_cls=0)
        title = '{0}_w_prototype_current_calculated'.format(title)
        draw_tsne_figure_w_avg_embedding(total_cascaded_embedding, total_cascaded_label, overall_avg_feature, overall_avg_cls, path, title, num_of_cls, start_cls=0)
    else:
        avg_feature = []
        avg_label = []
        for i in range(len(old_prototypes)):
            if not np.isnan(np.array(old_prototypes[i].cpu())).any():
                # print('old_prototypes[i][0]: {0}'.format(old_prototypes[i][0]))
                # print('i: {0}, old_prototypes[{0}] size: {1}'.format(i, old_prototypes[i].size()))
                avg_feature.append(old_prototypes[i])
                avg_label.append(i)
        """
        for i in range(len(avg_feature)):
            print('avg_feature[{0}] size: {1}'.format(i, avg_feature[i].size()))
        for i in range(len(avg_label)):
            print('avg_label[{0}]: {1}'.format(i, avg_label[i]))
        """

        # title = '{0}_w_prototype_current_calculated'.format(title)
        # draw_tsne_figure_w_avg_embedding(total_cascaded_embedding, total_cascaded_label, overall_avg_feature, overall_avg_cls, path, title, num_of_cls, start_cls=0)
        title = '{0}_w_prototype'.format(title)
        draw_tsne_figure_w_avg_embedding(total_cascaded_embedding, total_cascaded_label, avg_feature, avg_label, path, title, num_of_cls, start_cls=0)
