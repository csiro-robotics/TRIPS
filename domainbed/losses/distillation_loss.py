import torch
from torch import nn
import torch.nn.functional as F


def cross_entropy_w_temp_scaling(target_output, source_output, exp=1.0, size_average=True, eps=1e-5, overall_normalization=False):
    """
    Calculates cross-entropy with temperature scaling
    target_output: prediction output from target model
    source_output: prediction output from source model
    
    LwF method uses this function to calculate the distillation loss (normalized over only old classes).
    """
    num_imgs = source_output.size()[0]
    num_old_cls = source_output.size()[1]
    # print('distillation_loss | num_imgs: {0}, num_old_cls: {1}'.format(num_imgs, num_old_cls))

    if overall_normalization:  # target model output normalized over all classes (old & new)
        # print('target model output normalized over all classes (old & new)')
        target_out = torch.nn.functional.softmax(target_output, dim=1)
        target_out = target_out[:, :num_old_cls]
        source_out = torch.nn.functional.softmax(source_output, dim=1)
    else:
        # print('target model output normalized over only old classes')
        target_output = target_output[:, :num_old_cls]
        target_out = torch.nn.functional.softmax(target_output, dim=1)
        source_out = torch.nn.functional.softmax(source_output, dim=1)
    # print('target_out[:5, :]: {0}'.format(target_out[:5, :]))
    # print('source_out[:5, :]: {0}'.format(source_out[:5, :]))

    if exp != 1:
        target_out = target_out.pow(exp)
        target_out = target_out / target_out.sum(1).view(-1, 1).expand_as(target_out)
        source_out = source_out.pow(exp)
        source_out = source_out / source_out.sum(1).view(-1, 1).expand_as(source_out)
    target_out = target_out + eps / target_out.size(1)
    target_out = target_out / target_out.sum(1).view(-1, 1).expand_as(target_out)
    ce = -(source_out * target_out.log()).sum(1)
    if size_average:
        ce = ce.mean()
    return ce


def binary_cross_entropy_w_sigmoid(outputs, targets):
    """
    Calculates cross-entropy with sigmoid 
    outputs: prediction output from target model
    target: prediction output from source model

    iCaRL method uses this function to calculate the distillation loss.
    """

    out = torch.sigmoid(outputs)
    tar = torch.sigmoid(targets)

    ce = sum(torch.nn.functional.binary_cross_entropy(out[:, y], tar[:, y]) for y in range(tar.size()[1]))
    return ce


def feature_distillation_l2(target_feature, source_feature):
    feature_kd_loss = torch.dist(target_feature, source_feature, 2)
    return feature_kd_loss


def sim_matrix(a, b, eps=1e-8):
    """
    Batch cosine similarity taken from https://stackoverflow.com/a/58144658/10425618
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt



def feature_distillation_csc(target_feature, source_feature, targets, loss_offset=0):
    # cross-space clustering 
    targets_unsqueezed = targets.unsqueeze(1)
    indexes = (targets_unsqueezed == targets_unsqueezed.T).to(torch.int)
    indexes[indexes == 0] = -1
    computed_similarity = sim_matrix(target_feature, source_feature).flatten()
    csc_loss = 1 - computed_similarity      
    csc_loss *= indexes.flatten()
    csc_loss = loss_offset + csc_loss.mean()

    return csc_loss


def feature_distillation_ct(target_feature, source_feature, targets, num_old_cls, ct_temperature=2):
    # Controlled Transfer - need old class exemplars 
    source_feature_curtask = source_feature[targets < num_old_cls]
    source_feature_prevtask = source_feature[targets >= num_old_cls]
    target_feature_curtask = target_feature[targets < num_old_cls]
    target_feature_prevtask = target_feature[targets >= num_old_cls]
    previous_model_similarities = sim_matrix(source_feature_curtask, source_feature_prevtask)
    current_model_similarities = sim_matrix(target_feature_curtask, target_feature_prevtask)
    ct_loss = nn.KLDivLoss()(F.log_softmax(current_model_similarities/ct_temperature, dim=1), F.softmax(previous_model_similarities/ct_temperature, dim=1))   * (ct_temperature ** 2)

    return ct_loss
