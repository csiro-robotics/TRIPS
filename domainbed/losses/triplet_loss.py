import torch
from torch import nn
import domainbed.networks as networks


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    x_norm = torch.norm(x, dim=1)
    x_norm_mean = torch.mean(x_norm)
    y_norm = torch.norm(y, dim=1)
    y_norm_mean = torch.mean(y_norm)

    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    # dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection/(x_norm * y_norm)
    dist = 1. - dist
    return dist


def domain_hard_sample_mining(dist_mat, labels, domains, f2p_dismat=None):
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    is_domain_pos = domains.expand(N,N).eq(domains.expand(N, N).t())
    is_domain_neg = domains.expand(N,N).ne(domains.expand(N,N).t())
    is_label_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_label_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    domainpos_labelneg = is_domain_pos & is_label_neg  # negative pair 
    domainneg_labelpos = is_domain_neg & is_label_pos  # positive pair

    dist_dpln, dist_dnlp =[], []
    for i in range(N):
        if dist_mat[i][domainneg_labelpos[i]].shape[0] != 0:
            dist_dnlp.append(torch.max(dist_mat[i][domainneg_labelpos[i]].contiguous(), 0, keepdim=True)[0])
        else:
            dist_dnlp.append(torch.zeros(1).cuda())
        if dist_mat[i][domainpos_labelneg[i]].shape[0] != 0:
            a = torch.min(dist_mat[i][domainpos_labelneg[i]].contiguous(), 0, keepdim=True)[0]
            if f2p_dismat is not None:
                b = torch.min(f2p_dismat[i].contiguous(),0,keepdim=True)[0]
            else:
                b=a
            dist_dpln.append(torch.min(a,b))
        else:
            if f2p_dismat is None:
                dist_dpln.append(torch.zeros(1).cuda())
            else:
                dist_dpln.append(torch.min(f2p_dismat[i].contiguous(),0,keepdim=True)[0])

    dist_dnlp = torch.cat(dist_dnlp).clamp(min=1e-12,max=1e9)
    dist_dpln = torch.cat(dist_dpln).clamp(min=1e-12,max=1e9)
    return dist_dnlp, dist_dpln


class DomainTripletLoss(object):
    def __init__(self, hparams, margin=None, hard_factor=0.0, feature_output=2048, dist_type='cosine_dist'):
        self.margin = margin
        self.hard_factor = hard_factor
        self.hparams = hparams

        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()
        if dist_type == 'cosine_dist' or dist_type == 'euclidean_dist':
            self.dist_type = dist_type
        else:
            raise RuntimeError('The dist_type must be cosine_dist or euclidean_dist.')
    
    def __call__(self, global_feat, labels, domain_labels, old_prototype=None):
        
        # global_feat_norm = torch.norm(global_feat, dim=1)
        # global_feat_norm_mean = torch.mean(global_feat_norm)
        # print('DomainTripletLoss | global_feat_norm_mean: {0}'.format(global_feat_norm_mean))
        
        if self.dist_type == 'cosine_dist':
            dist_mat = cosine_dist(global_feat, global_feat)
        else:   # euclidean_dist 
            dist_mat = euclidean_dist(global_feat, global_feat)
        if not old_prototype == None:
            if self.dist_type == 'cosine_dist':
                f2p_dismat = cosine_dist(global_feat, old_prototype)
            else:  # euclidean_dist
                f2p_dismat = euclidean_dist(global_feat, old_prototype)
        else:
            f2p_dismat = None
  
        dist_ap, dist_an = domain_hard_sample_mining(dist_mat, labels, domain_labels, f2p_dismat=f2p_dismat)

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)

        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)

        return loss, dist_mat