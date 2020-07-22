
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import conv1d_layer_knnGraph, conv1d_resnet_block_knnGraph


# feature extraction for 3D and 2D points cloud
class FeatureExtractor(nn.Module):
  def __init__(self, config, in_channel):

    super(FeatureExtractor, self).__init__()

    activation = 'relu'
    idx_layer = 0
    self.numlayer = config.net_depth
    ksize = 1
    nchannel = config.net_nchannel
    act_pos = config.net_act_pos
    knn_nb = config.net_knn

    conv1d_block = conv1d_resnet_block_knnGraph
    # First convolution
    # just used to change the dim of in_chan to nchannel
    self.conv_in = conv1d_layer_knnGraph(
        in_channel = in_channel,
        out_channel = nchannel,
        ksize=ksize,
        activation=None,
        perform_bn=False,
        perform_gcn=False,
        nb_neighbors=knn_nb,
        act_pos="None"
    )
    # ResNet Knn graph
    for _ksize, _nchannel in zip([ksize] * self.numlayer, [nchannel] * self.numlayer):
      setattr(self, 'conv_%d' % idx_layer, conv1d_block(
          in_channel = nchannel,
          out_channel = nchannel,
          ksize=_ksize,
          activation=activation,
          perform_bn=config.net_batchnorm,
          perform_gcn=config.net_gcnorm,
          nb_neighbors=knn_nb,
          act_pos=act_pos
      ))

      idx_layer += 1


  def forward(self, x):
    x = self.conv_in(x)
    for i in range(self.numlayer):
      x = getattr(self, 'conv_%d' % i)(x)
    return x

# calculate the pairwise distance for 3D and 2D features
def pairwiseL2Dist(x1, x2):
    """ Computes the pairwise L2 distance between batches of feature vector sets
    res[..., i, j] = ||x1[..., i, :] - x2[..., j, :]||
    since 
    ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a^T*b

    Adapted to batch case from:
        jacobrgardner
        https://github.com/pytorch/pytorch/issues/15253#issuecomment-491467128
    """
    x1_norm2 = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm2 = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.baddbmm(
        x2_norm2.transpose(-2, -1),
        x1,
        x2.transpose(-2, -1),
        alpha=-2
    ).add_(x1_norm2).clamp_min_(1e-30).sqrt_()
    return res

# a small T-net to transform 3D points to a cano. direction
class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = torch.eye(3, dtype=x.dtype, device=x.device).unsqueeze(0)
        x = x.view(-1, 3, 3) + iden
        return x

# Sinkhorn to estimate the joint probability matrix P
class prob_mat_sinkhorn(torch.nn.Module):
    def __init__(self, config, mu=0.1, tolerance=1e-9, iterations=20):
        super(prob_mat_sinkhorn, self).__init__()
        self.config = config
        self.mu = mu  # the smooth term
        self.tolerance = tolerance  # don't change
        self.iterations = iterations  # max 30 is set, enough for a typical sized mat (e.g., 1000x1000)
        self.eps = 1e-12

    def forward(self, M, r=None, c=None):
        # r, c are the prior 1D prob distribution of 3D and 2D points, respectively
        # M is feature distance between 3D and 2D point
        K = (-M / self.mu).exp()
        # 1. normalize the matrix K
        K = K / K.sum(dim=(-2, -1), keepdim=True).clamp_min_(self.eps)

        # 2. construct the unary prior

        r = r.unsqueeze(-1)
        u = r.clone()
        c = c.unsqueeze(-1)

        i = 0
        u_prev = torch.ones_like(u)
        while (u - u_prev).norm(dim=-1).max() > self.tolerance:
            if i > self.iterations:
                break
            i += 1
            u_prev = u
            # update the prob vector u, v iteratively
            v = c / K.transpose(-2, -1).matmul(u).clamp_min_(self.eps)
            u = r / K.matmul(v).clamp_min_(self.eps)

        # assemble
        # P = torch.diag_embed(u[:,:,0]).matmul(K).matmul(torch.diag_embed(v[:,:,0]))
        P = (u * K) * v.transpose(-2, -1)
        return P


class KNNContextNormNet(nn.Module):
    def __init__(self, config):
        super(KNNContextNormNet, self).__init__()
        self.config = config
        self.in_channel_2d = 2 # normalize 2d points
        self.in_channel_3d = 3 # X,Y,Z coordinates of 3D points
        self.stn = STN3d() # a small T-net to transform 3D points to a cano. direction
        # feature extractors for 3D and 2D branch
        self.FeatureExtractor2d = FeatureExtractor(self.config, self.in_channel_2d)
        self.FeatureExtractor3d = FeatureExtractor(self.config, self.in_channel_3d)
        # calculate the pairwise distance for 3D and 2D features
        self.pairwiseL2Dist = pairwiseL2Dist
        # configurations for the estimation of joint probability matrix
        self.sinkhorn_mu = config.net_lambda
        self.sinkhorn_tolerance = 1e-9
        self.iterations = config.net_maxiter
        self.sinkhorn = prob_mat_sinkhorn(self.config, self.sinkhorn_mu, self.sinkhorn_tolerance, self.iterations)

    def forward(self, p3d, p2d):

        f3d = p3d
        f2d = p2d
        # Transform f3d to canonical coordinate frame:
        trans = self.stn(f3d.transpose(-2, -1)) # bx3x3
        f3d = torch.bmm(f3d, trans) # bxnx3
        # Extract features

        f2d = self.FeatureExtractor2d(f2d.transpose(-2,-1)).transpose(-2,-1) # b x m x 128
        f3d = self.FeatureExtractor3d(f3d.transpose(-2,-1)).transpose(-2,-1) # b x n x 128
        # L2 Normalise:
        f2d = torch.nn.functional.normalize(f2d, p=2, dim=-1)
        f3d = torch.nn.functional.normalize(f3d, p=2, dim=-1)
        # Compute pairwise L2 distance matrix:
        # row : 3d index; col : 2d index
        M = self.pairwiseL2Dist(f3d, f2d)

        # Sinkhorn to estimate the joint probability matrix P:
        b, m, n = M.size()
        r = M.new_zeros((b, m)) # bxm
        c = M.new_zeros((b, n)) # bxn
        for i in range(b):
            r[i, :] = 1.0 / m
            c[i, :] = 1.0 / n

        P = self.sinkhorn(M, r, c)
        return P










