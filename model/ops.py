import torch
import torch.nn as nn

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=10, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous() # (batch_size, num_dims*2, num_points, k)
    return feature


def bn_act(in_channel, perform_gcn, perform_bn, activation):

    """
    # Global Context normalization on the input
    """
    layers = []
    if perform_gcn:
      layers.append(gcn(in_channel))

    if perform_bn:
      layers.append(nn.BatchNorm1d(in_channel, affine=False))

    if activation == 'relu':
      layers.append(torch.nn.ReLU())

    return layers


def conv1d_layer_knnGraph(in_channel, out_channel, ksize, activation, perform_bn, perform_gcn, nb_neighbors=10, act_pos="post"):
    assert act_pos == "pre" or act_pos == "post" or act_pos == "None"
    layers = []
    # If pre activation
    if act_pos == "pre":
        new = bn_act(in_channel, perform_gcn, perform_bn, activation)
        for l in new:
            layers.append(l)


    if nb_neighbors > 0:
        # get the knn graph features
        layers.append(knn_feature(nb_neighbors, in_channel, out_channel, ksize))
    else:
        # no knn graph here, only MLP at per-point
        layers.append(torch.nn.Conv1d(in_channel, out_channel, ksize))

    # If post activation
    if act_pos == "post":

        new = bn_act(out_channel, perform_gcn, perform_bn, activation)
        for l in new:
            layers.append(l)

    return nn.Sequential(*layers)




class gcn(nn.Module):
    def __init__(self, in_channel):
        super(gcn, self).__init__()
        pass

    def forward(self, x):
        # x: [n, c, K]
        var_eps = 1e-3
        m = torch.mean(x, 2, keepdim=True)
        v = torch.var(x, 2, keepdim=True)
        inv = 1. / torch.sqrt(v + var_eps)
        x = (x - m) * inv
        return x

class knn_feature(nn.Module):
    def __init__(self, nb_neighbors, in_channel, out_channel, ksize):
        super(knn_feature, self).__init__()
        self.k = nb_neighbors
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.ksize = ksize
        self.conv = torch.nn.Conv2d(in_channel*2, out_channel, ksize)

    def forward(self, x):
        # x: [n, c, K]
        x = get_graph_feature(x, k=self.k)
        # conv
        x = self.conv(x)
        # avg pooling
        x = x.mean(dim=-1, keepdim=False)

        return x


class conv1d_resnet_block_knnGraph(nn.Module):
  def __init__(self, in_channel, out_channel, ksize, activation, perform_bn=False, perform_gcn=False, nb_neighbors=10, act_pos="post"):

    super(conv1d_resnet_block_knnGraph, self).__init__()

    # Main convolution
    self.conv1 = conv1d_layer_knnGraph(
      in_channel=in_channel,
      out_channel=out_channel,
      ksize=ksize,
      activation=activation,
      perform_bn=perform_bn,
      perform_gcn=perform_gcn,
      nb_neighbors=nb_neighbors,
      act_pos = act_pos
    )

    # Main convolution
    self.conv2 = conv1d_layer_knnGraph(
      in_channel=out_channel,
      out_channel=out_channel,
      ksize=ksize,
      activation=activation,
      perform_bn=perform_bn,
      perform_gcn=perform_gcn,
      nb_neighbors=nb_neighbors,
      act_pos=act_pos
    )

  def forward(self, x):
    xorg = x
    x = self.conv1(x)
    x = self.conv2(x)
    return x + xorg

