
import torch
import torch.nn.functional as F

def correspondenceProbabilityDistances(P, C):
    """ Difference between the probability mass assigned to inlier and
        outlier correspondences
    """
    return ((1.0 - 2.0 * C) * P).sum(dim=(-2, -1))

def correspondenceProbabilityBCE(P, C):
    """ BCE loss
    """
    num_pos = F.relu(C.sum(dim=(-2,-1))-1.0) + 1.0
    num_neg = F.relu((1.0- C).sum(dim=(-2,-1)) -1.0) + 1.0

    loss = ((P + 1e-20).log() * C).sum(dim=(-2,-1)) * 0.5 / num_pos
    loss += ((1.0 - P + 1e-20 ).log() * (1.0 - C)).sum(dim=(-2,-1)) * 0.5 / num_neg

    return -loss


def correspondenceLoss(P, C_gt):
    # Using precomputed C_gt
    return correspondenceProbabilityBCE(P, C_gt).mean() # [-1, 1)
#     return correspondenceProbabilityDistances(P, C_gt).mean() # [-1, 1)

class TotalLoss(torch.nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()
    def forward(self, P, C_gt):
        loss = correspondenceLoss(P, C_gt).view(1)
        return loss


