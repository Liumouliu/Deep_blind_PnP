# configuration of the method

import argparse

arg_lists = []
parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ("true", "1")

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# -----------------------------------------------------------------------------
# Logging
logging_arg = add_argument_group('Logging')
logging_arg.add_argument('--out_dir', type=str, default='output', help="base directory for models")
# logging_arg.add_argument('--resume_dir', type=str, default='output', help='path to latest checkpoint')
logging_arg.add_argument('--resume_dir', type=str, default=None, help='path to latest checkpoint')
logging_arg.add_argument('--weights', type=str, default="", help='path to preTrained weights')
logging_arg.add_argument('--debug_nb', type=str, default="1", help='path to latest checkpoint')
# -----------------------------------------------------------------------------
# Network
net_arg = add_argument_group("Network")
net_arg.add_argument('--model', type=str, default='KNNContextNormNet')
net_arg.add_argument("--net_depth", type=int, default=12, help="number of layers")
net_arg.add_argument("--net_nchannel", type=int, default=128, help="number of channels in a layer")
net_arg.add_argument("--net_act_pos", type=str, default="post", choices=["pre", "mid", "post"], help="where the activation should be in case of resnet")
net_arg.add_argument("--net_gcnorm", type=str2bool, default=True, help="whether to use context normalization for each layer")
net_arg.add_argument("--net_batchnorm", type=str2bool, default=True, help="whether to use batch normalization")
net_arg.add_argument("--net_topK", type=int, default=2000, help="how many matches you want to obtain?")
net_arg.add_argument("--net_lambda", type=float, default=0.1, help="lambda in optimal transport")
net_arg.add_argument("--net_maxiter", type=int, default=30, help="the maximum number of iteration in Sinkhorn")
net_arg.add_argument("--net_knn", type=int, default=10, help="number of nearest neighbors in Knn graph")
net_arg.add_argument('--best_val_metric', type=str, default='avg_inlier_ratio')
# -----------------------------------------------------------------------------
# Data

data_arg = add_argument_group("Data")
data_arg.add_argument("--data_dir", type=str, default="/media/liu/data/PAMI/Data", help="dir of dataset")
data_arg.add_argument("--dataset", type=str, default="modelnet40", help="used dataset")

# Data loader configs
data_arg.add_argument('--train_phase', type=str, default="train")
data_arg.add_argument('--val_phase', type=str, default="valid")

# -----------------------------------------------------------------------------
# Training
train_arg = add_argument_group("Train")
train_arg.add_argument("--train_batch_size", type=int, default=12, help="batch size")
train_arg.add_argument("--numPointsTrain", type=int, default=1000, help="the number of maximum 2D/3D points in training (For fast training and GPU memory)")
train_arg.add_argument("--train_lr", type=float, default=1e-3, help="learning rate")
train_arg.add_argument('--train_weight_decay', type=float, default=1e-3)
train_arg.add_argument('--train_momentum', type=float, default=0.8)
train_arg.add_argument('--exp_gamma', type=float, default=0.99)
train_arg.add_argument("--train_epoches", type=int, default=100, help="maximum training iterations to perform")
train_arg.add_argument("--train_start_epoch", type=int, default=0, help="the starting epoch to train")
train_arg.add_argument("--train_save_freq_epoch", type=int, default=1, help="saving fps")
train_arg.add_argument('--iter_size', type=int, default=1)


train_arg.add_argument('--val_max_iter', type=int, default=-1)
train_arg.add_argument('--val_epoch_freq', type=int, default=1)
train_arg.add_argument('--optimizer', type=str, default='Adam')
train_arg.add_argument('--train_num_thread', type=int, default=3)
train_arg.add_argument('--train_seed', type=int, default=0)
train_arg.add_argument('--use_gpu', type=bool, default=True)
train_arg.add_argument('--gpu_inds', type=int, default=0)
train_arg.add_argument('--val_num_thread', type=int, default=1)
train_arg.add_argument('--test_num_thread', type=int, default=1)
train_arg.add_argument("--print_freq", type=int, default=10, help="print fps")

# -----------------------------------------------------------------------------
# test
test_arg = add_argument_group("Test")
test_arg.add_argument("--test_flag", type=bool, default=False, help="Enable testing")
test_arg.add_argument("--test_outlier_ratio", type=float, default=0.0, help="the outlier ratio in testing")


def get_config():
  args = parser.parse_args()
  return args


def print_usage():
    parser.print_usage()

