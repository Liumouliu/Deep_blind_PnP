import numpy as np
import os
import pickle
import torch
import torch.utils.data
from numpy.random import default_rng
import sys


def load_data_3d2d_pairs(config, dataset_split):
    """Main data loading routine"""
    print("loading the dataset {} ....\n".format(config.dataset))

    # the vars to be loaded
    # var_name_list = ["matches", "focal", "R", "t", "inlier_matches", "non_matchable_2D", "non_matchable_3D"]
    var_name_list = ["matches", "focal", "R", "t", "inlier_matches"]

    # check system python version
    # I use python2 to save the dataset
    if sys.version_info[0] == 3:
        print("You are using python 3.")

    encoding = "latin1"
    # Let's unpickle and save data
    data = {}
    # load the data
    cur_folder = "/".join([config.data_dir, config.dataset + "_" + dataset_split])
    for var_name in var_name_list:
        in_file_name = os.path.join(cur_folder, var_name) + ".pkl"
        with open(in_file_name, "rb") as ifp:
            if var_name in data:
                if sys.version_info[0] == 3:
                    data[var_name] += pickle.load(ifp, encoding=encoding)
                else:
                    data[var_name] += pickle.load(ifp)
            else:
                if sys.version_info[0] == 3:
                    data[var_name] = pickle.load(ifp, encoding=encoding)
                else:
                    data[var_name] = pickle.load(ifp)

    print("[Done] loading the {} dataset of  {} ....\n".format(dataset_split, config.dataset))

    return data



class Dataset(torch.utils.data.Dataset):
    def __init__(self, phase, config, batch_size):

        self.phase = phase
        self.config = config
        self.batch_size = batch_size
        self.data = load_data_3d2d_pairs(config, phase)
        self.len = len(self.data["t"])

    def __getitem__(self, index):

        # input_3d_xyz
        input_3d_xyz = self.data["matches"][index][:, 2:5]
        # input_2d_xy, note 2D coordinates have been normalized using its camera intrinsic matrix
        input_2d_xy = self.data["matches"][index][:, 0:2]
        # the rotation matrix and translation vector
        Rs_tr = self.data["R"][index]
        ts_tr = self.data["t"][index]
        # focal length
        focal = self.data["focal"][index]
        # contains the inlier 3D-2D matching indexes
        input_match = self.data["inlier_matches"][index]
        # number of inlier matches
        numInliers = input_match.shape[0]

        if self.phase == "train":
            # padding to the required training number points
            matches = np.zeros([self.config.numPointsTrain, self.config.numPointsTrain], dtype=np.float32)
        else:
            matches = np.zeros([input_3d_xyz.shape[0], input_2d_xy.shape[0]], dtype=np.float32)
            # fill the inlier matching mask
            matches[input_match[:, 0], input_match[:, 1]] = 1.0

            return matches, numInliers, input_3d_xyz.astype(np.float32), input_2d_xy.astype(np.float32), Rs_tr.astype(np.float32), ts_tr[None].reshape(3, 1).astype(np.float32), np.asarray(focal, dtype=np.float32)

        # ------------------------------------------
        # padding for training
        # if the number of inlier correspondences is larger than numPointsTrain, randomly select numPointsTrain inliers

        selected_ind_x = np.zeros([self.config.numPointsTrain, 1], dtype=np.int32)
        selected_ind_y = np.zeros([self.config.numPointsTrain, 1], dtype=np.int32)

        if input_match.shape[0] >= self.config.numPointsTrain:
            # random select inlier matches
            idx = np.random.randint(input_match.shape[0], size=self.config.numPointsTrain)
            selected_ind_x[:self.config.numPointsTrain, 0] = input_match[idx, 0]
            selected_ind_y[:self.config.numPointsTrain, 0] = input_match[idx, 1]
            numInliers = self.config.numPointsTrain

        else:
            # less than the required number of training points, replicate
            repeat_times = self.config.numPointsTrain // input_match.shape[0] + 1
            repeat_mat = np.tile(input_match, (repeat_times, 1))
            selected_ind_x[:self.config.numPointsTrain, 0] = repeat_mat[:self.config.numPointsTrain, 0]
            selected_ind_y[:self.config.numPointsTrain, 0] = repeat_mat[:self.config.numPointsTrain, 1]
            numInliers = self.config.numPointsTrain

        # shuffle the inlier matches

        list_train = np.arange(self.config.numPointsTrain)

        shuffle_x_ind = np.random.permutation(list_train)
        itemindex_x, _ = self.overlap_mbk(shuffle_x_ind, list_train)
        shuffle_y_ind = np.random.permutation(list_train)
        itemindex_y, _ = self.overlap_mbk(shuffle_y_ind, list_train)

        # modify the inlier matches
        selected_ind_x[:, 0] = selected_ind_x[shuffle_x_ind, 0]
        selected_ind_y[:, 0] = selected_ind_y[shuffle_y_ind, 0]

        input_2d_xy = input_2d_xy[selected_ind_y[:, 0], :]
        input_3d_xyz = input_3d_xyz[selected_ind_x[:, 0], :]

        # fill the inlier matching mask
        matches[itemindex_x[:numInliers], itemindex_y[:numInliers]] = 1.0

        return matches, input_match.shape[0], input_3d_xyz.astype(np.float32), input_2d_xy.astype(np.float32), Rs_tr.astype(np.float32), ts_tr[None].reshape(3, 1).astype(np.float32), np.asarray(focal, dtype=np.float32)

    def __len__(self):
        return self.len

    def overlap_mbk(self, a, b):

        a1 = np.argsort(a)
        b1 = np.argsort(b)
        # use searchsorted:
        sort_left_a = a[a1].searchsorted(b[b1], side='left')
        sort_right_a = a[a1].searchsorted(b[b1], side='right')
        #
        sort_left_b = b[b1].searchsorted(a[a1], side='left')
        sort_right_b = b[b1].searchsorted(a[a1], side='right')

        # # which values are in b but not in a?
        # inds_b=(sort_right_a-sort_left_a == 0).nonzero()[0]
        # # which values are in b but not in a?
        # inds_a=(sort_right_b-sort_left_b == 0).nonzero()[0]

        # which values of b are also in a?
        inds_b = (sort_right_a - sort_left_a > 0).nonzero()[0]
        # which values of a are also in b?
        inds_a = (sort_right_b - sort_left_b > 0).nonzero()[0]

        return a1[inds_a], b1[inds_b]



def make_data_loader(config, phase, batch_size, num_threads=0, shuffle=None):

  assert phase in ['train', 'valid']
  # only shuffle in training
  if shuffle is None:
    shuffle = (phase != 'valid')

  dset = Dataset(phase, config=config, batch_size=batch_size)

  loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=num_threads, pin_memory=False, drop_last=True)

  return loader


"""uni-test"""

from config import get_config

if __name__ == '__main__':

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    config = get_config()

    config.dataset = "megaDepth"

    train_loader = make_data_loader(config, config.train_phase, config.train_batch_size, num_threads=config.train_num_thread)
    train_data_loader_iter = train_loader.__iter__()
    train_input_dict = train_data_loader_iter.next()

    val_loader = make_data_loader(config, config.val_phase, 1, num_threads=config.val_num_thread)
    val_data_loader_iter = val_loader.__iter__()
    val_input_dict = val_data_loader_iter.next()
