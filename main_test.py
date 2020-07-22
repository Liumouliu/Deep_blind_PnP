# test

import os
import numpy as np
import cv2
import logging
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from config import get_config
from lib.data_loaders import make_data_loader
from lib.utils import load_model
from lib.timer import *
from lib.transformations import quaternion_from_matrix


ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

logging.basicConfig(level=logging.INFO, format="")


def recalls(eval_res):
    ret_val = []
    ths = np.arange(7) * 5
    cur_err_q = np.array(eval_res["err_q"]) * 180.0 / np.pi
    # Get histogram
    q_acc_hist, _ = np.histogram(cur_err_q, ths)
    num_pair = float(len(cur_err_q))
    q_acc_hist = q_acc_hist.astype(float) / num_pair
    q_acc = np.cumsum(q_acc_hist)
    # Store return val
    ret_val += [np.mean(q_acc[:4])]
    ret_val += [np.median(cur_err_q)]
    ret_val += [np.median(eval_res["err_t"])]
    ret_val += [np.mean(eval_res["inlier_ratio"])]

    return ret_val


def evaluate_R_t( R_gt, t_gt, R_est, t_est, q_gt=None):

    t = t_est.flatten()
    t_gt = t_gt.flatten()
    eps = 1e-15

    if q_gt is None:
      q_gt = quaternion_from_matrix(R_gt)
    q = quaternion_from_matrix(R_est)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt) ** 2))
    err_q = np.arccos(1 - 2 * loss_q)
    # absolute distance error on t
    err_t = np.linalg.norm(t_gt - t)
    if np.sum(np.isnan(err_q)) or np.sum(np.isnan(err_t)):
      # This should never happen!
      err_q = np.pi
      err_t = np.pi / 2
    return err_q, err_t


def generate_outliers_synthetic(p2d, p3d, nb_outlier_2d, nb_outlier_3d):
    # add synthetic outliers
    # Get bounding boxes
    bb2d_min = p2d.min(dim=-2)[0]
    bb2d_width = p2d.max(dim=-2)[0] - bb2d_min
    bb3d_min = p3d.min(dim=-2)[0]
    bb3d_width = p3d.max(dim=-2)[0] - bb3d_min

    p2d_outlier = bb2d_width * torch.rand((p2d.size(0), nb_outlier_2d, p2d.size(-1)), dtype=p2d.dtype, layout=p2d.layout, device=p2d.device) + bb2d_min
    p3d_outlier = bb3d_width * torch.rand((p3d.size(0), nb_outlier_3d, p3d.size(-1)), dtype=p3d.dtype, layout=p3d.layout, device=p3d.device) + bb3d_min

    return p2d_outlier, p3d_outlier


def generate_outliers_real(p2d, p3d, nb_outlier_2d, nb_outlier_3d):
    # add real outliers
    nb_2d_min, nb_3d_min = min(nb_outlier_2d, p2d.shape[0]), min(nb_outlier_3d, p3d.shape[0])
    return p2d[:nb_2d_min,:], p3d[:nb_3d_min,:]

# main function

def main(config):
    # loading the test data
    val_data_loader = make_data_loader(configs, "valid", 1, num_threads=configs.val_num_thread, shuffle=False)

    # no gradients
    with torch.no_grad():
        # Model initialization
        Model = load_model(config.model)
        model = Model(config)

        # limited GPU
        if config.gpu_inds > -1:
            torch.cuda.set_device(config.gpu_inds)
            device = torch.device('cuda', config.gpu_inds)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model.to(device)

        # load the weights
        if config.weights:
            checkpoint = torch.load(config.weights)
            model.load_state_dict(checkpoint['state_dict'])

        logging.info(model)

        # evaluation model
        model.eval()

        num_data = 0
        data_timer, matching_timer = Timer(), Timer()
        tot_num_data = len(val_data_loader.dataset)

        data_loader_iter = val_data_loader.__iter__()

        # collecting the errors in rotation, errors in tranlsation, num of inliers, inlier ratios
        measure_list = ["err_q", "err_t", "inlier_ratio"]
        nb_outlier_ratios = config.outlier_ratios.shape[0]
        eval_res = {}
        for measure in measure_list:
            eval_res[measure] = np.zeros((nb_outlier_ratios, tot_num_data))

        # for gpu memory consideration
        max_nb_points = 20000

        for batch_idx in range(tot_num_data):

            data_timer.tic()
            matches, numInliers, input_3d_xyz, input_2d_xy, R_gt, t_gt, _ = data_loader_iter.next()
            data_timer.toc()

            p3d_cur = input_3d_xyz
            p2d_cur = input_2d_xy
            matches_cur = matches

            # for each outlier_ratio level
            for oind in range(nb_outlier_ratios):
                outlier_ratio = config.outlier_ratios[oind]
                if outlier_ratio > 0:
                    # adding outliers
                    nb_added_outlier_2d = min(int(round(numInliers.item() * outlier_ratio)), max_nb_points - numInliers.item())
                    nb_added_outlier_3d = min(int(round(numInliers.item() * outlier_ratio)), max_nb_points - numInliers.item())

                    if config.outlier_type == "synthetic":
                        # add synthetic outliers
                        p2d_outlier, p3d_outlier = generate_outliers_synthetic(input_2d_xy, input_3d_xyz, nb_added_outlier_2d, nb_added_outlier_3d)
                    elif config.outlier_type == "real":
                        # add real outliers, you can load real-outlier from the non-matchable-2d/3d.pkl
                        raise NameError('Please modify dataloader')
                        # p2d_outlier, p3d_outlier = generate_outliers_real(p2d_outlier_all, p3d_outlier_all, nb_added_outlier_2d, nb_added_outlier_3d)
                    else:
                        raise NameError('Invalid outlier type')

                    # padding the outliers
                    p2d_cur = torch.cat((input_2d_xy, p2d_outlier), -2)
                    p3d_cur = torch.cat((input_3d_xyz, p3d_outlier), -2)

                    # Fill ground-truth matching indexes with outliers
                    b = p2d_cur.size(0)
                    m = p3d_cur.size(-2)
                    n = p2d_cur.size(-2)
                    matches_cur = matches.new_full((b,m,n),0.0)
                    matches_cur[:,:numInliers.item(),:numInliers.item()] = matches

                # print([p3d_cur.size(1), p2d_cur.size(1)])

                p2d_cur, p3d_cur, R_gt, t_gt, matches_cur = p2d_cur.to(device), p3d_cur.to(device), R_gt.to(
                    device), t_gt.to(device), matches_cur.to(device)

                # Compute output
                matching_timer.tic()
                prob_matrix = model(p3d_cur, p2d_cur)
                matching_timer.toc()

                # compute the topK correspondences
                # note cv2.solvePnPRansac is not stable, sometimes wrong!
                # please use https://github.com/k88joshi/posest, and cite the original paper
                k = min(2000, round(p3d_cur.size(1) * p2d_cur.size(1)))  # Choose at most 2000 points in the testing stage
                _, P_topk_i = torch.topk(prob_matrix.flatten(start_dim=-2), k=k, dim=-1, largest=True, sorted=True)
                p3d_indices = P_topk_i / prob_matrix.size(-1)  # bxk (integer division)
                p2d_indices = P_topk_i % prob_matrix.size(-1)  # bxk
                # let's check the inliner ratios within the topK matches
                # retrieve the inlier/outlier 1/0 logit
                inlier_inds = matches_cur[:, p3d_indices, p2d_indices].cpu().numpy()
                # inlier ratio
                inlier_ratio = np.sum(inlier_inds) / k * 100.0

                # in case cannot be estimated
                err_q = np.pi
                err_t = np.inf
                # more than 5 2D-3D matches
                if k > 5:
                    # compute the rotation and translation error
                    K = np.float32(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
                    dist_coeff = np.float32(np.array([0.0, 0.0, 0.0, 0.0]))
                    #  RANSAC p3p
                    p2d_np = p2d_cur[0, p2d_indices[0, :k], :].cpu().numpy()
                    p3d_np = p3d_cur[0, p3d_indices[0, :k], :].cpu().numpy()
                    retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                        p3d_np, p2d_np, K, dist_coeff,
                        iterationsCount=1000,
                        reprojectionError=0.01,
                        flags=cv2.SOLVEPNP_P3P)
                    if rvec is not None and tvec is not None:
                        R_est, _ = cv2.Rodrigues(rvec)
                        t_est = tvec
                        err_q, err_t = evaluate_R_t(R_est, t_est, R_gt[0, :, :].cpu().numpy(), t_gt.cpu().numpy())

                torch.cuda.empty_cache()

                eval_res["err_q"][oind, batch_idx] = err_q
                eval_res["err_t"][oind, batch_idx] = err_t
                eval_res["inlier_ratio"][oind, batch_idx] = inlier_ratio

                logging.info(' '.join([
                    f"Validation iter {num_data} / {tot_num_data} : Data Loading Time: {data_timer.avg:.3f},",
                    f"outlier ratio: {outlier_ratio:.3f},",
                    f"Matching Time: {matching_timer.avg:.3f},",
                    f"err_rot: {err_q:.3f}, err_t: {err_t:.3f}, inlier_ratio: {inlier_ratio:.3f},",
                ]))
                data_timer.reset()

            num_data += 1

        # after checking all the validation samples, let's calculate statistics

        # for each outlier_ratio level
        for oind in range(nb_outlier_ratios):
            outlier_ratio = config.outlier_ratios[oind]
            eval_res_cur = {}
            for measure in measure_list:
                eval_res_cur[measure] = eval_res[measure][oind,:]

            recall = recalls(eval_res_cur)

            logging.info(' '.join([
                f" outlier_ratio: {outlier_ratio:.3f}, recall_rot: {recall[0]:.3f}, med. rot. : {recall[1]:.3f}, med. trans. : {recall[2]:.3f}, avg. inlier ratio: {recall[3]:.3f},",
            ]))



if __name__ == '__main__':

    configs = get_config()

    configs.gpu_inds = 1
    # dataset dir
    # configs.data_dir = "/media/liu/data"
    # the used dataset
    configs.dataset = "megaDepth"
    # configs.dataset = "modelnet40"
    # pre-trained networks
    configs.debug_nb = "preTrained"

    # configs.outlier_type = "real"
    configs.outlier_type = "synthetic"
    configs.outlier_ratios = np.linspace(0.0, 0.0, num=1, dtype=np.float32)
    # configs.outlier_ratios = np.linspace(0.0, 1.0, num=11, dtype=np.float32)

    configs.weights = os.path.join(configs.out_dir, configs.dataset, configs.debug_nb) + '/best_val_checkpoint.pth'

    main(configs)







