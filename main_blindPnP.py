import os
import random
from easydict import EasyDict as edict
import json
import logging
import sys
import torch.backends.cudnn as cudnn
import torch.utils.data
from config import get_config
from lib.data_loaders import make_data_loader
from trainer import BlindPnPTrainer


# logging
ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

logging.basicConfig(level=logging.INFO, format="")



# main function
def main(configs):

    # train and validation dataloaders
    train_loader = make_data_loader(configs, "train", configs.train_batch_size, num_threads = configs.train_num_thread, shuffle=True)

    val_loader = make_data_loader(configs, "valid", 1, num_threads = configs.val_num_thread, shuffle=False)

    trainer = BlindPnPTrainer(configs, train_loader, val_loader)

    trainer.train()


if __name__ == '__main__':


    configs = get_config()
    # -------------------------------------------------------------
    """You can change the configurations here or in the file config.py"""
    # dataset dir
    # configs.data_dir = "/media/liu/data"
    # dataset used
    # "megaDepth", "modelnet40", "nyu_non_overlap"
    configs.dataset = "megaDepth"
    # 1e-3 for megaDepth; 1e-4 for modelnet40; 1e-4 for nyu_non_overlap
    configs.train_lr = 1e-3
    # select which GPU to be used
    configs.gpu_inds = 0
    # This is a debug number, set it to whatever you want
    configs.debug_nb = "preTrained"
    # training batch size
    configs.train_batch_size = 12

    # if your training is terminated unexpectly, uncomment the following line and set the resume_dir to continue
    # configs.resume_dir = 'output'
    # -------------------------------------------------------------
    dconfig = vars(configs)

    if configs.resume_dir:
        resume_config = json.load(open(configs.resume_dir + "/" + configs.dataset + "/" + configs.debug_nb + '/config.json', 'r'))
        for k in dconfig:
            if k in resume_config:
                dconfig[k] = resume_config[k]
        dconfig['resume'] = os.path.join(resume_config['out_dir'], resume_config['dataset'], configs.debug_nb) + '/checkpoint.pth'
    else:
        dconfig['resume'] = None

    # print the configurations
    logging.info('===> Configurations')
    for k in dconfig:
        logging.info('    {}: {}'.format(k, dconfig[k]))

    # Convert to dict
    configs = edict(dconfig)

    # set the seeds
    if configs.train_seed is not None:
        random.seed(configs.train_seed)
        torch.manual_seed(configs.train_seed)
        torch.cuda.manual_seed(configs.train_seed)
        cudnn.deterministic = True


    main(configs)







