# Copyright (c) SEU - PatternRec
# Licensed under the MIT License.
# Written by Sen Yang (yangsenius@seu.edu.cn)

import torch
import torchvision
import os
import numpy as np
import pprint


import yaml
from easydict import EasyDict as edict
import logging
import argparse
from timeit import default_timer as timer

from src.COCO_Dataset import cocodataset
from src.loss import MSELoss
from src.evaluate import evaluate
from src.models import keypoints_output_net

import datetime


def args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    parser.add_argument('--cfg',            help='experiment configure file name',  required=True,   default='config.yaml', type=str)
    parser.add_argument('--exp_name',       help='experiment name',        default='default_exp'     , type=str)

    args = parser.parse_args()
    return args

def logging_set(output_dir):

    logging.basicConfig(filename = os.path.join(output_dir,'test_{}.log'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))),
                    format = '%(asctime)s - %(name)s: L%(lineno)d - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    return logger

def main():

    arg = args()
    
    if not os.path.exists('output/{}'.format(arg.exp_name)):
        os.makedirs('output/{}'.format(arg.exp_name))
    output_dir = 'output/{}'.format(arg.exp_name)

    logger = logging_set(output_dir)
    logger.info('\n================ experient name:[{}] ===================\n'.format(arg.exp_name))
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.backends.cudnn.enabled = True
 
    config = edict( yaml.load( open(arg.cfg,'r')))

    logger.info('------------------------------ configuration ---------------------------')
    logger.info('\n==> available {} GPUs , numbers are {}\n'.format(torch.cuda.device_count(),os.environ["CUDA_VISIBLE_DEVICES"]))
    logger.info(pprint.pformat(config))
    logger.info('------------------------------- -------- ----------------------------')
    
    
    A = keypoints_output_net(config, is_train=True , num_layers = 50)
    logger.info(">>> total params of Model: {:.2f}M".format(sum(p.numel() for p in A.parameters()) / 1000000.0))

    if config.test.ckpt !='':
        logger.info('\n===> load ckpt in : '+ config.test.ckpt +'...')
        A.load_state_dict(torch.load(config.test.ckpt))
    elif os.path.exists(os.path.join(output_dir,'best_ckpt.tar')):
        logger.info('\n===> load ckpt in : '+ os.path.join(output_dir,'best_ckpt.tar'))
        A.load_state_dict(torch.load(os.path.join(output_dir,'best_ckpt.tar')))
    else:
        logger.info('\n===>no ckpt is found, use the initial model ...')
        raise ValueError

    A = torch.nn.DataParallel(A).cuda()
    

    valid_dataset = cocodataset(config,config.images_root_dir,
                            config.annotation_root_dir,
                            mode='val',
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                            ]))
    valid_dt_dataset =cocodataset(config,config.images_root_dir,
                            config.person_detection_results_path,
                            mode='dt',
                            dataset = 'val',
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                            ]))


    valid_dataloader = torch.utils.data.DataLoader(valid_dt_dataset, batch_size = config.test.batchsize, shuffle = False , num_workers = 4 , pin_memory=True )



    results = evaluate( A, valid_dataloader , config, output_dir)
    logger.info('map = {}'.format(results))
    

if __name__=='__main__':
    main()
