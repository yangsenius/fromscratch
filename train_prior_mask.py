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
    parser = argparse.ArgumentParser(description='Train multi-keypoints peak mask network')
    parser.add_argument('--cfg',            help='experiment configure file name',  required=True,   default='config.yaml', type=str)
    parser.add_argument('--exp_name',       help='experiment name',        default='train_prior_mask'     , type=str)

    args = parser.parse_args()
    return args

def logging_set(output_dir):

    logging.basicConfig(filename = os.path.join(output_dir,'train_{}.log'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))),
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

    Prior = keypoints_output_net(config, is_train=True , num_layers = 34, heatmap_channels=1 )
    Prior = torch.nn.DataParallel(Prior).cuda()
    logger.info(">>> total params of Model: {:.2f}M".format(sum(p.numel() for p in Prior.parameters()) / 1000000.0))
    optimizer = torch.optim.Adam(Prior.parameters(), lr = config.train.lr)

    loss = MSELoss()

    train_dataset = cocodataset( config, config.images_root_dir,
                            config.annotation_root_dir,
                            mode='train',
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                            ]))

    valid_dataset = cocodataset(config,config.images_root_dir,
                            config.annotation_root_dir,
                            mode='val',
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                            ]))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = config.train.batchsize, shuffle = True , num_workers = 4 , pin_memory=True )
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size = config.test.batchsize, shuffle = False , num_workers = 4 , pin_memory=True )

    begin, end = config.train.epoch_begin, config.train.epoch_end
    

    logger.info("\n=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+= training +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+==")
    for epoch in range(begin, end):
        logger.info('==>training...')
        for iters, (input, _ , _ ,  info) in enumerate(train_dataloader):
            
            start = timer()
            optimizer.zero_grad()
            input = input.cuda()
            prior_mask_gt = info['prior_mask']
            prior_mask_gt = prior_mask_gt.cuda()
            
            Prior= Prior.train()
            prior_mask_dt, _ = Prior(input)
            
            backward_loss =  ((prior_mask_dt- prior_mask_gt)**2).sum(dim=3).sum(dim=2).sum(dim=1).mean(dim=0)

            backward_loss.backward()

            with open(os.path.join(output_dir,'train_loss.log'),'w') as ff:
                ff.write(str(np.round(backward_loss.item(),2))+',')

            optimizer.step()
            time = timer() - start

            if iters % 100 == 0:

               logger.info('epoch: {}\t iters:[{}|{}]\t loss:{:.4f}\t feed-speed:{:.2f} samples/s'.format(epoch,iters,len(train_dataloader),backward_loss,len(input)/time))
        
        for id, (input,_ ,_ ,_, _, info) in enumerate(valid_dataloader):
            start = timer()
            mask_gt = info['prior_mask']
            mask_gt = mask_gt.cuda()
            input = input.cuda()
            Prior= Prior.eval()
            mask_dt, _ = Prior(input)

            average_loss =  ((mask_dt- mask_gt)**2).sum(dim=3).sum(dim=2).sum(dim=1).mean(dim=0)

            if id % 50 == 0:
    
               logger.info('epoch: {}\t iters:[{}|{}]\t valid_loss:{:.4f}\t inference-speed:{:.2f} samples/s'.format(epoch,id,len(valid_dataloader),average_loss,len(input)/time))
        


        torch.save(Prior.state_dict(),os.path.join(output_dir,'ckpt.tar'))
        logger.info('keypoints Prior model is saved')

if __name__ == '__main__':
    main()
