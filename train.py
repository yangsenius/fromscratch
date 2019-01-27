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
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--cfg',            help='experiment configure file name',  required=True,   default='config.yaml', type=str)
    parser.add_argument('--exp_name',       help='experiment name',        default='default_train_baseline'     , type=str)
    parser.add_argument('--gpu',            help='gpu ids',                 default = '0,1',                      type =str)
    
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
    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu
    torch.backends.cudnn.enabled = True
 
    config = edict( yaml.load( open(arg.cfg,'r')))

    # Baseline model does not need mask information
    config.model.only_add_mask_channel = False
    config.model.extra_mask_module = False
    
    logger.info('------------------------------ configuration ---------------------------')
    logger.info('\n==> available {} GPUs , numbers are {}\n'.format(torch.cuda.device_count(),os.environ["CUDA_VISIBLE_DEVICES"]))
    logger.info(pprint.pformat(config))
    logger.info('------------------------------- -------- ----------------------------')

    A = keypoints_output_net(config, is_train=True , num_layers = 50)
    A = torch.nn.DataParallel(A).cuda()
    logger.info(">>> total params of Model: {:.2f}M".format(sum(p.numel() for p in A.parameters()) / 1000000.0))
    

    loss = MSELoss()

    train_dataset = cocodataset( config, config.images_root_dir,
                            config.annotation_root_dir,
                            mode='train',
                            augment = config.train.augmentation,
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


    train_dataloader_init = torch.utils.data.DataLoader(train_dataset, 
                                                    batch_size = config.train.batchsize, 
                                                    shuffle = True , 
                                                    num_workers = 4 , 
                                                    pin_memory=True )

    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, 
                                                    batch_size = config.test.batchsize, 
                                                    shuffle = False , 
                                                    num_workers = 4 , 
                                                    pin_memory=True )

    train_dataset.augmentation_reset(aug_occlusion=False)

    train_dataloader_no_occ =  torch.utils.data.DataLoader(train_dataset, 
                                                            batch_size = config.train.batchsize, 
                                                            shuffle = True , 
                                                            num_workers = 4 , 
                                                            pin_memory=True )

    begin, end = config.train.epoch_begin, config.train.epoch_end

    optimizer = torch.optim.Adam(A.parameters(), lr = config.train.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size = config.train.lr_step_size, 
                                                gamma = config.train.lr_decay_gamma )
                                                
    best = 0
    logger.info("\n=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+= training +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+==")
    for epoch in range(begin, end):
        logger.info('==>training...')
        scheduler.step()

        if epoch > 20:
            # random occlusion begin from epoch 20
            train_dataloader = train_dataloader_init
        else:
            train_dataloader = train_dataloader_no_occ

        for iters, (input, heatmap_gt, kpt_visible,  info) in enumerate(train_dataloader):
            
            start = timer()
            optimizer.zero_grad()

            input = input.cuda()
            heatmap_gt = heatmap_gt.cuda()
            kpt_visible = kpt_visible.float().cuda()

            A = A.train()

            heatmap_dt, _ = A(input)

            backward_loss , losses = loss(heatmap_dt, heatmap_gt, kpt_visible)

            backward_loss.backward()

            with open(os.path.join(output_dir,'train_loss.log'),'w') as ff:
                ff.write(str(np.round(backward_loss.item(),2))+',')

            optimizer.step()
            time = timer() - start

            if iters % 100 == 0:

               logger.info('epoch: {}\t iters:[{}|{}]\t loss:{:.4f}\t feed-speed:{:.2f} samples/s'.format(epoch,iters,len(train_dataloader),backward_loss,len(input)/time))
        

        # evalute
        eval_results = evaluate( A, valid_dataloader , config, output_dir)

        logger.info('==> mAP is: {:.3f}\n'.format(eval_results[0]))
        torch.save(A.module.state_dict(),os.path.join(output_dir,'ckpt.tar'))

        if  eval_results[0] > best :
            best = eval_results[0]
            torch.save(A.module.state_dict(),os.path.join(output_dir,'best_ckpt.tar'))
            logger.info('\n!(^ 0 ^)! New Best mAP = {} and all oks metrics is {} \n'.format(best,eval_results))
            


if __name__ == '__main__':
    main()
