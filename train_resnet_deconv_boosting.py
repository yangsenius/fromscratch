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
from src.meta_model import Kpt_and_Mask_Boosting_Net

import datetime

def args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--cfg',            help='experiment configure file name',  required=True,   default='config.yaml', type=str)
    parser.add_argument('--exp_name',       help='experiment name',        default='train_res_deconv_boosting'     , type=str)

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
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

    torch.backends.cudnn.enabled = True
 
    config = edict( yaml.load( open(arg.cfg,'r')))

    ######   we use extra information to boost the keypoints results
    #config.model.extra_mask_flag = True
    #config.model.extra_feature_flag = True

    A = Kpt_and_Mask_Boosting_Net(config, is_train=True , num_layers = 50)
    A = torch.nn.DataParallel(A).cuda()
   
     # print(A.module.boosting.stack_metaboosters[0].graph)
    logger.info('------------------------------- Model Struture ----------------------------')
    logger.info(A)

    logger.info('------------------------------ configuration ---------------------------')
    logger.info('\n==> available {} GPUs , numbers are {}\n'.format(torch.cuda.device_count(),os.environ["CUDA_VISIBLE_DEVICES"]))
    logger.info(pprint.pformat(config))
    logger.info('------------------------------- -------- ----------------------------')

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


    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = config.train.batchsize, shuffle = True , num_workers = 4 , pin_memory=True )
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size = config.test.batchsize, shuffle = False , num_workers = 4 , pin_memory=True )


    begin, end = config.train.epoch_begin, config.train.epoch_end

    # in `DataParallel` mode ,  all parameters of the `state_dict` are wraped in `module`
    optimizer_base = torch.optim.Adam(A.module.resnet_deconv.parameters(), lr = config.train.lr)
    optimizer_boosting = torch.optim.Adam(A.module.boosting.parameters(), lr = config.train.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_base, step_size=config.train.lr_step_size, 
                                                            gamma = config.train.lr_decay_gamma)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer_boosting, step_size=config.train.lr_step_size, 
                                                            gamma = config.train.lr_decay_gamma) 
                                                                                   
    best ,best_o= 0, 0
    result_thres = config.model.boosting_requirement
    
    logger.info("\n=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+= training +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+==")
    for epoch in range(begin, end):

        logger.info('==>training...')
        if best_o > result_thres:
            logger.info('best result of Baseline is {} >{} ,train Baseline and Metabooster '.format(best_o,result_thres))
        else:
            logger.info('best result of Baseline is {} <={},only train Baseline model '.format(best_o,result_thres))

        scheduler.step()
        scheduler2.step()
        
        for iters, (input, heatmap_gt, kpt_visible,  info) in enumerate(train_dataloader):
            
            start = timer()
            optimizer_base.zero_grad()
            optimizer_boosting.zero_grad()

            input = input.cuda()
            
            prior_mask_gt = info['prior_mask']
            prior_mask_gt = prior_mask_gt.cuda()

            heatmap_gt = heatmap_gt.cuda()
            
            kpt_visible = kpt_visible.float().cuda()

            A = A.train()

            heatmap_dt, prior_mask_dt, refine_heatmap_dt = A(input)

            heatmap_dt_loss , _ = loss(heatmap_dt, heatmap_gt, kpt_visible)
            mask_loss =  ((prior_mask_dt- prior_mask_gt)**2).sum(dim=3).sum(dim=2).sum(dim=1).mean(dim=0)
            
            mask_loss_weight = config.train.mask_loss_weight
            backward_loss = mask_loss_weight * mask_loss + heatmap_dt_loss 

            #print(heatmap_dt)
            #print(refine_heatmap_dt)

            # only when resnet_deconv has a better results,we begin to train the boosting part
            retain_graph_flag = True if best_o > result_thres else False
            backward_loss.backward(retain_graph = retain_graph_flag) # don't release the dynamic graph

            with open(os.path.join(output_dir,'train_loss.log'),'w') as ff:
                ff.write(str(np.round(backward_loss.item(),2))+',')

            optimizer_base.step()

            refine_heatmap_dt_loss = 0

            if best_o > result_thres:
                ## update boosting part parameters
                refine_heatmap_dt_loss, _ = loss( refine_heatmap_dt, heatmap_gt, kpt_visible)
                refine_heatmap_dt_loss.backward()
                optimizer_boosting.step()
                refine_heatmap_dt_loss = refine_heatmap_dt_loss.item()

            time = timer() - start

            mask_loss = mask_loss.item()
            heatmap_dt_loss = heatmap_dt_loss.item()

            #print('refine',A.module.boosting.stack_metaboosters[0].mini_encoder_decoder.data)
            #print('mask,')

            if iters % 100 == 0:

               logger.info('epoch: {}\t iters:[{}|{}]\t kpt_loss:{:.4f}\t  mask_loss:{:.4f}\t refine_kpt_loss:{:.4f}\t feed-speed:{:.2f} samples/s'
                .format( epoch,
                        iters,
                        len(train_dataloader),
                        heatmap_dt_loss,
                        mask_loss,
                        refine_heatmap_dt_loss,
                        len(input)/time))
        
        ## eval orignal heatmaps of keypoints##
        logger.info('==> eval original heatmaps of keypoints')
        eval_results = evaluate( A, valid_dataloader , config, output_dir, with_mask = True)

        ## eval refine heatmaps of keypoints
        logger.info('==> eval refine heatmaps of keypoints')
        eval_results_refine = evaluate( A, valid_dataloader , config, output_dir, with_mask = True
                                                                                , use_refine = True)

        logger.info('==> original mAP is: {:.3f}\n'.format(eval_results[0]))
        logger.info('==> refine   mAP is: {:.3f}\n'.format(eval_results_refine[0]))

        torch.save(A.module.state_dict(),os.path.join(output_dir,'ckpt.tar'))

        if  eval_results[0] > best_o :
            best_o = eval_results[0]
            torch.save(A.module.resnet_deconv.state_dict(),os.path.join(output_dir,'best_res_deconv_ckpt.tar'))
            logger.info('\n!(^ 0 ^)! New Best mAP of res_deconv = {} and all oks metrics is {} \n'.format(best_o,eval_results))
        
        if best_o > result_thres:
            if  eval_results_refine[0] > best :
                best = eval_results_refine[0]
                torch.save(A.module.state_dict(),os.path.join(output_dir,'best_ckpt.tar'))
                logger.info('\n!(^ 0 ^)! New Best mAP = {} and all oks metrics is {} \n'.format(best,eval_results_refine))
            


if __name__ == '__main__':
    main()
