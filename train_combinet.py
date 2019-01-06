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
from src.combinetworks import Combinetworks

import datetime

def args():
    parser = argparse.ArgumentParser(description='Train combine network')
    parser.add_argument('--cfg',            help='experiment configure file name',  required=True,   default='config.yaml', type=str)
    parser.add_argument('--exp_name',       help='experiment name',        default='combine_train_exp'     , type=str)

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

    ### define two baseline models A and B 
    A = keypoints_output_net(config, is_train= True , num_layers = 34)
    B = keypoints_output_net(config, is_train= True , num_layers = 18)

    logger.info(">>> total params of A: {:.2f}M".format(sum(p.numel() for p in A.parameters()) / 1000000.0))
    logger.info(">>> total params of B: {:.2f}M".format(sum(p.numel() for p in B.parameters()) / 1000000.0))

    seed = torch.randn(1,3,config.model.input_size.h,config.model.input_size.w)

    hmp_a, fmp_a = A(seed)
    hmp_b, fmp_b = B(seed)

    # the size of feature map in A and B must be equal 
    h_cat = torch.cat([hmp_a,hmp_b],dim=1)
    f_cat = torch.cat([fmp_a,fmp_b],dim=1)

    f_channels,f_h,f_w = f_cat.size()[1],f_cat.size()[2],f_cat.size()[3]
    h_channels = h_cat.size()[1]

    C = Combinetworks(f_channels, f_h, f_w, h_channels)
    logger.info(">>> total params of C: {:.2f}M".format(sum(p.numel() for p in C.parameters()) / 1000000.0))

    optimizer_A = torch.optim.Adam(A.parameters(), lr = config.train.lr)
    optimizer_B = torch.optim.Adam(B.parameters(), lr = config.train.lr)
    optimizer_C = torch.optim.Adam(C.parameters(), lr = config.train.lr)

    A = torch.nn.DataParallel(A).cuda()
    B = torch.nn.DataParallel(B).cuda()
    C = torch.nn.DataParallel(C).cuda()

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
    
    best_A,best_B,best_C = 0,0,0
    logger.info("\n=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+= training +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+==")
    for epoch in range(begin, end):
        logger.info('==>training...')
        for iters, (input, heatmap_gt, kpt_visible,  index) in enumerate(train_dataloader):
            
            start = timer()
            optimizer_A.zero_grad()
            optimizer_B.zero_grad()
            input = input.cuda()
            heatmap_gt = heatmap_gt.cuda()
            kpt_visible = kpt_visible.float().cuda()
            A.train()
            B.train()
            
            ha, fa = A(input)
            hb, fb = B(input)

            a_loss , losses_a = loss(ha, heatmap_gt, kpt_visible)
            b_loss , losses_b = loss(hb, heatmap_gt, kpt_visible)

            a_loss.backward()
            b_loss.backward()

            optimizer_A.step()
            optimizer_B.step()

            if best_A > 0.4 and best_B > 0.4:
                # train Combinetwork ,
                # combine the feature maps and heatmaps of A and B's outputs
                C.train()
                h = torch.cat([ha,hb],dim=1).detach()
                f = torch.cat([fa,fb],dim=1).detach()

                # we choose A as our main baseline models
                # we expect C can learn a meta combination weight of A and B
                # to have a better results than A 
                # Here we use the heatmap loss not direct coordiante loss
                # C' objective is making the `meta loss` samller
                # The ideal value of `meta loss` should be smalller than 1, which
                # means C's output meta-heatmap is better than A's heatmap

                meta_heatmap = C(f,h)

                baseline_loss = a_loss.item()+1e-8
                meta_loss,_ = loss(meta_heatmap,heatmap_gt,kpt_visible)
                meta_loss = meta_loss/baseline_loss
                meta_loss.backward()
                optimizer_C.step()

            time = timer() - start

            if best_A > 0.4 and best_B > 0.4:
                with open(os.path.join(output_dir,'all_train_loss.log'),'w') as ff:
                    ff.write(str(np.round(a_loss.item(),2))+','+str(np.round(a_loss.item(),2))+','+str(np.round(meta_loss.item(),2))+'\n')
            else:
                with open(os.path.join(output_dir,'all_train_loss.log'),'w') as ff:
                    ff.write(str(np.round(a_loss.item(),2))+','+str(np.round(a_loss.item(),2))+'\n')

            if iters % 100 == 0:
                if best_A > 0.4 and best_B > 0.4:
                    logger.info(
                    'epoch: {}\t iters:[{}|{}] \t A_loss:{:.4f}\t B_loss:{:.4f}\t feed-speed:{:.2f} samples/s\nMeta_loss:{:.4f}' \
                            .format(epoch,iters,len(train_dataloader),a_loss,b_loss,meta_loss,len(input)/time))
                else:
                    logger.info(
                    'epoch: {}\t iters:[{}|{}] \t A_loss:{:.4f}\t B_loss:{:.4f}\t feed-speed:{:.2f} samples/s' \
                            .format(epoch,iters,len(train_dataloader),a_loss,b_loss,len(input)/time))
        ## eval ##
        eval_results_A = evaluate( A, valid_dataloader , config, output_dir)
        eval_results_B = evaluate( B, valid_dataloader , config, output_dir)

        logger.info('==> mAP of A is: {:.3f}\n'.format(eval_results_A[0]))
        logger.info('==> mAP of B is: {:.3f}\n'.format(eval_results_B[0]))

        torch.save(A.cpu().state_dict(),os.path.join(output_dir,'A_ckpt.tar'))
        torch.save(B.cpu().state_dict(),os.path.join(output_dir,'B_ckpt.tar'))
        
        def if_best(best,eval_results,name):
            
            if  eval_results[0] > best :
                best =  eval_results[0]
                torch.save(eval(name).cpu().state_dict(),os.path.join(output_dir,'best_{}_ckpt.tar'.format(name)))
                logger.info('\n!(^ 0 ^)! New Best mAP of {} = {} and all oks metrics is {} \n'.format(name, best, eval_results[0]))

            return best
        
        best_A = if_best(best_A,eval_results_A,'A')
        best_B = if_best(best_B,eval_results_B,'B')



if __name__ == '__main__':
    main()
