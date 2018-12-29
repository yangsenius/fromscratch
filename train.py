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


logging.basicConfig(filename='train.log',format = '%(asctime)s - %(name)s: L%(lineno)d - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
logging.getLogger('').addHandler(console)


def args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--cfg',            help='experiment configure file name',  required=True,   default='config.yaml', type=str)
    parser.add_argument('--exp_name',       help='experiment name',                                                         type=str)

    args = parser.parse_args()
    return args

def main():
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

    logger.info('\navailable GPU numbers {}'.format(torch.cuda.device_count()))

    config_file= args().cfg
    config = edict( yaml.load( open(config_file,'r'))) 
    
    A = keypoints_output_net(config, is_train=True , num_layers = 50)
    
   
    A = torch.nn.DataParallel(A).cuda()
    
    optimizer = torch.optim.Adam(A.parameters(), lr = config.train.lr)
    
    loss = MSELoss()
    
    train_dataset = cocodataset(  config.images_root_dir,
                            config.annotation_root_dir,
                            mode='train',
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                            ]))
                            
    valid_dataset = cocodataset(config.images_root_dir,
                            config.annotation_root_dir,
                            mode='val',
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                            ]))

    Val_dt_dataset =cocodataset(config.images_root_dir,
                            config.person_detection_results_path,
                            mode='dt',
                            dataset = 'val',
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                            ])) 
                            
    
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = config.train.batchsize, shuffle = True , num_workers = 4 , pin_memory=True )
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size = config.test.batchsize, shuffle = False , num_workers = 4 , pin_memory=True )
    
    
    begin, end = config.train.epoch_begin, config.train.epoch_end
    results = evaluate( A, valid_dataloader , config)
    best = results[0]

    logger.info("\n=+=+=+=+=+=+=+=+=+= training +=+=+=+=+=+=+=+=+=+==")
    for epoch in range(begin, end):

        for iters, (input, heatmap_gt, kpt_visible,  index) in enumerate(train_dataloader):
            
            start = timer()
            optimizer.zero_grad()
            input = input.cuda()
            heatmap_gt = heatmap_gt.cuda()
            kpt_visible = kpt_visible.float().cuda()
            A = A.train()
            heatmap_dt, _ = A(input)

            backward_loss , losses = loss(heatmap_dt, heatmap_gt, kpt_visible)

            backward_loss.backward()
            optimizer.step()
            time = timer() - start

            if iters % 100 == 0:
               
               logger.info('epoch: {}\t iters:[{}|{}]\t loss:{:.4f}\t feed-speed:{:.2f} samples/s'.format(epoch,iters,len(train_dataloader),backward_loss,len(input)/time))
       
        eval_results = evaluate( A, valid_dataloader , config)
        logger.info('==> mAP : {:3f}'.format(eval_results[0]))
        torch.save(A.state_dict(),'chpt.pt')
        if  eval_results[0] > best :
            best = eval_results[0]
            logger.info('\n!(^ 0 ^)! Best AP = {}\n'.format(best))
            torch.save(A.state_dict(),'best_chpt.pt')


if __name__ == '__main__':
    main()