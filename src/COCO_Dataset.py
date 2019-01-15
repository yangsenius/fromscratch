# Copyright (c) SEU - PatternRec
# Licensed under the MIT License.
# Written by Sen Yang (yangsenius@seu.edu.cn)

import torch
import torch.nn as nn
import torchvision
import os
import numpy as np
import cv2
import logging
import json

from pycocotools.coco import COCO
from pycocotools import mask 
from skimage.filters import gaussian

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class cocodataset(Dataset):

    def __init__(self, config, images_dir, annotions_path,  mode='train', 
                                                            transform = None, 
                                                            dataset = None,
                                                            augment=False,
                                                            **kwargs):
        super(cocodataset,self).__init__()
        self.mode = mode
        logger.info("\n+=+=+=+=+= dataset:{} +=+=+=+=+=".format(mode))
        self.score_threshold = config.test.bbox_score_threshold

        self.input_size =  (config.model.input_size.w ,config.model.input_size.h) # w,h
        self.heatmap_size = (config.model.heatmap_size.w ,config.model.heatmap_size.h)

        self.transform = transform
        self.margin_to_border = config.model.margin_to_border  # input border/bbox border

        ## data augmentation setting
        self.augment = augment
        if self.augment:
            self.aug_scale = config.train.aug_scale
            self.aug_rotation = config.train.aug_rotation
            self.flip = config.train.aug_flip
            logger.info('augmentation is used for training, setting :scale=1±{},rotation=0±{},flip={}(p=0.5)'
                            .format(self.aug_scale,self.aug_rotation,self.flip))
        else:
            logger.info('augmentation is not used ')

    
        if self.mode != 'dt':
            ## load train or val groundtruth data
            self.images_root = os.path.join(images_dir ,mode +'2017')

            self.coco=COCO( os.path.join(annotions_path,'person_keypoints_{}2017.json'.format(mode)))   # train or val
            self.index_to_image_id = self.coco.getImgIds()
            
            self.data = self.get_gt_db()
            cats = [cat['name']
                    for cat in self.coco.loadCats(self.coco.getCatIds())]
            self.classes = ['__background__'] + cats
            logger.info("==> classes:{}".format(self.classes))
            logger.info("dataset:{} , total images:{}".format(mode,len(self.index_to_image_id)))

            self.sigma_factor = config.train.heatmap_peak_sigma_factor
             # From COCO statistics to determine the sigma for each keypoint's heatmap gaussian
            self.sigmas = self.sigma_factor *np.array(
                    [1.0,1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])                                                                                      
            logger.info('@ Standard deviation of gaussian kernel for different keypoints heatmaps is:\n==> {} '
                                                .format(self.sigmas))

        if self.mode == 'dt':
            ## load detection results
            self.images_root = os.path.join(images_dir,dataset +'2017')
            self.annotations = json.load( open(annotions_path,'r')) # dt.json
            self.data = self.get_dt_db()

        logger.info("dataset:{} , total samples: {}".format(mode,len(self.data)))
        

    def get_image_path(self,file_name):
        image_path = os.path.join(self.images_root,file_name)
        return image_path

    def get_dt_db(self,):
        "get detection results"

        score_threshold = self.score_threshold
        container = []
        index = 0
        logger.info("=> total bbox: {}".format(len(self.annotations)))
        for ann in self.annotations:

            image_id = ann['image_id']
            category_id = ann['category_id']
            bbox = ann['bbox']
            bbox[0],bbox[1],bbox[2],bbox[3] = int(bbox[0]),int(bbox[1]),int(bbox[2])+1,int(bbox[3])+1
            score = ann['score']
            if score < score_threshold or category_id != 1:
                continue
            file_name = '%012d.jpg' % image_id

            container.append({
                    'bbox':bbox,
                    'score':score,
                    'index':index,
                    'image_id':image_id,
                    'file_name':file_name,

                })
            index = index + 1

        return container


    def get_gt_db(self,):
        "get groundtruth database"

        container = []
        index = 0
       
        for image_id in self.index_to_image_id:

            img_info = self.coco.loadImgs(image_id)[0]
            width = img_info['width']
            height = img_info['height']
            file_name = img_info['file_name']

            annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id,iscrowd=None))

            bbox_index = 0
            for ann in annotations:

                keypoints = ann['keypoints']
                if ann['area'] <=0 or  max(keypoints)==0 or ann['iscrowd']==1:
                    continue
                bbox = ann['bbox']
                
                rle = mask.frPyObjects(ann['segmentation'], height, width)
                seg_mask = mask.decode(rle)    
                

                bbox =bbox_rectify(width,height,bbox,keypoints)

                container.append({
                    'bbox':bbox,
                    'keypoints':keypoints,
                    'index':index,
                    'bbox_index':bbox_index,
                    'image_id':image_id,
                    'file_name':file_name,
                    'mask':seg_mask,

                })

                index = index + 1
                bbox_index = bbox_index + 1
        
        return container

    def make_affine_matrix(self, bbox, target_size, margin=1, aug_rotation= 0, aug_scale=1):
        """
        transform bbox ROI to adapat the net-input size
        
        t: 3x3 matrix, means transform BBOX-ROI to input center-roi

        rs: 3x3 matrix , means augmentation of rotation and scale

        """

        (w,h)=target_size
        scale = min((w/margin) /bbox[2],
                    (h/margin) /bbox[3])
        #choose small-proportion side to make scaling
        t = np.zeros((3, 3))
        offset_X= w/2 - scale*(bbox[0]+bbox[2]/2)
        offset_Y= h/2 - scale*(bbox[1]+bbox[3]/2)
        t[0, 0] = scale
        t[1, 1] = scale
        t[0, 2] = offset_X
        t[1, 2] = offset_Y
        t[2, 2] = 1

        theta = aug_rotation*np.pi/180
        alpha = np.cos(theta)*aug_scale
        beta = np.sin(theta)*aug_scale
        rs = np.zeros((3,3))
        rs[0, 0] = alpha
        rs[0, 1] = beta
        rs[0, 2] = (1-alpha)*(w/2)-beta*(h/2)
        rs[1, 0] = -beta
        rs[1, 1] = alpha
        rs[1, 2] = beta *(w/2) + (1-alpha)*(h/2)
        rs[2, 2] = 1
        
        # matrix multiply
        # first: t , orignal-transform
        # second: rs, augment scale and augment rotation
        final_t = np.dot(rs,t)
        return final_t

    def kpt_affine(self,keypoints,affine_matrix):
        '[17*3] ==affine==> [17,3]'

        keypoints = np.array(keypoints).reshape(-1,3)
        for id,points in enumerate(keypoints):
            if points[2]==0:
                continue
            vis = points[2] # python value bug
            points[2] = 1
            keypoints[id][0:2] = np.dot(affine_matrix, points)[0:2]
            keypoints[id][2] = vis 

            if keypoints[id][0]<=0 or (keypoints[id][0]+1)>=self.input_size[0] or \
                    keypoints[id][1]<=0 or (keypoints[id][1]+1)>=self.input_size[1]:
                keypoints[id][0]=0
                keypoints[id][1]=0
                keypoints[id][2]=0

        return keypoints

    def make_gt_heatmaps(self,keypoints):
        """
        Generate `gt heatmaps` from keypoints coordinates 

        We can generate adaptive `kernel size` and `peak value` of `gaussian distribution`

        """
        
        heatmap_gt = np.zeros((len(keypoints),self.heatmap_size[1],self.heatmap_size[0]),dtype=np.float32)
        kpt_visible = np.array(keypoints[:,2])

        downsample = self.input_size[1] / self.heatmap_size[1]

        for id,kpt in enumerate(keypoints):
            if kpt_visible[id]==0:
                continue
            if kpt_visible[id]==1 or kpt_visible[id]==2:  # 1: label but invisible 2: label visible

                gt_x = min(int((kpt[0]+0.5)/downsample), self.heatmap_size[0]-1)
                gt_y = min(int((kpt[1]+0.5)/downsample), self.heatmap_size[1]-1)

                #sigma_loose = (2/kpt_visible[id])  # loose punishment for invisible label keypoints: sigma *2
                heatmap_gt[id,gt_y,gt_x] = 1
                heatmap_gt[id,:,:] = gaussian(heatmap_gt[id,:,:],sigma=self.sigmas[id])#*sigma_loose)
                amx = np.amax(heatmap_gt[id])
                heatmap_gt[id] /= amx  # make the max value of heatmap equals 1
                loose = 2/kpt_visible[id] # loose = 2: loose punishment for invisible label keypoints
                heatmap_gt[id] /= loose 

        kpt_visible = kpt_visible > 0
        kpt_visible = kpt_visible.astype(np.float32)
        
        return heatmap_gt, kpt_visible

    def __len__(self,):

        return len(self.data)

    def __getitem__(self,id):

        data = self.data[id]
        keypoints = data['keypoints'] if 'keypoints' in data else ''
        bbox =      data['bbox']
        score =     data['score'] if 'score' in data else 1
        index =     data['index']
        file_name = data['file_name']
        image_id =  data['image_id']
        mask     =  data['mask'] if 'mask' in data else None

        image_path = self.get_image_path(file_name)
        #(h,w,3)
        input_data = cv2.imread( image_path )

        #if self.mode != 'train':
            # when inferencing, 
            # we expect the input without the interference from outside bbox
        #    input = input_data.copy()
        #    input_data = np.zeros_like(input,dtype=np.uint8)
        #    input_data[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2],:] = \
        #                            input[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2],:]

        if self.mode == 'train' and self.augment == True:
            s = self.aug_scale # 0.3
            r = self.aug_rotation #30^.
            aug_scale = np.clip(np.random.randn()*s + 1, 1 - s, 1 + s)
            aug_rotation = np.clip(np.random.randn()*r, -r, r) 

        else:
            aug_scale = 1
            aug_rotation = 0
            self.margin_to_border = 1.1

        affine_matrix = self.make_affine_matrix(bbox,self.input_size,
                                                margin = self.margin_to_border,
                                                aug_scale=aug_scale,
                                                aug_rotation=aug_rotation)

        input_data = cv2.warpAffine(input_data,affine_matrix[[0,1],:], self.input_size,)

        if mask is not None:

            mask = cv2.warpAffine(mask      ,affine_matrix[[0,1],:], self.input_size) #note! mask:(h_,w_,1)->(h,w)
            mask = cv2.resize(mask,self.heatmap_size)
            mask = mask.astype(np.float32)
            mask = mask[np.newaxis,:,:] #(1,h,w,x) or(1,h,w)

            ## mask may be divided into several parts( ndim =4)
            ## we make them into a singel heatmap
            if mask.ndim == 4:
                mask = np.amax(mask,axis=3)

        if self.transform is not None:
            #  torchvision.transforms.ToTensor() :
            #  (H,W,3) range [0,255] numpy.ndarray  ==> (c,h,w) [0.0,1.0] torch.FloatTensor
            input_data = self.transform(input_data)

        if self.mode == 'train':

            keypoints = self.kpt_affine(keypoints,affine_matrix)

            ## flip with 0.5 probability
            if self.augment and self.flip and np.random.random() <= 0.5 and self.transform is not None:

                input_data = torch.flip(input_data,[2])
                mask = torch.from_numpy(mask)
                mask = torch.flip(mask,[2]).numpy()
                
                keypoints[:,0] = self.input_size[0] - 1 - keypoints[:,0] 

            heatmap_gt, kpt_visible = self.make_gt_heatmaps(keypoints)
            
            info = {
                'index':index,
                'prior_mask':mask
            }

            return input_data , heatmap_gt, kpt_visible, info

        if self.mode == 'val' or self.mode =='dt':

            keypoints = self.kpt_affine(keypoints,affine_matrix)
          
            info = {
                'index':index,
                'prior_mask':mask
            }

            return input_data , image_id  , score, np.linalg.inv(affine_matrix), np.array(bbox), info #inverse

def bbox_rectify(width,height,bbox,keypoints,margin=5):
        """
        use bbox_rectify() function to let the bbox cover all visible keypoints
        to reduce the label noise resulting from some visible (or invisible) keypoints not in bbox
        """
        kps = np.array(keypoints).reshape(-1, 3)   #array([[x1,y1,1],[],[x17,y17,1]]]
        # for label ：      kps[kps[:,2]>0]
        # for visibel label：  kps[kps[:,2]==2]
        border = kps[kps[:,2] >=1 ]
        if sum(kps[:,2] >=1) > 0:
            a, b = min(border[:,0].min(),bbox[0]), min(border[:,1].min(), bbox[1])
            c, d = max(border[:,0].max(),bbox[0]+bbox[2]), max(border[:,1].max(),bbox[1]+bbox[3])
            assert abs(margin)<20 ,"margin is too large"
            a,b,c,d=max(0,int(a-margin)),max(0,int(b-margin)),min(width,int(c+margin)),min(height,int(d+margin))

            return [a,b,c-a,d-b]
        else:
        	return bbox



def test():

    import yaml
    from easydict import EasyDict as edict
    config = edict( yaml.load( open('../config.yaml','r')))

    dataset_root_dir = '../../data/coco/images/'
    annotation_root_dir = '../../data/coco/annotations/'
    person_detection_results_path = '../../data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json'

    torch_transform = torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            #torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.2, hue=0.1),
                            #torchvision.transforms.Normalize(
                            #       mean=[0.485, 0.456, 0.406],
                            #        std=[0.229, 0.224, 0.225])
                            ])

    '''dataset = cocodataset(  dataset_root_dir,
                            person_detection_results_path,
                            mode = 'dt',
                            dataset = 'val',
                            transform = torch_transform)'''
    dataset = cocodataset(  config,
                            dataset_root_dir,
                            annotation_root_dir,
                            mode = 'train',
                            #dataset = 'val',
                            transform = torch_transform)

    print('dataset size:',len(dataset))



    tup = dataset[34]  #112234
    print(tup)

    x = tup[0]
    print(tup[2])
    print(x.dtype)
    if x.dtype == torch.float32:
        x = np.array(x).transpose(1,2,0)
    print(x)




if __name__ == '__main__':
    test()
