# Copyright (c) SEU - PatternRec
# Licensed under the MIT License.
# Written by Sen Yang (yangsenius@seu.edu.cn)

import torch
import numpy as np
import os

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from collections import defaultdict
from collections import OrderedDict

from timeit import default_timer as timer
import tqdm
import json
import logging
logger = logging.getLogger(__name__)

##################### evaluate #####################3

def evaluate( model , dataset , config, output_dir, with_mask = False, use_refine = False ):

    logger.info("\n=+=+=+=+=+=+=+=+=+=+=+= evalute +==+=+=+=+=+=+=+=+=+=+=+=+=+=+=\n")
    model.eval()

    predict_results =[]

    #####   output results in val-or-test dataset  #########
    with torch.no_grad():

        for id ,(input,image_id ,score ,affine_matrix_inv, bbox, info) in enumerate(dataset):

            start = timer()

            if with_mask: # for eval Resnet_Deconv_Boosting net

                if use_refine:

                    _, _ , heatmap_refine_dt = model(input)
                    heatmap_dt = heatmap_refine_dt

                else: # original dt
                    
                    heatmap_dt, _ , _ = model(input)
                
            else: # for eval simple baseline net

                heatmap_dt, _ = model(input)

            heatmap_dt = heatmap_dt.cpu()
            coordinate_argmax, maxval =  get_final_coord( heatmap_dt, post_processing = True)

            pred_kpt = compute_orignal_coordinate(affine_matrix=affine_matrix_inv, 
                                                    heatmap_coord=coordinate_argmax,
                                                    bounding = bbox)

            pred_kpt[:,:,2] = maxval[:,:]
            pred_kpt = pred_kpt.numpy().round(3).reshape(len(pred_kpt),-1)
            
            index = info['index']

            for i in range(len(input)):

                predict_results.append({
                    'image_id': image_id[i].item(),
                    'keypoints':pred_kpt[i].tolist(),
                    'score': score[i].item(),
                    'index':index[i].item(),
                    'bbox':bbox[i].tolist()
                })

            time = timer() - start

            if id % 100 ==0:
                logger.info("iters[{}/{}], inference-speed: {:.2f} samples/s".format(id,len(dataset),len(input)/time))


    logger.info('\n==> the number of predict_results samples is {} before OKS NMS'.format(len(predict_results)))

    ##########  analysize the oks metric
    image_kpts = defaultdict(list)

    # 1 image_id <==> n person keypoints results
    for sample in predict_results:
        image_kpts[sample['image_id']].append(sample)
    logger.info('==> oks_nms ...')
    logger.info('==> images num: {}'.format(len(image_kpts)))

    ##### oks nms ####
    begin = timer()
    dt_list = []
    for img in image_kpts.keys():

        kpts = image_kpts[img]

        # score = bbox_score * kpts_mean_score
        for sample in kpts:
            bbox_score = sample['score']

            kpt_scores = sample['keypoints'][2::3]
            high_score_kpt = [s for s in kpt_scores if s > config.test.confidence_threshold]
            kpts_mean_score = sum(high_score_kpt)/(len(high_score_kpt)+1e-5)
            sample['score'] = bbox_score * kpts_mean_score

        ## oks nms
        kpts = oks_nms(kpts, config.test.oks_nms_threshold)
        #kpts = oks_nms_sb(kpts, config.test.oks_nms_threshold)

        #if all(a['score']<= 0.2 for a in kpts):
        #    continue  #skip peroson bbox with low-score!

        #score_list = [a['score'] for a in kpts]
        #if sum(score_list)/(len(score_list)+1e-8) <=0.10:
        #    continue

        for sample in kpts:

            image_id = sample['image_id']
            keypoints = sample['keypoints']
            score =sample['score']
            tmp = {'image_id':image_id, 'keypoints':keypoints,'category_id':1,'score':score }

            dt_list.append(tmp)

    logger.info('\n==>  the number of predict_results samples is {} after OKS NMS , consuming time = {:.3f} \n'.format(len(dt_list),timer()-begin))

    eval_reults = coco_eval(config,dt_list,output_dir)

    return eval_reults


def coco_eval(config, dt, output_dir):
    """
    Evaluate the result with COCO API
    """

    gt_path = os.path.join( config.annotation_root_dir,'person_keypoints_val2017.json')

    
    dt_path = os.path.join(output_dir,'dt.json')


    with open(dt_path, 'w') as f:
        json.dump(dt, f)
        
    logger.info('==>dt.json has been written in '+ os.path.join(output_dir,'dt.json'))

    coco = COCO(gt_path)

    coco_dets = coco.loadRes(dt_path)
    coco_eval = COCOeval(coco, coco_dets, "keypoints")
   
    #coco_eval.params.imgIds  =  image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats




###########################  Util Functions    ##########################

def get_max_coord(target):

    '''target = [N,C,H,W]
    heatmap max position to coordinate '''
    N = target.size()[0]
    C = target.size()[1]
    W = target.size()[-1]

    target_index = target.view( N ,C ,-1)
    # max_index = torch.argmax(target_index,1,keepdim =True)  #如果值全为0，会返回最后一个索引，那么要对其置零
    max_value ,  max_index = torch.max (target_index,2) #元组1

    max_index = max_index * torch.tensor(torch.gt(max_value,torch.zeros(size = max_index.size())).numpy(),dtype = torch.long) #
    #如果值全为0，会返回最后一个索引，那么要判断是否大于0对其置零 类型转换

    max_index_view = max_index.view(N,C)

    keypoint_x_y = torch.zeros((N,C,2),dtype = torch.long)

    keypoint_x_y[:,:,0] = max_index_view[:,:] % W # index/48 = y....x 商为行数，余为列数 target.size()[-1]=48
    keypoint_x_y[:,:,1] = max_index_view[:,:] // W # index/48 = y....x 商为行数，余为列数
    #max_value = max_value.view(N,C,1)
    return keypoint_x_y, max_value


def get_final_coord(batch_heatmaps,post_processing = True):
    '''simple baseline coordinates offset'''

    coords, maxvals = get_max_coord(batch_heatmaps)
    heatmap_height = batch_heatmaps.size()[2]
    heatmap_width = batch_heatmaps.size()[3]

    if post_processing:
        for n in range(coords.size()[0]):
            for p in range(coords.size()[1]):
                hm = batch_heatmaps[n][p]
                px = int(np.floor(coords[n][p][0] + 0.5))
                py = int(np.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = torch.tensor([hm[py][px+1] - hm[py][px-1],
                                     hm[py+1][px]-hm[py-1][px]])
                    coords = coords.float()
                    coords[n][p] += torch.sign(diff) * .25

    preds = coords.clone()

    return preds, maxvals

def compute_orignal_coordinate(affine_matrix, heatmap_coord, up = 4, bounding= None):
    '''
    A = [j,n,m]  B = [j,m,p]
    [j,n,p]= [j,n,m] * [j,m,p]

    C = torch.matmul(A,B) = [j,n,p]

    A : `affine_matrix` :[N,3,3]
    B : `orig_coord`:    [N,17,3] -> [N,3,17]
    C : `affine_coord`:      [N,3,3]*[N,3,17] = [N,3,17]  =>[N,17,3]

    return C

    Note: `bounding` is prevent the final coordinates falling outside the image border

    `bounding` = [left_x,up_y, w, h] like `bbox`

    '''

    N = heatmap_coord.size()[0]
    C = heatmap_coord.size()[1]

    heatmap_coord = up*heatmap_coord + 0.5
    orig_coord = torch.ones(N,C,3)

    orig_coord[:,:,0:2] = heatmap_coord[:,:,0:2]
    orig_coord = orig_coord.permute(0,2,1)

    #[N,3,17]    =              [N,3,3]   matmul   [N,3,17]
    affine_coord = torch.matmul(affine_matrix.float(), orig_coord.float())

    affine_coord = affine_coord.permute(0,2,1)
    
    if bounding is not None:
        # restrict the keypoints falling into bbox
        for i in range(len(affine_coord)):
            affine_coord_x = affine_coord[i,:,0].clamp(bounding[i,0],bounding[i,0]+bounding[i,2])
            affine_coord_y = affine_coord[i,:,1].clamp(bounding[i,1],bounding[i,1]+bounding[i,3])
            affine_coord[i,:,0] = affine_coord_x
            affine_coord[i,:,1] = affine_coord_y

    return affine_coord



def oks_nms(kpts_candidates, threshold):
    """
    keypoints : data-format: [17,3]

    `NMS algorithm`
         while(1):
     `1.` take max scores keypoints(index = 0) as groudtruth 'gt', if break or continue

     `2.` supression all other keypoints 'kpt' and pop it from the list, if computeOKS(gt,kpt) > threshold  (note: supression is a inner-while)
            id = index + 1
            while(1):
                if break or continue
                if computeOKS(gt,kpt(id)) > threshold
                   list.pop()
                id +=1

            index +=1

    """
    # data format [51] ->[17,3]
    if np.array(kpts_candidates[0]['keypoints']).shape !=(17,3):
        for q,k in enumerate(kpts_candidates):
            kpts_candidates[q]['keypoints'] = np.array(k['keypoints']).reshape(17,3)

    # sort the list by keypoints scores, from bigger to smaller
    kpts_order = sorted(kpts_candidates, key=lambda tmp:tmp['score'], reverse=True)

    # nms #
    index = 0
    while True:

        if index >=len(kpts_order):
            break

        gt = kpts_order[index]['keypoints']

        bbox = kpts_order[index]['bbox']
        gt_area = bbox[2]*bbox[3]

        id = index + 1
        while True:

            if id >=len(kpts_order):
                break
            kpt = kpts_order[id]['keypoints']

            if ComputeOKS(gt,kpt,gt_area) > threshold:
                kpts_order.pop(id)

            id += 1

        index += 1

    # numpy  [:,17,3] --> list [:,51]
    for q,k in enumerate(kpts_order):
        kpts_order[q]['keypoints'] = np.array(k['keypoints']).reshape(-1).round(3).tolist()

    return kpts_order


def ComputeOKS(dt_keypoints,gt_keypoints,gt_area,invisible_threshold = 0):
    """
    Args:   dt_keypoints = [17,3]
            gt_keypoints = [17,3]

    Return:
            oks = [17]
            sum_oks = sum(oks)/sum(label_gt)

    """
    #print(dt_keypoints,gt_keypoints,gt_area)
    dt_keypoints = np.array(dt_keypoints)
    gt_keypoints = np.array(gt_keypoints)
    sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
    vars = (sigmas * 2)**2
    k = len(sigmas)
    oks = np.zeros(k)
        # compute oks between each detection and ground truth object
    label_num_gt = gt_keypoints[:,2]
    dx =  dt_keypoints[:,0]-gt_keypoints[:,0]
    dy =  dt_keypoints[:,1]-gt_keypoints[:,1]

    oks = (dx**2 + dy**2) / vars / (gt_area+np.spacing(1)) / 2

    #print(oks)
    oks = np.exp(-oks)
    sum_oks = np.sum((label_num_gt>0)*oks)/(sum(label_num_gt>invisible_threshold)+1e-5)
    #print(label_num_gt>0)
    label_num = sum(label_num_gt>0)
    return  np.round(sum_oks,4)


############  simple baseline contrast ######
def oks_iou(g, d, a_g, a_d, sigmas=None, in_vis_thre=None):
    if not isinstance(sigmas, np.ndarray):
        sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
    vars = (sigmas * 2) ** 2
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    ious = np.zeros((d.shape[0]))
    for n_d in range(0, d.shape[0]):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx ** 2 + dy ** 2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if in_vis_thre is not None:
            ind = list(vg > in_vis_thre) and list(vd > in_vis_thre)
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / e.shape[0] if e.shape[0] != 0 else 0.0
    return ious

def oks_nms_sb(kpts_db, thresh, sigmas=None, in_vis_thre=None):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh, overlap = oks
    :param kpts_db
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    if len(kpts_db) == 0:
        return []

    scores = np.array([kpts_db[i]['score'] for i in range(len(kpts_db))])
    kpts = np.array([kpts_db[i]['keypoints'] for i in range(len(kpts_db))])
    areas = np.array([kpts_db[i]['bbox'][2]*kpts_db[i]['bbox'][3] for i in range(len(kpts_db))])

    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]

        keep.append(i)

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]], sigmas, in_vis_thre)

        inds = np.where(oks_ovr <= thresh)[0]
        order = order[inds + 1]
    
    kpts_left=[]

    for ke in keep:
        kpts_left.append(kpts_db[ke])

    return kpts_left

def test():

    a = torch.randn(3,7,48,72).numpy()

    print('xxxxxxxxxxxxxxxxxxxxxx')
    coord , maxval= get_final_coord(torch.tensor(a))
    print(coord)

    q = torch.eye(3)
    q = torch.stack([q,q,q],dim=0)
    c = compute_orignal_coordinate( q , coord)
    c[:,:,2] = maxval[:,:]
    print(c)
    c = c.numpy().reshape(3,-1)
    print(c,maxval)

    c= np.ones((51))

    kpts_candidates = [{"image_id": 397133, "keypoints": [[317.697265625, 178.166015625, 17.687856674194336], [314.943359375, 179.083984375, 9.009297370910645], [319.533203125, 191.017578125, 3.9613606929779053], [314.943359375, 179.083984375, 21.99308204650879], [314.943359375, 179.083984375, 35.71976089477539], [314.943359375, 201.115234375, 32.76402282714844], [314.943359375, 201.115234375, 32.638763427734375], [314.943359375, 201.115234375, 36.89710998535156], [317.697265625, 178.166015625, 15.9923734664917], [317.697265625, 176.330078125, 20.671897888183594], [314.943359375, 201.115234375, 23.281190872192383], [314.943359375, 179.083984375, 21.850238800048828], [314.943359375, 179.083984375, 27.898820877075195], [314.943359375, 201.115234375, 34.260032653808594], [314.943359375, 201.115234375, 21.916322708129883], [314.943359375, 201.115234375, 25.71013641357422], [471.916015625, 211.212890625, 5.586215019226074]], "score": 1, "index": 0,"bbox":[1,2,90,120]},
     {"image_id": 397133, "keypoints": [[-5.916666507720947, 307.02777099609375, 17.212087631225586], [-6.75, 307.3055725097656, 8.803095817565918], [-5.361110687255859, 302.02777099609375, 3.7973849773406982], [-6.75, 307.3055725097656, 21.463369369506836], [-6.75, 307.3055725097656, 34.83835220336914], [-6.75, 305.0833435058594, 31.70177459716797], [-6.75, 307.3055725097656, 31.713335037231445], [-6.75, 307.3055725097656, 35.9440803527832], [-5.916666507720947, 307.02777099609375, 15.585540771484375], [-5.916666507720947, 306.47222900390625, 20.15411376953125], [-6.75, 307.3055725097656, 22.628522872924805], [-6.75, 307.3055725097656, 21.293516159057617], [-6.75, 307.3055725097656, 27.186229705810547], [-6.75, 305.0833435058594, 33.165287017822266], [-6.75, 305.0833435058594, 21.208131790161133], [-6.75, 305.0833435058594, 24.949434280395508], [31.861114501953125, 290.3611145019531, 4.879560470581055]], "score": 1, "index": 1,"bbox":[1,2,90,120]},
     {"image_id": 252219, "keypoints": [[272.1884765625, 314.0478515625, 17.692873001098633], [270.2255859375, 251.8896484375, 9.01416015625], [273.4970703125, 302.2705078125, 3.9639458656311035], [270.2255859375, 251.8896484375, 22.002004623413086], [270.2255859375, 251.8896484375, 35.732444763183594], [270.2255859375, 309.4677734375, 32.774959564208984], [270.2255859375, 309.4677734375, 32.648067474365234], [270.2255859375, 309.4677734375, 36.907997131347656], [272.1884765625, 251.2353515625, 15.998085975646973], [272.1884765625, 249.9267578125, 20.678987503051758], [270.2255859375, 309.4677734375, 23.289644241333008], [270.2255859375, 314.7021484375, 21.85584831237793], [270.2255859375, 314.7021484375, 27.90572166442871], [270.2255859375, 309.4677734375, 34.2704963684082], [270.2255859375, 309.4677734375, 21.925018310546875], [270.2255859375, 309.4677734375, 25.723825454711914], [361.1728515625, 232.9150390625, 5.149745941162109]], "score": 1, "index": 2,"bbox":[1,2,90,120]}
                      ,
    {"image_id": 252219, "keypoints": [[272.1884765625, 314.0478515625, 17.692873001098633], [270.2255859375, 251.8896484375, 9.01416015625], [273.4970703125, 302.2705078125, 3.9639458656311035], [270.2255859375, 251.8896484375, 22.002004623413086], [270.2255859375, 251.8896484375, 35.732444763183594], [270.2255859375, 309.4677734375, 32.774959564208984], [270.2255859375, 309.4677734375, 32.648067474365234], [270.2255859375, 309.4677734375, 36.907997131347656], [272.1884765625, 251.2353515625, 15.998085975646973], [272.1884765625, 249.9267578125, 20.678987503051758], [270.2255859375, 309.4677734375, 23.289644241333008], [270.2255859375, 314.7021484375, 21.85584831237793], [270.2255859375, 314.7021484375, 27.90572166442871], [270.2255859375, 309.4677734375, 34.2704963684082], [270.2255859375, 309.4677734375, 21.925018310546875], [270.2255859375, 309.4677734375, 25.723825454711914], [361.1728515625, 232.9150390625, 5.149745941162109]], "score": 1, "index": 2,"bbox":[1,2,90,120]}
         ,
         {"image_id": 397133, "keypoints": [[200.697265625, 10.166015625, 17.687856674194336], [0.943359375, 0.083984375, 9.009297370910645], [0.533203125, 0.017578125, 3.9613606929779053], [314.943359375, 179.083984375, 21.99308204650879], [314.943359375, 179.083984375, 35.71976089477539], [314.943359375, 201.115234375, 32.76402282714844], [314.943359375, 0.115234375, 0.638763427734375], [0.943359375, 0.115234375, 36.89710998535156], [0.697265625, 0.166015625, 15.9923734664917], [317.697265625, 176.330078125, 20.671897888183594], [314.943359375, 0.115234375, 23.281190872192383], [0.943359375, 179.083984375, 21.850238800048828], [314.943359375, 179.083984375, 27.898820877075195], [0.943359375, 201.115234375, 34.260032653808594], [314.943359375, 201.115234375, 21.916322708129883], [314.943359375, 201.115234375, 25.71013641357422], [471.916015625, 211.212890625, 5.586215019226074]], "score": 1, "index": 0,"bbox":[1,2,90,120]},
            ]
    print(kpts_candidates)
    a = oks_nms(kpts_candidates , 0.5)
    print(a)
    print(len(kpts_candidates))
    print(len(a))
    s = [('yellow',1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]

    d =defaultdict(list)

    for k in kpts_candidates:

       d[k['image_id']].append(k)

    ddd = []
    for k in d.keys():
        a = d[k]
        print(a)
        ddd.append(a)
    print(ddd.shape)




if __name__ ==  '__main__':
    test()
