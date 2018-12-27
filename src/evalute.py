
import torch
import torch
import numpy as np
import logging
logger = logging.getLogger(__name__)

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

    max_value = max_value.view(N,C,1)
    
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


def evalute(model,dataset):

    model.eval()

    with torch.no_grad():

        for id ,(input,image_id ,index ,affine_matrix) in enumerate(dataset):

            heatmap_dt, _ = model(input)

            heatmap_dt = heatmap_dt.cpu()
            coordinate_argmax, maxval =  get_final_coord( heatmap_dt, post_processing = True)

            logger.info(coordinate_argmax, maxval, affine_matrix)



def main():
    
    a = torch.randn(2,2,48,72).numpy()
    print(a)
    preds , maxvals = get_max_preds(a)
    preds1, maxvals1 = get_max_coord(torch.tensor(a))
    print('preds',preds,'\nmaxvals',maxvals)
    print('preds1',preds1,'\nmaxvals1',maxvals1)
    print('xxxxxxxxxxxxxxxxxxxxxx')
    print(get_final_coord(torch.tensor(a)))
    print(get_final_preds(a))

if __name__ ==  '__main__':
    main()