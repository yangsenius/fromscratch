# Copyright (c) SEU - PatternRec
# Licensed under the MIT License.
# Written by Sen Yang (yangsenius@seu.edu.cn)

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class Combinetworks_v_0_1(nn.Module):
    '''
    v 0.1

    Learning to combine two low-confidence keypoints heatmaps to be more accurate by gradient descent 
    Use hypernetworks to predict the weight-parameters of the Convolution filter.

        

    The weight denotes the meta-combination weight of two low-confidence heatmaps, which means
    two model are uncertain about this keypoint position and two peak positions of heatmaps may be 


    Args :
        `feature_channels`:   the number of combine_feature_channels
        `feature_h`       :   the height of combine_feature_map
        `feature_w`       :   the width  of combine_feature_map
        `heatmap_channels`:   the number of combine_heatmap_channels
        `activation`      :   activation for propress

    Input: `combine heatmaps`: [N,keypoints_channel*model_num,h1,w1]
           `combine features`: [N,featuremap_channels*model_num,h2,w2]
    
    Output: `meta_heatmaps` : [N,keypoints_channel,h1,w1]

    note:  heatmaps combine -> meta-heatmap  [17*2,h,w]-> [17,h,w]
        
        `combination_conv` parameters respresent the meta-weight, which means how to combine two heatmaps to get a better results
        but in forward computing, we use the `torch.nn.functional.conv2d` instead of `torch.nn.Conv2d`, because we need to use
        the predicted weights to compute the convolution operation to generate the computing graph

        we use the `hypernetworks` to predicting weights

     '''

    def __init__(self, feature_channels, feature_h, feature_w, heatmap_channels, activation='relu'):
        super().__init__()
      
        self.AtrousRate = 6 # enlarge the receptive field to cover heatmap gaussian kernel
        self.kernel_size = 5 
        self.padding = int(self.AtrousRate*(self.kernel_size-1)/2) # keep heatmap size
        
        self.combination_conv = nn.Conv2d(heatmap_channels,    17,     kernel_size = self.kernel_size,
                                                                dilation = self.AtrousRate,
                                                                padding = self.padding,
                                                                stride = 1,
                                                                bias=False)  # combination does not need bias

        conv_filter_param_num = torch.numel(self.combination_conv.weight.data)

        ################### predicting weight networks  ################
        ###### reduce the dimension of input featuremap channels
        self.activation = nn.ModuleDict([
                                        ['sigmoid', nn.Sigmoid()],
                                        ['relu', nn.ReLU()]  
                                        ])
    
        self.preprocess_feature  = nn.Sequential(

                        nn.Conv2d(feature_channels,1024,   kernel_size = 3, padding=1, stride=1),
                        nn.BatchNorm2d(1024),
                        self.activation['relu'],
                        nn.Conv2d(1024,512,kernel_size=1),
                        nn.BatchNorm2d(512),
                        self.activation['relu'])

        ###################  hypernetworks two-layers linear network  ###########       
        self.hypernetworks = nn.Sequential(
            nn.Linear(   512*feature_h*feature_w,  2048,   bias=True),
            nn.Linear(   2048,  conv_filter_param_num ,bias=True)
             )
        ################################################################

    def forward(self, feature_maps , heatmaps):

        # propress
        feature_embedding = self.preprocess_feature(feature_maps)
        feature_embedding = feature_embedding.view(feature_embedding.size()[0],-1)

        # hpyernetworks
        predict_weights = self.hypernetworks(feature_embedding)

        weight_shape = self.combination_conv.weight.data.size()
        predict_weights = predict_weights.mean(dim=0) # average on the mini-batch
       

        Conv_Kernel_parameters  = predict_weights.view(weight_shape).clone()
        print('shape',Conv_Kernel_parameters.size())
        meta_heatmap = nn.functional.conv2d(heatmaps, Conv_Kernel_parameters,   dilation = self.AtrousRate,
                                                                                padding = self.padding,
                                                                                stride = 1,
                                                                                bias=None)
        return meta_heatmap


def test_v_0_1():
        
    f_A, f_B = torch.randn(3,512,12,9),torch.randn(3,512,12,9)
    h_A, h_B = torch.randn(3,17,96,72),torch.randn(3,17,96,72)

    gt_heatmap = torch.randn(3,17,96,72)
    
    AB_F_Cat = torch.cat([f_A,f_B],dim=1).detach()
    AB_H_Cat = torch.cat([h_A,h_B],dim=1).detach()

    feature_channels= AB_F_Cat.size()[1]
    f_h = AB_F_Cat.size()[2]
    f_w = AB_F_Cat.size()[3]

    heatmap_channels = AB_H_Cat.size()[1]
  
    Combine_Network = Combinetworks_v_0_1(feature_channels, f_h, f_w ,heatmap_channels )

    Meta_heatmap = Combine_Network( AB_F_Cat, AB_H_Cat)

    print('\n############################### combine network ##############################\n\n',Combine_Network)
    print('#######################################################################################')
    print(Meta_heatmap.size())

    loss = torch.sum((Meta_heatmap-gt_heatmap)**2)
    print(loss)
    loss.backward()
    print(loss.is_leaf)
    print(Combine_Network.hypernetworks[0].weight.grad.size())
    print(Combine_Network.combination_conv.weight.grad)

##3######################## v ###################

class combinetworks_v_0_2(nn.Module):
    '1x1 conv'
    def __init__(self,combine_channels, heatmap_channels ):
        super(combinetworks_v_0_2,self).__init__()
        
        # 1x1 conv_kernel_size to refine the keypoints heatmaps
        self.conv = torch.nn.Conv2d(combine_channels, heatmap_channels, kernel_size = 1, stride=1, padding= 0)

    
    def forward(self, prior_mask, keypints_heatmap):

        
        combine = torch.cat([prior_mask,keypints_heatmap],dim=1)
        refine_heatmaps = self.conv(combine)
        
        return refine_heatmaps



class combinetworks_v_1(nn.Module):
    r"""
    In this structure, we design a `combinetworks` ,its' parameters include:

    like 1x1 convolution fiter:
        `conv_weight : [k+1,k,1,1]`
        `conv_bias : [k,1,1]`

    For a given mask heatmap [1,h,w] and kpt_heatmaps [17,h,w] ,
    it just implements elementwise linear operation on different channels, 
    which produces a element-wise map for each keypoint heatmap
    
    In another word, We use this combinetworks to predict a meta-weight `\alpha` and `\beta` ,
    which means combination between mask_heatmap and kpts_heatmap: (we default the `\beta` as `1`)

    So, for each of keypoint heatmap[i] :
        meta_keypoint heatmap[i] = `\alpha[i]` * `mask_heatmap` + `\beta` * `keypoint heatmap[i]`
    

    Input: `Mask_heatmap :[N,1,H,W]`
           `kpts_heatmap :[N,k,H,W]`
    
    Output: `Meta_weight :[N,k,H,W]`
            `Meta_heatmap:[N,k,H,W]`


    where : `combine_heatmap` = torch.cat([`Mask_heatmap`,`kpts_heatmap`],dim = 1) = `[N,k+1,H,W]`

            `Meta_weight` = torch.matmul(`combine_heatmap`,`conv_weight`) + `conv_bias`

            `Meta_heatmap[:,i,:,:]` = `Mask_heatmap[:,1,:,:]` * `Meta_weight[:,i,:,:]` + `kpts_heatmap[:,i,:,:]`

    
    In addition, we can also combine the `image feature` to predict the `Meta_weight`

    Our combinet try to learn the relationship between huamn mask shape and 
    
    skeleton keypoints ,also can learn the Constraint relationship between keypoints


    """

    def __init__(self, combine_channels, heatmap_channels):
        super(combinetworks_v_1,self).__init__()

        conv_w = torch.empty(heatmap_channels, combine_channels, 1, 1)
        conv_b = torch.empty(heatmap_channels)
        nn.init.uniform_(conv_w, a=0,b=1)
        nn.init.uniform_(conv_b, a=0,b=1)

        self.combination_conv_w = Parameter(conv_w)
        self.combination_conv_b = Parameter(conv_b)

    def forward(self, kpts_heatmaps , mask_heatmap):
        
        combine_heatmaps = torch.cat([mask_heatmap,kpts_heatmaps],dim=1)
        #                                   [N,K+1,H,W]         [K,K+1,1,1]             [K,1,1]
        meta_weight = nn.functional.conv2d(combine_heatmaps, self.combination_conv_w, bias = self.combination_conv_b)

        alpha = meta_weight
        beta = 1
        
        meta_heatmaps = mask_heatmap * alpha + kpts_heatmaps * beta

        return meta_heatmaps

def test_v1():

    mask = torch.randn(3,1,12,8)
    kpts = torch.randn(3,17,12,8)

    combine_net = combinetworks_v_1(18,17)

    meta_kpt = combine_net(kpts,mask)
    print(meta_kpt.size())

if __name__ == '__main__':
    test_v1()
