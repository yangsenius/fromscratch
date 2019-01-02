# Copyright (c) SEU - PatternRec
# Licensed under the MIT License.
# Written by Sen Yang (yangsenius@seu.edu.cn)

import torch
import torch.nn as nn

        
class Combinetworks(nn.Module):
    '''
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
      
        self.AtrousRate = 6 # enlarge the receptive field to heatmap
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
            nn.Linear(   512*feature_h*feature_w,  1024,   bias=True),
            nn.Linear(   1024,  conv_filter_param_num ,bias=True)
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

        meta_heatmap = nn.functional.conv2d(heatmaps, Conv_Kernel_parameters,   dilation = self.AtrousRate,
                                                                                padding = self.padding,
                                                                                stride = 1,
                                                                                bias=None)
        return meta_heatmap
        
def test():
        
    f_A, f_B = torch.randn(3,512,12,9),torch.randn(3,512,12,9)
    h_A, h_B = torch.randn(3,17,96,72),torch.randn(3,17,96,72)

    gt_heatmap = torch.randn(3,17,96,72)
    
    AB_F_Cat = torch.cat([f_A,f_B],dim=1).detach()
    AB_H_Cat = torch.cat([h_A,h_B],dim=1).detach()

    feature_channels= AB_F_Cat.size()[1]
    f_h = AB_F_Cat.size()[2]
    f_w = AB_F_Cat.size()[3]

    heatmap_channels = AB_H_Cat.size()[1]
  
    Combine_Network = Combinetworks(feature_channels, f_h, f_w ,heatmap_channels )

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

if __name__ == '__main__':
    test()
