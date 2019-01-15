from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_Deconv(nn.Module):

    def __init__(self, block, layers,   out_channels, cfg, 
                                        use_mask = False, 
                                        use_feature = False, 
                                        **kwargs):
        self.inplanes = 64
       # extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = False

        super(ResNet_Deconv, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        layer_num = 3
        deconv_channels = [256,256,256]
        deconv_kernel_size = [4,4,4]

        
        deconv_channels_mask = [256,128,64]

        deconv_layers = []
        final_layers = []
        # used for deconv layers for keypoints heatmaps
        self.deconv_layers_kpt = self._make_deconv_layer(
            layer_num,               # layers
            deconv_channels,   # channels
            deconv_kernel_size  )       # kernel_size

        self.inplanes = 512 * block.expansion 

        deconv_layers.append(self.deconv_layers_kpt)

        self.final_layer_kpt = nn.Conv2d(
            in_channels= deconv_channels[-1],
            out_channels = out_channels,
            kernel_size= 1,
            stride=1,
            padding =0 ) #if extra.FINAL_CONV_KERNEL == 3 else 0

        final_layers.append(self.final_layer_kpt)

        ## add
        self.use_feature = use_feature
        self.use_mask = use_mask

        if use_mask:
            mask_channel = 1
            self.deconv_layers_mask = self._make_deconv_layer(
                layer_num,               # layers
                deconv_channels_mask,   # channels
                deconv_kernel_size  )       # kernel_size

            deconv_layers.append(self.deconv_layers_mask)

            self.final_layer_mask = nn.Conv2d(
                in_channels= deconv_channels_mask[-1],
                out_channels = mask_channel,
                kernel_size= 1,
                stride=1,
                padding =0)  #if extra.FINAL_CONV_KERNEL == 3 else 0
            final_layers.append(self.final_layer_mask)

        self.deconv_layers = deconv_layers
        self.final_layers = final_layers

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        '''
        Reference: A Simple baseline for human pose estimation
        Written by Bin Xiao (Bin.Xiao@microsoft.com)
        '''
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        '''
        Reference: A Simple baseline for human pose estimation
        Written by Bin Xiao (Bin.Xiao@microsoft.com)
        '''
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)   
        low_feature = x.clone()

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        
        kpt = self.deconv_layers_kpt(x)
        kpt = self.final_layer_kpt(kpt)
        
        if self.use_mask :
            mask = self.deconv_layers_mask(x)
            mask = self.final_layer_mask(mask)

            if self.use_feature:
                return kpt , mask , low_feature
            else:
                return kpt , mask
        else:
            return kpt

    def init_weights(self, use_pretrained=False, pretrained=''):
        
        for i in self.deconv_layers:
            logger.info('=> init deconv weights from normal distribution')
            for name, m in i.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            

        for i in self.final_layers:
            logger.info('=> init final conv weights from normal distribution')
            for m in i.modules():
                if isinstance(m, nn.Conv2d):
                    
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)
        if use_pretrained == True:
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            logger.info('=> no imagenet pretrained !')
            


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


###  we design this class as `torch.jit.ScriptModule`

class metabooster(torch.jit.ScriptModule):
    """
    a structure to boost or refine the `target heatmap` with `extra information` 

    For human pose estimation :
    
    `target heatmap` is `keypoint_heatmaps`
    `extra information` is `mask_heatmap` or  `image_feature`

    How to use the heatmap max value of each channel?

    `high max-value channels` means `certain keypoints`, 
    which `metabooster` will `reserve  more self-information` and  `receive less extra information`

    `low max-value channe`l means `uncertain keypoints`, 
    which `metabooster` will `reserve  less self-information` and  `receive more extra information`
        
    
    """
    def __init__(self, target_dim, extra_dim ):
        super(metabooster,self).__init__()
        self.conv = torch.jit.trace(
            # 1x1 conv
            nn.Conv2d( target_dim + extra_dim, target_dim, kernel_size = 1 ) , 
            torch.randn(1,target_dim + extra_dim,4,4) )

        self.sigmoid = nn.Sigmoid() # output range [0,1]
        self.softmax = nn.Softmax2d()

    @torch.jit.script_method
    def forward(self, target , extra ):
        # note: extra information maybe include image feature

        
        combine = torch.cat([target,extra],dim=1)

        extra_info = self.sigmoid(self.conv(combine))

        # like np.amax and keepdims
        target_max = torch.max(target.view(target.size(0), target.size(1),-1), 2, keepdim=True)[0]
        target_max = target_max.unsqueeze(-1).repeat(1,1,target.size(2),target.size(3)) # keep same size as target  #

        # because the peak-value ranges from [0,1] we use sqrt() to overcome the decay
        target_reserve_weight = torch.sqrt(target_max).clamp(min=0,max=1)
        
        target_receive_weight = torch.ones_like(target) - target_reserve_weight #

        alpha = target_reserve_weight
        beta  = target_receive_weight

        target_refine = target * alpha +  extra_info * beta # *extra

        #target_refine = self.softmax(target_refine)

        return  target_refine

class StackMetaBooster(torch.jit.ScriptModule): 
    """
    stack the `metabooster` Module

    """
    def __init__(self,target_dim, extra_dim, stack_nums):
        super(StackMetaBooster,self).__init__()
        self.stack_nums = stack_nums
        self.stack_metaboosters = nn.ModuleList([ 
            metabooster(target_dim, extra_dim) 
            for i in range(stack_nums)])

    def forward(self, target , extra):
        target_list = []

        for i in range(self.stack_nums):
            target = self.stack_metaboosters[i](target , extra=extra)
            target_list.append(target)

        return target

def test():

    torch.utils.backcompat.broadcast_warning.enabled=True
        # https://pytorch.org/docs/stable/notes/broadcasting.html#backwards-compatibility
    kpt = torch.randn(2,7,4,4)
    mask = torch.randn(2,1,4,4)
    img_f = torch.randn(2,64,4,4)
    extra = torch.cat([mask,img_f],dim=1)

    boosting = StackMetaBooster(7,65,4)

    target = boosting(kpt,extra)

    print(torch.argmax(kpt,1))
    print(torch.max(kpt))
    print(torch.argmax(target,1))
    print(torch.max(target))

class ResNet_Deconv_Boosting(torch.jit.ScriptModule):
    """
    1. Use image to predict the `coarse keypoints heatmaps` and `body mask`
    2. Concencate `the low-level image feature` and `mask` information 
                    and then process them to get `extra information `by a convolution filter 
    3. Concencate the `extra information` and `keypoints heatmap` and send them to the `boosting` part

    4. output: `kpt_heatmaps`,`mask`,`refine_kpt_heatmap`

    """

    def __init__(self,block, layers, out_channels, cfg, target_dim, extra_dim, stack_nums,  **kwargs):

        super(ResNet_Deconv_Boosting,self).__init__()

        assert cfg.model.extra_mask_flag == True,'mask infomation should be taken into consideration' 

        self.resnet_deconv = ResNet_Deconv(block, layers, out_channels, cfg,  **kwargs)
        '''self.resnet_deconv = torch.jit.trace( 
            ResNet_Deconv(block, layers, out_channels, cfg,  **kwargs),
            torch.randn(1,3,256,256)
            )'''
        
        self.boosting = nn.ModuleDict({

            'process_convolution':nn.Conv2d(extra_dim, extra_dim, kernel_size = 5, padding = 2),
            'process_activation':nn.Sigmoid(),
            'StackMetaBooster': StackMetaBooster(target_dim,extra_dim,stack_nums)})

    def forward(self,x):

        kpts , mask , feature = self.resnet_deconv(x)

        # 'feature map size must be consistent with mask heatmap size in (n,*,h,w)'
        extra = torch.cat([mask,feature],dim=1)

        extra_pro = self.boosting['process_convolution'](extra )
        extra_pro = self.boosting['process_activation'](extra_pro)
        refine_kpts = self.boosting['StackMetaBooster'](kpts,extra_pro)

        return kpts, mask , refine_kpts



def Kpt_and_Mask_Boosting_Net(cfg, is_train, num_layers,  **kwargs):
    
    block_class, layers = resnet_spec[num_layers]
    
   

    out_channels = cfg.model.keypoints_num
    extra_dim = cfg.model.extra_mask_channels + 64 # extra_feature_channels, from resent-config
    stack_nums = cfg.model.booster_stacks

    model = ResNet_Deconv_Boosting(block_class, layers, out_channels, cfg,
                                            out_channels, extra_dim, stack_nums,  
                                            use_mask = cfg.model.extra_mask_flag,
                                            use_feature = cfg.model.extra_feature_flag,
                                            **kwargs)

    if is_train and cfg.model.init_weights:
        model.resnet_deconv.init_weights(pretrained=cfg.model.pretrained_path)

    return model



if __name__ =='__main__':
    test()