# Copyright (c) SEU - PatternRec
# Licensed under the MIT License.
# Written by Sen Yang (yangsenius@seu.edu.cn)

import torch
import numpy as np 

# data control-flow mode
#@torch.jit.script
class MSELoss(torch.nn.Module):

    def __init__(self,):
       super(MSELoss,self).__init__()
        
    def forward(self,preds, heatmap_gt, weight):

        losses = weight*((preds-heatmap_gt)**2).sum(dim=3).sum(dim=2)
        back_loss = losses.mean(dim=1).mean(dim=0)
        return   back_loss  , losses

#trace_MSEloss = torch.jit.trace(MSELoss,(torch.randn(2,2,2,2),torch.randn(2,2,2,2),torch.ones(2)))



def test():

    preds = torch.randn(3,7,96,72)
    heatmap_gt = torch.randn(3,7,96,72)
    weight = np.ones((3,7))
    MSELoss = MSELoss()
    #traced_loss,losses = torch.jit.trace(MSELoss, (preds , heatmap_gt, weight ))
    loss, losses  = MSELoss(preds , heatmap_gt, weight )
    print(loss,'\n',losses)

if __name__ == '__main__':
    test()
    