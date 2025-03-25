# script defining AMM 

# imports 
import torch
import numpy as np

class AMM_module(torch.nn.Module):
    def __init__(self):
        super(AMM_module, self). __init__()
        self.conv = torch.nn.Conv2d(3,10,3,padding='same') # assuming input has 3 channels (e.g., RGB) and there are 10 classes - we need one binary mask per class 
    def forward(self,x):
        c1 = self.conv(x)
        s1 = c1 + torch.nn.functional.gumbel_softmax(c1,dim=1)
        hard = torch.argwhere(s1 < 0.5)
        hi = torch.empty_like(s1,requires_grad=True) # baseline for the binary mask 
        hi[hard] = 0 # set all the values where s1 is less than 0.5 to 0
        hi[~hard] = 1 # set all the values where s1 is >= 0.5 to 1 
        return hi

class AMM_loss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(AMM_loss).__init__(*args, **kwargs)
    def forward(self,h,label):
        l = [] # will contain the loss for each layer 
        for i in range(len(label)): # for the ground truth mask of each layer of the FPN
            pi = sum(torch.argwhere(label[i]>0))/(label[i].shape[1]*label[i].shape[2]*label[i].shape[3]) # ratio of pixels containing classified objects to total pixels in GT - assumes batch size of one currently
            li = ((sum(torch.argwhere(h[i]>0))/(h[i].shape[1]*h[i].shape[2]*h[i].shape[3]))-pi)**2 # difference in ratio between the label ratio and data ratio
            l.append(li)
        l_amm = sum(l)/len(l)
        return l_amm

