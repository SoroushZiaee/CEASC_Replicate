# script defining AMM 

# imports 
import torch
import numpy as np

class AMM_module(torch.nn.Module):
    def __init__(self):
        super(AMM_module, self). __init__()
        self.modes = ["train","test"]
        self.conv = torch.nn.Conv2d(256,1,3,padding='same') # number of input channels is 256 based on FPN feature channel dimensions and output is one channel based on paper 
    def forward(self,xi,mode):
        if mode not in self.modes:
            raise ValueError(f"Invalid mode. Expected one of: {self.modes}") # make sure training or testing mode for amm is set
        si = self.conv(xi)
        hi = torch.empty_like(si,requires_grad=True) # baseline for the binary mask -- set requires grad as true for now, might not be the case later -- work through this 
        if mode == "train":

        else:
            hard = torch.argwhere(si > 0)
            hi[hard] = 1 # set all the values where si > 0 to 1
            hi[~hard] = 0 # set all the values where si < 0 to 0

        s1 = c1 + torch.nn.functional.gumbel_softmax(c1,dim=1) # make sure this implimentation is the same as the paper and contains learnable parameters as discussed in the paper 
        return hi

class AMM_loss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(AMM_loss).__init__(*args, **kwargs)
    def forward(self,h,label):
        l = [] # will contain the loss for each layer 
        for i in range(len(label)): # for the ground truth mask of each layer of the FPN
            pi = sum(torch.argwhere(label[i]>0))/(label[i].size[1]*label[i].size[2]*label[i].size[3]) # ratio of pixels containing classified objects to total pixels in GT - assumes batch size of one currently
            li = ((sum(torch.argwhere(h[i]>0))/(h[i].size[1]*h[i].size[2]*h[i].size[3]))-pi)**2 # difference in ratio between the label ratio and data ratio
            l.append(li)
        l_amm = sum(l)/len(l)
        return l_amm

