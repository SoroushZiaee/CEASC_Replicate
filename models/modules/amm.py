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
        # hard thresholding here 
