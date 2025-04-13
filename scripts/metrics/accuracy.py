# the accuracy (precision and recall) metrics we are focusing on reproducing
import torch

# mAP


# AP50 and AP75 
class AP(torch.nn.Module):
    def __init__(self,t):
        super(AP, self,t).__init__()
        self.t = t # set the threshold
    def forward():
        """
        computes the average precision 
        """


# AR1

# AR10

# AR100

# AR500