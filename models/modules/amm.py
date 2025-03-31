# script defining AMM

# imports
import torch
import numpy as np


class AMM_module(torch.nn.Module):
    def __init__(self):
        super(AMM_module, self).__init__()
        self.modes = ["train", "val"]
        self.conv = torch.nn.Conv2d(
            256, 1, 3, padding="same"
        )  # number of input channels is 256 based on FPN feature channel dimensions and output is one channel based on paper
        self.temp = 1  # temperature on Gumble-Softmax
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,xi,mode):
        if mode not in self.modes:
            raise ValueError(
                f"Invalid mode. Expected one of: {self.modes}"
            )  # make sure training or testing mode for amm is set
        si = self.conv(xi)
        if mode == "train":
            # This returns a non-learnable, CPU-based tensor, possibly on the wrong device, and disconnected from PyTorch's autograd.
            # g1 = torch.from_numpy(np.random.gumbel(size=si.size())) # make random Gumbel noise
            # g2 = torch.from_numpy(np.random.gumbel(size=si.size()))
            g1 = -torch.empty_like(si).exponential_().log()
            g2 = -torch.empty_like(si).exponential_().log()

            hi_soft = self.sigmoid(
                (si + g1 - g2) / self.temp
            )  # modulate si by the noise and push through the sigmoid function
            hi_hard = (
                hi_soft >= 0.5
            ).float()  # make a binary mask from soft values -- hard thresholding
            # hi[hard] = 1
            # hi[~hard] = 0
        else:
            hi_soft = torch.empty_like(si)  # baseline for the binary mask
            hi_hard = (si > 0).float()
            # hi_hard[hard] = 1  # set all the values where si > 0 to 1
            # hi_hard[~hard] = 0  # set all the values where si < 0 to 0

        # Return both hard and soft masks for calculating Loss
        return hi_hard, hi_soft


class AMM_loss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(AMM_loss).__init__(*args, **kwargs)

    def forward(self, h, label):
        l = []  # will contain the loss for each layer
        for i in range(
            len(label)
        ):  # for the ground truth mask of each layer of the FPN
            pi = sum(label[i] > 0) / (
                label[i].size[0] * label[i].size[2] * label[i].size[3]
            )  # ratio of pixels containing classified objects to total pixels in GT - now works with multiple batches by just including them in the calculation
            li = (
                (sum(h[i] > 0) / (h[i].size[0] * h[i].size[2] * h[i].size[3])) - pi
            ) ** 2  # difference in ratio between the label ratio and data ratio
            l.append(li)
        l_amm = sum(l) / len(l)
        return l_amm
